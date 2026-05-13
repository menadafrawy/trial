"""Fine-tune the handwriting model on human-accepted stroke samples."""
import sys
import threading
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Callable

# Make ml/src importable from api/src/
_ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from src.models.rnn import HandwritingRNN
from src.data.preprocessing import stroke_coords_to_offsets, normalize
from src.training.loss import gaussian_mixture_loss

logger = logging.getLogger(__name__)

_MAX_STROKE_LEN = 1200
_MAX_TEXT_LEN = 2

_lock = threading.Lock()
status: dict = {"running": False, "progress": "idle", "last_result": None}


def get_status() -> dict:
    return dict(status)


def _absolute_to_offsets(absolute_strokes: list[list[float]]) -> np.ndarray:
    """Convert absolute coordinate strokes back to normalised offset format."""
    arr = np.array(absolute_strokes, dtype=np.float32)
    offsets = stroke_coords_to_offsets(arr)
    offsets = offsets[:_MAX_STROKE_LEN]
    return normalize(offsets)


def _make_batch(
    samples: list[dict],
    char_to_index: dict[str, int],
    device: torch.device,
):
    xs, lens, cs, clens = [], [], [], []
    for s in samples:
        try:
            off = _absolute_to_offsets(s["strokes"])
        except Exception:
            continue
        n = len(off)
        padded = np.zeros((_MAX_STROKE_LEN, 3), dtype=np.float32)
        padded[:n] = off
        xs.append(padded)
        lens.append(n)
        idx = char_to_index.get(s["char"], 0)
        enc = np.zeros(_MAX_TEXT_LEN, dtype=np.int64)
        enc[0] = idx
        cs.append(enc)
        clens.append(2)

    if not xs:
        return None
    return (
        torch.tensor(np.array(xs), dtype=torch.float32, device=device),
        torch.tensor(lens, dtype=torch.long, device=device),
        torch.tensor(np.array(cs), dtype=torch.long, device=device),
        torch.tensor(clens, dtype=torch.long, device=device),
    )


def run_fine_tune(
    packaged_model_path: Path,
    accepted_samples: list[dict],
    char_to_index: dict[str, int],
    n_epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 4,
    grad_clip: float = 5.0,
    on_complete: Callable | None = None,
):
    """Start fine-tuning in a background thread. Non-blocking."""

    def _worker():
        global status
        if not _lock.acquire(blocking=False):
            status["progress"] = "Another fine-tune is already running."
            return
        try:
            status.update({"running": True, "progress": "Loading model…", "last_result": None})
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            pkg = torch.load(packaged_model_path, map_location=dev, weights_only=False)
            mp = pkg["model_params"]
            alph_size = len(pkg["config_full"]["dataset"]["alphabet_string"]) + 1

            model = HandwritingRNN(
                lstm_size=mp["lstm_size"],
                output_mixture_components=mp["output_mixture_components"],
                attention_mixture_components=mp["attention_mixture_components"],
                alphabet_size=alph_size,
                dropout_prob=mp["dropout_prob"],
            )
            model.load_state_dict(pkg["model_state_dict"])
            model.to(dev).train()

            opt = torch.optim.Adam(model.parameters(), lr=lr)
            n = len(accepted_samples)
            losses = []

            for epoch in range(n_epochs):
                status["progress"] = f"Epoch {epoch + 1}/{n_epochs} ({n} samples)…"
                perm = np.random.permutation(n)
                ep_loss, ep_batches = 0.0, 0

                for start in range(0, n, batch_size):
                    batch = [accepted_samples[int(i)] for i in perm[start: start + batch_size]]
                    tensors = _make_batch(batch, char_to_index, dev)
                    if tensors is None:
                        continue
                    x, xlen, c, clen = tensors
                    y = x[:, 1:, :]
                    xi = x[:, :-1, :]
                    xl = (xlen - 1).clamp(min=1)

                    opt.zero_grad()
                    pis, sigmas, rhos, mus, es, _ = model(xi, c, clen)
                    _, elem_loss = gaussian_mixture_loss(y, xl, pis, sigmas, rhos, mus, es)
                    elem_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    ep_loss += elem_loss.item()
                    ep_batches += 1

                avg = ep_loss / max(ep_batches, 1)
                losses.append(round(avg, 4))
                logger.info(f"Fine-tune epoch {epoch+1}/{n_epochs} loss={avg:.4f}")

            # Save updated weights
            status["progress"] = "Saving model…"
            pkg["model_state_dict"] = model.state_dict()
            torch.save(pkg, packaged_model_path)

            # Re-script for inference
            model.eval()
            try:
                scripted = torch.jit.script(model)
                scripted_path = packaged_model_path.parent / (
                    packaged_model_path.stem + ".scripted.pt"
                )
                torch.jit.save(scripted, scripted_path)
                logger.info(f"Scripted model saved to {scripted_path}")
            except Exception as e:
                logger.warning(f"Could not re-script model (weights still saved): {e}")

            result = {"n_samples": n, "epochs": n_epochs, "losses": losses}
            status.update({"running": False, "progress": "Done.", "last_result": result})

            if on_complete:
                on_complete()

        except Exception as e:
            logger.error(f"Fine-tune failed: {e}", exc_info=True)
            status.update({"running": False, "progress": f"Error: {e}", "last_result": None})
        finally:
            _lock.release()

    threading.Thread(target=_worker, daemon=True).start()
