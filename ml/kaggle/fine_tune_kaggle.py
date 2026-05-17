"""
Standalone fine-tuning script for Kaggle GPU notebooks.

HOW TO USE:
  1. Upload TWO files to a new Kaggle dataset (e.g. "scriptify-finetune"):
       - ml/packaged_models/handwriting_model.pt
       - ml/data/feedback/feedback_data.json
  2. Create a new Kaggle notebook, attach that dataset, enable GPU T4.
  3. Add your Kaggle API key as secrets (KAGGLE_USERNAME, KAGGLE_KEY) so the
     script can push the finished model back to Kaggle automatically.
     → Notebook settings → Add-ons → Secrets → add both keys.
  4. Set OUTPUT_DATASET below to "your-kaggle-username/scriptify-finetuned"
     (create that empty dataset on Kaggle first, or leave blank to skip push).
  5. Paste this entire file into a single code cell and run it, then go to sleep.
  6. When done the model is pushed to your OUTPUT_DATASET. Download:
       - handwriting_model.pt
       - handwriting_model.scripted.pt
     and replace ml/packaged_models/ with those two files.
"""

# ── 0. Install / imports ───────────────────────────────────────────────────────
import json, math, os, shutil, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

print(f"PyTorch {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Paths & settings ────────────────────────────────────────────────────────
# Change DATASET_SLUG to match the name you gave your INPUT dataset on Kaggle.
DATASET_SLUG   = "scriptify-finetune"          # ← edit if your input slug differs

# Output dataset: script will push back to the SAME dataset you uploaded the inputs to.
# You do NOT need to create a separate dataset — just leave this as-is.
# The username is read automatically from your KAGGLE_USERNAME secret.
OUTPUT_DATASET = f"__auto__/{DATASET_SLUG}"   # resolved to "{your_username}/{DATASET_SLUG}"

INPUT_DIR      = Path("/kaggle/input")   # search entire input tree
OUT_PT         = Path("/kaggle/working/handwriting_model.pt")
OUT_SCRIPTED   = Path("/kaggle/working/handwriting_model.scripted.pt")

# Auto-find the files — Kaggle sometimes nests them in a subdirectory
def _find_file(root: Path, filename: str) -> Path:
    for match in root.rglob(filename):
        return match
    raise FileNotFoundError(
        f"{filename} not found anywhere under {root}\n"
        f"Files present: {[str(p) for p in root.rglob('*') if p.is_file()]}"
    )

print("Scanning input directory …")
MODEL_PT      = _find_file(INPUT_DIR, "handwriting_model.pt")
FEEDBACK_JSON = _find_file(INPUT_DIR, "feedback_data.json")
print(f"  model    : {MODEL_PT}")
print(f"  feedback : {FEEDBACK_JSON}")

# ── 2. Hyper-parameters ────────────────────────────────────────────────────────
N_EPOCHS   = 15
LR         = 1e-4
BATCH_SIZE = 8        # larger batch fits on GPU (was 4 on CPU)
GRAD_CLIP  = 5.0
MAX_STROKE = 1200
MAX_TEXT   = 2

# ── 3. Model code (self-contained, no ml/src import needed) ───────────────────

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size, num_mixture_components, alphabet_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_mixture_components = num_mixture_components
        self.alphabet_size = alphabet_size
        self.attention_params = nn.Linear(
            alphabet_size + 3 + hidden_size,
            3 * num_mixture_components,
        )

    def forward(self, hidden_state, prev_window, prev_kappa, inputs,
                char_seq, char_seq_lengths):
        B = hidden_state.size(0)
        attn_in = torch.cat([prev_window, inputs, hidden_state], dim=1)
        params   = F.softplus(self.attention_params(attn_in))
        alpha, beta, kappa_delta = torch.split(params, self.num_mixture_components, dim=1)
        kappa = prev_kappa + kappa_delta / 25.0
        beta  = torch.clamp(beta, min=0.01)
        max_char_len = char_seq.size(1)
        u = torch.arange(max_char_len, device=hidden_state.device).float()
        u = u.unsqueeze(0).unsqueeze(1).expand(B, self.num_mixture_components, -1)
        kappa_e = kappa.unsqueeze(2).expand(-1, -1, max_char_len)
        beta_e  = beta.unsqueeze(2).expand(-1, -1, max_char_len)
        alpha_e = alpha.unsqueeze(2).expand(-1, -1, max_char_len)
        phi = torch.sum(alpha_e * torch.exp(-(kappa_e - u) ** 2 / (2.0 * beta_e ** 2)), dim=1)
        mask = torch.arange(max_char_len, device=hidden_state.device).unsqueeze(0)
        mask = mask.expand(B, -1) < char_seq_lengths.unsqueeze(1)
        phi  = phi * mask.float()
        context = torch.bmm(phi.unsqueeze(1), char_seq).squeeze(1)
        return context, kappa, phi


class GMMLayer(nn.Module):
    def __init__(self, input_size, num_mixtures, eps=1e-8, sigma_eps=1e-4):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.eps = eps
        self.sigma_eps = sigma_eps
        self.output_layer = nn.Linear(input_size, num_mixtures * 6 + 1)

    def forward(self, x, bias):
        z = self.output_layer(x)
        pis_logits, sigmas_logits, rhos, mus, es = torch.split(
            z, [self.num_mixtures, 2 * self.num_mixtures,
                self.num_mixtures, 2 * self.num_mixtures, 1], dim=1)
        if not isinstance(bias, torch.Tensor):
            bias = torch.tensor(bias, device=x.device, dtype=x.dtype)
        if bias.numel() == 1:
            pis_logits     = pis_logits * (1 + bias)
            sigmas_logits  = sigmas_logits - bias
        else:
            pis_logits     = pis_logits * (1 + bias.unsqueeze(-1))
            sigmas_logits  = sigmas_logits - bias.unsqueeze(-1)
        pis    = F.softmax(pis_logits, dim=-1)
        sigmas = torch.exp(sigmas_logits).clamp(min=self.sigma_eps)
        rhos   = torch.tanh(rhos).clamp(min=self.eps - 1.0, max=1.0 - self.eps)
        es     = torch.sigmoid(es).clamp(min=self.eps, max=1.0 - self.eps)
        return pis, sigmas, rhos, mus, es

    def sample(self, pis, sigmas, rhos, mus, es):
        batch_size = pis.size(0)
        idx = torch.multinomial(pis, 1).squeeze(1)
        mu_x, mu_y = torch.split(mus, self.num_mixtures, dim=1)
        sx,   sy   = torch.split(sigmas, self.num_mixtures, dim=1)
        ie = idx.unsqueeze(1)
        mx = torch.gather(mu_x, 1, ie).squeeze(1)
        my = torch.gather(mu_y, 1, ie).squeeze(1)
        sx = torch.gather(sx, 1, ie).squeeze(1)
        sy = torch.gather(sy, 1, ie).squeeze(1)
        rho = torch.gather(rhos, 1, ie).squeeze(1)
        z0 = torch.randn_like(mx); z1 = torch.randn_like(my)
        xs = mx + sx * z0
        ys = my + sy * (rho * z0 + torch.sqrt((1 - rho**2).clamp(min=self.eps)) * z1)
        e  = torch.bernoulli(es.squeeze(-1) if es.dim() > 1 else es).unsqueeze(-1)
        return torch.cat([torch.stack([xs, ys], dim=1), e], dim=1)


class HandwritingRNN(nn.Module):
    def __init__(self, lstm_size=400, output_mixture_components=20,
                 attention_mixture_components=10, alphabet_size=29, dropout_prob=0.2):
        super().__init__()
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.attention_mixture_components = attention_mixture_components
        self.alphabet_size = alphabet_size
        self.dropout_prob = dropout_prob
        self.lstm1 = nn.LSTMCell(3 + alphabet_size, lstm_size)
        self.lstm2 = nn.LSTMCell(3 + lstm_size + alphabet_size, lstm_size)
        self.lstm3 = nn.LSTMCell(3 + 2 * lstm_size + alphabet_size, lstm_size)
        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)
        self.attention = AttentionMechanism(lstm_size, attention_mixture_components, alphabet_size)
        self.gmm = GMMLayer(3 * lstm_size, output_mixture_components)

    def one_hot_encode(self, char_seq):
        return F.one_hot(char_seq, num_classes=self.alphabet_size).float()

    @torch.jit.ignore  # only sample() is needed for inference; ignore forward so scripting works
    def forward(self, inputs, char_seq, char_seq_lengths, hidden_state=None, bias=None):
        if bias is None:
            bias = torch.tensor([0.5] * inputs.size(0), device=inputs.device, dtype=inputs.dtype)
        B, T, _ = inputs.shape
        if hidden_state is None:
            h1 = c1 = h2 = c2 = h3 = c3 = torch.zeros(B, self.lstm_size, device=inputs.device)
            window = torch.zeros(B, self.alphabet_size, device=inputs.device)
            kappa  = torch.zeros(B, self.attention_mixture_components, device=inputs.device)
        else:
            h1, c1, h2, c2, h3, c3, window, kappa = hidden_state
        char_oh = self.one_hot_encode(char_seq)
        pis_list, sigmas_list, rhos_list, mus_list, es_list = [], [], [], [], []
        for t in range(T):
            x_t = inputs[:, t, :]
            h1, c1 = self.lstm1(torch.cat([window, x_t], dim=1), (h1, c1))
            h1_d = self.dropout(h1) if (self.dropout_prob > 0 and self.training) else h1
            window, kappa, _ = self.attention(h1_d, window, kappa, x_t, char_oh, char_seq_lengths)
            h2, c2 = self.lstm2(torch.cat([x_t, h1_d, window], dim=1), (h2, c2))
            h2_d = self.dropout(h2) if (self.dropout_prob > 0 and self.training) else h2
            h3, c3 = self.lstm3(torch.cat([x_t, h1_d, h2_d, window], dim=1), (h3, c3))
            h3_d = self.dropout(h3) if (self.dropout_prob > 0 and self.training) else h3
            pis, sigmas, rhos, mus, es = self.gmm(torch.cat([h1_d, h2_d, h3_d], dim=1), bias)
            pis_list.append(pis); sigmas_list.append(sigmas); rhos_list.append(rhos)
            mus_list.append(mus); es_list.append(es)
        return (torch.stack(pis_list, 1), torch.stack(sigmas_list, 1),
                torch.stack(rhos_list, 1), torch.stack(mus_list, 1),
                torch.stack(es_list, 1), (h1, c1, h2, c2, h3, c3, window, kappa))

    @torch.jit.export
    def sample(self, char_seq: torch.Tensor, char_seq_lengths: torch.Tensor,
               max_length: int = 1000, bias: float = 0.5,
               prime=None, min_length: int = 200) -> list[torch.Tensor]:
        B = char_seq.size(0); dev = char_seq.device
        char_oh = self.one_hot_encode(char_seq)
        h1 = c1 = h2 = c2 = h3 = c3 = torch.zeros(B, self.lstm_size, device=dev)
        window = torch.zeros(B, self.alphabet_size, device=dev)
        kappa  = torch.zeros(B, self.attention_mixture_components, device=dev)
        x = torch.zeros(B, 3, device=dev); x[:, 2] = 1.0
        bias_t = torch.tensor([bias] * B, device=dev, dtype=torch.float32)
        pen_down_count = torch.zeros(B, dtype=torch.long, device=dev)
        prev_pen_up    = torch.ones(B, dtype=torch.bool, device=dev)
        last_char_idx  = char_seq_lengths - 1
        attn_at_end    = torch.zeros(B, dtype=torch.bool, device=dev)
        finished       = torch.zeros(B, dtype=torch.bool, device=dev)
        out_lens       = torch.full((B,), -1, dtype=torch.long, device=dev)
        strokes = []
        for t in range(max_length):
            h1, c1 = self.lstm1(torch.cat([window, x], dim=1), (h1, c1))
            window, kappa, phi = self.attention(h1, window, kappa, x, char_oh, char_seq_lengths)
            h2, c2 = self.lstm2(torch.cat([x, h1, window], dim=1), (h2, c2))
            h3, c3 = self.lstm3(torch.cat([x, h1, h2, window], dim=1), (h3, c3))
            pis, sigmas, rhos, mus, es = self.gmm(torch.cat([h1, h2, h3], dim=1), bias_t)
            stroke = self.gmm.sample(pis, sigmas, rhos, mus, es)
            strokes.append(stroke); x = stroke
            attn_at_end = attn_at_end | (torch.argmax(phi, dim=1) == last_char_idx)
            pen_is_up = stroke[:, 2] >= 0.5
            pen_down_count = pen_down_count + (prev_pen_up & ~pen_is_up).long()
            prev_pen_up = pen_is_up
            can_stop = (pen_down_count >= 1) if t >= min_length else torch.zeros(B, dtype=torch.bool, device=dev)
            newly = can_stop & attn_at_end & pen_is_up
            out_lens[newly & (out_lens == -1)] = t + 1
            finished = finished | newly
            if finished.all():
                break
        all_s = torch.stack(strokes, dim=1)
        out_lens[out_lens == -1] = all_s.size(1)
        return [all_s[i, :out_lens[i]] for i in range(B)]


# ── 4. Loss ────────────────────────────────────────────────────────────────────
PEN_UP_WEIGHT = 5.0

def gaussian_mixture_loss(y, y_lengths, pis, sigmas, rhos, mus, es, eps=1e-8, sigma_eps=1e-4):
    seq_length = y.size(1)
    y1, y2, y3 = torch.split(y, 1, dim=2)
    K = pis.size(2)
    s1, s2 = torch.split(sigmas, K, dim=2)
    m1, m2 = torch.split(mus, K, dim=2)
    norm = 1.0 / (2 * math.pi * s1.clamp(min=sigma_eps) * s2.clamp(min=sigma_eps) *
                  torch.sqrt((1 - rhos ** 2).clamp(min=sigma_eps)))
    Z = ((y1 - m1) / s1) ** 2 + ((y2 - m2) / s2) ** 2 - 2 * rhos * (y1 - m1) * (y2 - m2) / (s1 * s2)
    gaussian = torch.exp(-Z / (2 * (1 - rhos ** 2))) * norm
    gmm_ll = torch.sum(pis * gaussian, dim=2).clamp(min=eps)
    bern   = torch.where(y3 > 0.5, es, 1 - es).squeeze(2)
    w      = torch.where(y3.squeeze(2) > 0.5,
                         torch.full_like(bern, PEN_UP_WEIGHT),
                         torch.ones_like(bern))
    nll    = -(torch.log(gmm_ll) + w * torch.log(bern.clamp(min=eps)))
    mask   = torch.arange(seq_length, device=y.device)[None] < y_lengths[:, None]
    nll    = torch.where(mask, nll, torch.zeros_like(nll))
    elem_loss = torch.sum(nll) / torch.sum(y_lengths.float())
    return torch.sum(nll, dim=1) / y_lengths.float(), elem_loss


# ── 5. Preprocessing helpers ───────────────────────────────────────────────────
def stroke_coords_to_offsets(arr):
    diffs = arr[1:, :2] - arr[:-1, :2]
    eos   = arr[1:, 2:3]
    return np.concatenate([np.array([[0, 0, 1]]), np.concatenate([diffs, eos], axis=1)], axis=0)

def normalize(offsets):
    """Zero-mean, unit-std normalisation on dx and dy only."""
    mean = offsets[:, :2].mean(axis=0)
    std  = offsets[:, :2].std(axis=0).clip(min=1e-6)
    out  = offsets.copy()
    out[:, :2] = (offsets[:, :2] - mean) / std
    return out.astype(np.float32)

def absolute_to_offsets(absolute_strokes):
    arr = np.array(absolute_strokes, dtype=np.float32)
    off = stroke_coords_to_offsets(arr)
    off = off[:MAX_STROKE]
    return normalize(off)


# ── 6. Load model ──────────────────────────────────────────────────────────────
import pathlib
pathlib.WindowsPath = pathlib.PurePosixPath  # fix for models saved on Windows

print(f"\nLoading model from {MODEL_PT} …")
pkg = torch.load(MODEL_PT, map_location=device, weights_only=False)
mp  = pkg["model_params"]
alphabet_string = pkg["config_full"]["dataset"]["alphabet_string"]
alph_size = len(alphabet_string) + 1          # +1 for NULL char
char_to_index = {c: i + 1 for i, c in enumerate(alphabet_string)}

model = HandwritingRNN(
    lstm_size=mp["lstm_size"],
    output_mixture_components=mp["output_mixture_components"],
    attention_mixture_components=mp["attention_mixture_components"],
    alphabet_size=alph_size,
    dropout_prob=mp["dropout_prob"],
)
model.load_state_dict(pkg["model_state_dict"])
model.to(device).train()
print(f"Model loaded. Alphabet size: {alph_size}, Device: {device}")


# ── 7. Load feedback data ──────────────────────────────────────────────────────
print(f"\nLoading feedback from {FEEDBACK_JSON} …")
with open(FEEDBACK_JSON, encoding="utf-8") as f:
    raw = json.load(f)
samples_all = raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw
accepted = [s for s in samples_all if s.get("accepted")]
print(f"Total samples: {len(samples_all)}, Accepted: {len(accepted)}")


# ── 8. Batch builder ───────────────────────────────────────────────────────────
def make_batch(samples):
    xs, lens, cs, clens = [], [], [], []
    for s in samples:
        try:
            off = absolute_to_offsets(s["strokes"])
        except Exception:
            continue
        n = len(off)
        pad = np.zeros((MAX_STROKE, 3), dtype=np.float32)
        pad[:n] = off
        xs.append(pad); lens.append(n)
        idx = char_to_index.get(s["char"], 0)
        enc = np.zeros(MAX_TEXT, dtype=np.int64); enc[0] = idx
        cs.append(enc); clens.append(2)
    if not xs:
        return None
    return (
        torch.tensor(np.array(xs),    dtype=torch.float32, device=device),
        torch.tensor(lens,            dtype=torch.long,    device=device),
        torch.tensor(np.array(cs),    dtype=torch.long,    device=device),
        torch.tensor(clens,           dtype=torch.long,    device=device),
    )


# ── 9. Checkpoint helpers ──────────────────────────────────────────────────────
CKPT_DIR = Path("/kaggle/working/checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

def save_checkpoint(epoch: int, model, opt, losses: list):
    ckpt = {
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "opt_state_dict":   opt.state_dict(),
        "losses":           losses,
        "pkg":              pkg,          # carries alphabet / config metadata
    }
    path = CKPT_DIR / f"epoch_{epoch:02d}.pt"
    torch.save(ckpt, path)
    print(f"    checkpoint saved → {path}")
    return path

def load_latest_checkpoint(model, opt):
    ckpts = sorted(CKPT_DIR.glob("epoch_*.pt"))
    if not ckpts:
        return 0, []
    latest = ckpts[-1]
    print(f"Resuming from checkpoint: {latest}")
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["opt_state_dict"])
    return ckpt["epoch"], ckpt["losses"]   # epoch = last completed epoch


# ── 10. Fine-tuning loop ───────────────────────────────────────────────────────
opt        = torch.optim.Adam(model.parameters(), lr=LR)
start_epoch, losses = load_latest_checkpoint(model, opt)
n          = len(accepted)

if start_epoch >= N_EPOCHS:
    print(f"Already completed {N_EPOCHS} epochs — skipping training.")
else:
    print(f"\nStarting fine-tune: epochs {start_epoch+1}–{N_EPOCHS}, "
          f"{n} samples, lr={LR}, batch={BATCH_SIZE}\n")
    model.train()
    for epoch in range(start_epoch, N_EPOCHS):
        perm = np.random.permutation(n)
        ep_loss, ep_batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            batch = [accepted[int(i)] for i in perm[start: start + BATCH_SIZE]]
            tensors = make_batch(batch)
            if tensors is None:
                continue
            x, xlen, c, clen = tensors
            y = x[:, 1:, :]; xi = x[:, :-1, :]; xl = (xlen - 1).clamp(min=1)
            opt.zero_grad()
            pis, sigmas, rhos, mus, es, _ = model(xi, c, clen)
            _, elem_loss = gaussian_mixture_loss(y, xl, pis, sigmas, rhos, mus, es)
            elem_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_loss += elem_loss.item(); ep_batches += 1
        avg = ep_loss / max(ep_batches, 1)
        losses.append(round(avg, 4))
        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS}  loss={avg:.4f}")
        save_checkpoint(epoch + 1, model, opt, losses)

print(f"\nFinal loss: {losses[-1]}  |  All losses: {losses}")


# ── 11. Save final model ───────────────────────────────────────────────────────
print(f"\nSaving .pt to {OUT_PT} …")
pkg["model_state_dict"] = model.state_dict()
torch.save(pkg, OUT_PT)

print("Scripting model …")
model.eval()
try:
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, OUT_SCRIPTED)
    print(f"Scripted model saved to {OUT_SCRIPTED}")
except Exception as e:
    print(f"WARNING: scripting failed ({e}) — .pt weights were still saved")

print("\nDone! Loss history:", losses)

# ── 12. Auto-push output back to Kaggle ───────────────────────────────────────
print("\nSetting up auto-push …")

# Read credentials from secrets
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    _username = _s.get_secret("KAGGLE_USERNAME")
    _key      = _s.get_secret("KAGGLE_KEY")
    _kdir = Path.home() / ".kaggle"
    _kdir.mkdir(exist_ok=True)
    (_kdir / "kaggle.json").write_text(json.dumps({"username": _username, "key": _key}))
    os.chmod(_kdir / "kaggle.json", 0o600)
    print(f"Kaggle credentials loaded for: {_username}")
    _push_ok = True
except Exception as _e:
    print(f"WARNING: Could not read Kaggle secrets ({_e})")
    print("Files are in /kaggle/working/ — download them manually from the Output panel.")
    _push_ok = False

if _push_ok:
    # Resolve dataset id: replace __auto__ with the real username
    _dataset_id = OUTPUT_DATASET.replace("__auto__", _username)
    print(f"Pushing to dataset: {_dataset_id}")

    # Stage output files
    _staging = Path("/kaggle/working/_push_staging")
    _staging.mkdir(exist_ok=True)
    shutil.copy2(OUT_PT, _staging / "handwriting_model.pt")
    if OUT_SCRIPTED.exists():
        shutil.copy2(OUT_SCRIPTED, _staging / "handwriting_model.scripted.pt")
    (_staging / "dataset-metadata.json").write_text(json.dumps({
        "title":    _dataset_id.split("/")[-1],
        "id":       _dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }))

    _msg = f"Fine-tuned {len(accepted)} samples, {N_EPOCHS} epochs, final_loss={losses[-1]}"

    # Try versioning existing dataset first; if not found, create it
    _r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(_staging), "-m", _msg],
        capture_output=True, text=True,
    )
    if _r.returncode == 0:
        print("✓ Push successful! New version uploaded.")
        print(_r.stdout.strip())
    else:
        print("Dataset version not found — creating new dataset …")
        _r2 = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(_staging)],
            capture_output=True, text=True,
        )
        if _r2.returncode == 0:
            print("✓ Dataset created and files uploaded!")
            print(_r2.stdout.strip())
        else:
            print("ERROR: push failed. Download files manually from the Output panel.")
            print(_r.stderr.strip())
            print(_r2.stderr.strip())

    shutil.rmtree(_staging, ignore_errors=True)

print("\nAll done! Download handwriting_model.pt and handwriting_model.scripted.pt")
print("from your Kaggle dataset (or Output panel) and drop them into ml/packaged_models/")
