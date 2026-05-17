"""Re-script a fine-tuned handwriting_model.pt on Windows.

Usage:
    python ml/scripts/rescript.py
    python ml/scripts/rescript.py path/to/handwriting_model.pt

Saves handwriting_model.scripted.pt next to the .pt file.
"""
import sys
from pathlib import Path
import torch

_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from src.models.rnn import HandwritingRNN

def rescript(pt_path: Path) -> None:
    print(f"Loading {pt_path} …")
    pkg = torch.load(pt_path, map_location="cpu", weights_only=False)
    mp  = pkg["model_params"]
    alph_size = len(pkg["config_full"]["dataset"]["alphabet_string"]) + 1

    model = HandwritingRNN(
        lstm_size=mp["lstm_size"],
        output_mixture_components=mp["output_mixture_components"],
        attention_mixture_components=mp["attention_mixture_components"],
        alphabet_size=alph_size,
        dropout_prob=mp["dropout_prob"],
    )
    model.load_state_dict(pkg["model_state_dict"])
    model.eval()

    scripted = torch.jit.script(model)
    out = pt_path.parent / (pt_path.stem + ".scripted.pt")
    torch.jit.save(scripted, out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    pt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ml/packaged_models/handwriting_model.pt")
    rescript(pt)
