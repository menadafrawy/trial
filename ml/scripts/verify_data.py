"""
Sanity-check the processed AOLAH dataset before training.

Run from the ml/ directory:
    python scripts/verify_data.py

Checks performed
----------------
1. All three split directories exist with the expected .npy files.
2. No NaN or Inf values in any array.
3. Stroke lengths are within bounds (> 0 and <= max_stroke_len).
4. Char lengths are 1 or 2 (single char + optional EOS).
5. Every character in the alphabet appears in every split.
6. Offset magnitudes look sensible (not all zeros, not exploding).
7. Saves a PNG plot of 5 random samples per split so you can visually
   confirm the strokes look like Arabic characters.

Output: prints a PASS / FAIL report and saves plots to data/processed/plots/.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Allow running from the ml/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import load_config
from src.data.preprocessing import offsets_to_stroke_coords


def _load_split(processed_dir: Path, split: str) -> dict:
    split_dir = processed_dir / split
    return {
        'strokes':     np.load(split_dir / 'strokes.npy'),
        'strokes_len': np.load(split_dir / 'strokes_len.npy'),
        'chars':       np.load(split_dir / 'chars.npy'),
        'chars_len':   np.load(split_dir / 'chars_len.npy'),
        'writer_ids':  np.load(split_dir / 'writer_ids.npy'),
    }


def _check_split(name: str, data: dict, alphabet: list[str], max_stroke_len: int) -> list[str]:
    """Return a list of failure messages (empty = all passed)."""
    failures = []
    strokes     = data['strokes']
    strokes_len = data['strokes_len']
    chars       = data['chars']
    chars_len   = data['chars_len']
    N = strokes.shape[0]

    # 1. Shape sanity
    if strokes.ndim != 3 or strokes.shape[2] != 3:
        failures.append(f"{name}: strokes shape {strokes.shape} — expected (N, L, 3)")
    if chars.ndim != 2:
        failures.append(f"{name}: chars shape {chars.shape} — expected (N, T)")

    # 2. NaN / Inf
    if not np.isfinite(strokes).all():
        n_bad = int((~np.isfinite(strokes)).any(axis=(1, 2)).sum())
        failures.append(f"{name}: {n_bad}/{N} stroke samples contain NaN or Inf")

    # 3. Stroke lengths in bounds
    zero_len = int((strokes_len == 0).sum())
    over_len = int((strokes_len > max_stroke_len).sum())
    if zero_len:
        failures.append(f"{name}: {zero_len} samples with stroke_len == 0")
    if over_len:
        failures.append(f"{name}: {over_len} samples with stroke_len > {max_stroke_len}")

    # 4. Char lengths (expect 1 or 2 for single-char + EOS)
    bad_clen = int(((chars_len < 1) | (chars_len > 2)).sum())
    if bad_clen:
        failures.append(f"{name}: {bad_clen} samples with unexpected chars_len (not 1 or 2)")

    # 5. Character coverage — every char in the alphabet must appear
    char_indices_seen = set(int(chars[i, 0]) for i in range(N))
    alphabet_indices  = set(range(1, len(alphabet) + 1))  # index 0 is null
    missing = alphabet_indices - char_indices_seen
    if missing:
        missing_chars = [alphabet[i - 1] for i in missing]
        failures.append(f"{name}: missing characters in split: {missing_chars}")

    # 6. Offset magnitude check (after normalization, median ||(dx,dy)|| ~ 1)
    flat_xy = strokes[:, :, :2].reshape(-1, 2)
    norms = np.linalg.norm(flat_xy, axis=1)
    norms = norms[norms > 0]  # ignore padding zeros
    if len(norms) == 0:
        failures.append(f"{name}: all offsets are zero — preprocessing likely failed")
    else:
        med = float(np.median(norms))
        if med < 0.01 or med > 100:
            failures.append(f"{name}: suspicious median offset magnitude {med:.4f} "
                            "(expected ~ 1.0 after normalize)")

    return failures


def _plot_samples(name: str, data: dict, alphabet: list[str],
                  out_dir: Path, n_samples: int = 5, seed: int = 0) -> None:
    """Plot n_samples random strokes as reconstructed absolute coordinates."""
    rng = np.random.default_rng(seed)
    N = data['strokes'].shape[0]
    indices = rng.choice(N, size=min(n_samples, N), replace=False)

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        length = int(data['strokes_len'][idx])
        offsets = data['strokes'][idx, :length]           # (L, 3)
        coords  = offsets_to_stroke_coords(offsets)        # back to absolute (x, y, eos)

        char_idx = int(data['chars'][idx, 0])
        char_label = alphabet[char_idx - 1] if 1 <= char_idx <= len(alphabet) else '?'

        # Draw each stroke segment in a different colour.
        start = 0
        for end in range(length):
            if coords[end, 2] == 1.0:
                seg = coords[start:end + 1]
                ax.plot(seg[:, 0], -seg[:, 1], linewidth=1.5)  # flip y for display
                start = end + 1

        ax.set_title(f"char: {char_label}  len: {length}", fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle(f"Split: {name}", fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_samples.png"
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def main() -> int:
    config = load_config()
    processed_dir  = config.paths.processed_data_dir
    max_stroke_len = config.dataset.max_stroke_len
    plots_dir      = processed_dir / 'plots'

    # Load alphabet.
    alphabet_path = processed_dir / 'alphabet.txt'
    if not alphabet_path.exists():
        print(f"FAIL: alphabet.txt not found at {alphabet_path}")
        print("      Run:  python -m src.data.aolah_dataset")
        return 1
    with open(alphabet_path, 'r', encoding='utf-8') as f:
        alphabet = list(f.read().strip())
    print(f"Alphabet ({len(alphabet)} chars): {''.join(alphabet)}\n")

    all_failures: list[str] = []

    for split in ('train', 'val', 'test'):
        split_dir = processed_dir / split
        # Check files exist.
        for fname in ('strokes.npy', 'strokes_len.npy', 'chars.npy',
                      'chars_len.npy', 'writer_ids.npy', 'source_files.txt'):
            if not (split_dir / fname).exists():
                all_failures.append(f"{split}: missing file {fname}")

        if any(f.startswith(split + ':') for f in all_failures):
            continue  # skip further checks for this split

        data = _load_split(processed_dir, split)
        N = data['strokes'].shape[0]

        # Per-character sample counts.
        char_counts = Counter(int(data['chars'][i, 0]) for i in range(N))
        counts_str = ', '.join(
            f"{alphabet[k-1]}:{v}" for k, v in sorted(char_counts.items())
            if 1 <= k <= len(alphabet)
        )
        print(f"[{split}]  {N} samples")
        print(f"  per-char: {counts_str}")

        failures = _check_split(split, data, alphabet, max_stroke_len)
        all_failures.extend(failures)
        for f in failures:
            print(f"  FAIL: {f}")

        _plot_samples(split, data, alphabet, plots_dir)

    print()
    if all_failures:
        print(f"{'='*60}")
        print(f"RESULT: FAILED  ({len(all_failures)} issue(s))")
        for f in all_failures:
            print(f"  • {f}")
        print(f"{'='*60}")
        return 1
    else:
        print(f"{'='*60}")
        print("RESULT: ALL CHECKS PASSED — data looks good for training.")
        print(f"{'='*60}")
        return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
