"""
Build processed .npy files for the AOLAH Arabic character dataset.

Run from the ml/ directory:
    python -m src.data.aolah_dataset

What it does
------------
1. Reads all raw CSV files via aolah_reader.collect_samples().
2. Preprocesses each sample:
       smooth_strokes  →  stroke_coords_to_offsets  →  normalize
3. Splits the data 80 / 10 / 10 (train / val / test), stratified by character
   so every Arabic character is equally represented in every split.
4. Saves six .npy files + one source-file list per split:

    data/processed/
        train/
            strokes.npy          float32  (N, MAX_STROKE_LEN, 3)
            strokes_len.npy      int32    (N,)
            chars.npy            int64    (N, MAX_TEXT_LEN)
            chars_len.npy        int32    (N,)
            writer_ids.npy       int32    (N,)
            source_files.txt
        val/   (same structure)
        test/  (same structure)
        alphabet.txt             the 28 Arabic chars (no null), one string
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

from config.config import load_config
from src.data.aolah_reader import collect_samples, ARABIC_ALPHABET
from src.data.preprocessing import smooth_strokes, stroke_coords_to_offsets, normalize
from src.utils.text_utils import encode_text, get_alphabet_map, construct_alphabet_list


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def _stratified_split(
    samples: list,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Return (train_indices, val_indices, test_indices) stratified by character.

    For every character class the samples are shuffled then split as:
        first  (1 - val_ratio - test_ratio) fraction → train
        next   val_ratio fraction                     → val
        last   test_ratio fraction                    → test

    Each split always gets at least 1 sample per class when the class has
    enough samples (>= 3).  When a class has fewer samples all go to train.
    """
    by_char: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_char[s.char].append(i)

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)

    for char in sorted(by_char):
        indices = np.array(by_char[char], dtype=np.int64)
        rng.shuffle(indices)
        n = len(indices)

        if n < 3:
            train_idx.extend(indices.tolist())
            continue

        n_val  = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Edge case: too few samples → put everything in train
            train_idx.extend(indices.tolist())
            continue

        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train : n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val :].tolist())

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(strokes: np.ndarray, max_stroke_len: int) -> np.ndarray:
    """
    Full preprocessing pipeline for one character sample.

    Steps
    -----
    1. smooth_strokes  – reduces digitisation jitter per stroke segment.
    2. stroke_coords_to_offsets – converts absolute (x, y) to relative (dx, dy)
                                   so the model learns movements, not positions.
    3. Truncate to max_stroke_len.
    4. normalize – scales offsets to median unit norm so magnitudes are
                   comparable across writers and characters.

    Note: deskew_line is intentionally omitted.  AOLAH data is collected on a
    touchscreen with consistent orientation, so deskewing is unnecessary and
    would distort character shapes.

    Args:
        strokes:       float32 array (N, 3), columns: x, y, eos
        max_stroke_len: maximum number of points to keep

    Returns:
        float32 array (M, 3) where M <= max_stroke_len, columns: dx, dy, eos
    """
    strokes = smooth_strokes(strokes)
    offsets = stroke_coords_to_offsets(strokes)
    offsets = offsets[:max_stroke_len]
    offsets = normalize(offsets)
    return offsets


# ---------------------------------------------------------------------------
# Array builder
# ---------------------------------------------------------------------------

def _build_arrays(
    indices: list[int],
    samples: list,
    max_stroke_len: int,
    max_text_len: int,
    char_to_index: dict[str, int],
) -> dict:
    """
    Convert a list of sample indices into padded numpy arrays ready for training.

    Returns a dict with keys:
        strokes, strokes_len, chars, chars_len, writer_ids, source_files
    """
    n = len(indices)
    strokes_arr    = np.zeros((n, max_stroke_len, 3), dtype=np.float32)
    strokes_len_arr = np.zeros(n, dtype=np.int32)
    chars_arr      = np.zeros((n, max_text_len),      dtype=np.int64)
    chars_len_arr  = np.zeros(n, dtype=np.int32)
    writer_ids_arr = np.zeros(n, dtype=np.int32)
    source_files: list[str] = []

    valid_flags: list[bool] = []

    for i, idx in enumerate(indices):
        s = samples[idx]
        try:
            offsets = _preprocess(s.strokes, max_stroke_len)
        except Exception as e:
            print(f"  Warning: preprocessing failed for {s.source_file}: {e}")
            valid_flags.append(False)
            continue

        length = len(offsets)
        strokes_arr[i, :length] = offsets
        strokes_len_arr[i]      = length

        # Encode the single Arabic character + EOS into a fixed-length array.
        encoded, char_len = encode_text(
            s.char,
            char_to_index,
            max_length=max_text_len,
            add_eos=True,
            eos_char_index=0,
        )
        chars_arr[i]      = encoded
        chars_len_arr[i]  = char_len
        writer_ids_arr[i] = s.writer_id
        source_files.append(s.source_file)
        valid_flags.append(True)

    # Drop rows that failed preprocessing.
    valid = np.array(valid_flags, dtype=bool)
    n_skipped = int((~valid).sum())
    if n_skipped:
        print(f"  Dropped {n_skipped} samples due to preprocessing errors.")
        # Rebuild compact arrays without the failed rows.
        keep = np.where(valid)[0]
        strokes_arr    = strokes_arr[keep]
        strokes_len_arr = strokes_len_arr[keep]
        chars_arr      = chars_arr[keep]
        chars_len_arr  = chars_len_arr[keep]
        writer_ids_arr = writer_ids_arr[keep]

    return {
        'strokes':      strokes_arr,
        'strokes_len':  strokes_len_arr,
        'chars':        chars_arr,
        'chars_len':    chars_len_arr,
        'writer_ids':   writer_ids_arr,
        'source_files': source_files,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_split(split_dir: Path, arrays: dict) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / 'strokes.npy',     arrays['strokes'])
    np.save(split_dir / 'strokes_len.npy', arrays['strokes_len'])
    np.save(split_dir / 'chars.npy',       arrays['chars'])
    np.save(split_dir / 'chars_len.npy',   arrays['chars_len'])
    np.save(split_dir / 'writer_ids.npy',  arrays['writer_ids'])
    with open(split_dir / 'source_files.txt', 'w', encoding='utf-8') as f:
        for p in arrays['source_files']:
            f.write(p + '\n')
    n = arrays['strokes'].shape[0]
    print(f"  Saved {n} samples → {split_dir}")


def _print_split_stats(name: str, indices: list[int], samples: list) -> None:
    by_char: dict[str, int] = defaultdict(int)
    for i in indices:
        by_char[samples[i].char] += 1
    counts = sorted(by_char.items())
    min_c = min(v for _, v in counts)
    max_c = max(v for _, v in counts)
    print(f"  {name}: {len(indices)} samples, "
          f"{len(counts)} chars, "
          f"per-char min={min_c} max={max_c}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    config = load_config()

    MAX_STROKE_LEN = config.dataset.max_stroke_len   # 1200
    MAX_TEXT_LEN   = config.dataset.max_text_len     # 2  (char + EOS)
    SEED           = config.dataset.random_seed      # 42
    processed_dir  = config.paths.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Build the alphabet (null char at index 0, then the 28 Arabic chars).
    alphabet_string = ''.join(ARABIC_ALPHABET)           # 28 chars, Unicode order
    alphabet_list   = construct_alphabet_list(alphabet_string)  # ['\x00'] + list(alphabet_string)
    char_to_index   = get_alphabet_map(alphabet_list)

    # Save the alphabet so the dataloader can read it without importing this module.
    with open(processed_dir / 'alphabet.txt', 'w', encoding='utf-8') as f:
        f.write(alphabet_string)
    print(f"Alphabet ({len(alphabet_list)} entries incl. null): {alphabet_string}")

    # Collect all samples from both dataset formats.
    print("\nCollecting samples …")
    samples = collect_samples(config.paths.raw_data_root)
    if not samples:
        raise RuntimeError(
            "No samples collected.  Check that raw_data_root in config.yaml "
            "points to the AOLAH dataset directory."
        )

    # 80 / 10 / 10 stratified split.
    print("\nSplitting 80/10/10 (stratified by character) …")
    train_idx, val_idx, test_idx = _stratified_split(
        samples, val_ratio=0.1, test_ratio=0.1, seed=SEED
    )
    _print_split_stats('train', train_idx, samples)
    _print_split_stats('val',   val_idx,   samples)
    _print_split_stats('test',  test_idx,  samples)

    # Preprocess and save each split.
    for split_name, indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        print(f"\nProcessing {split_name} …")
        arrays = _build_arrays(indices, samples, MAX_STROKE_LEN, MAX_TEXT_LEN, char_to_index)
        _save_split(processed_dir / split_name, arrays)

    print(f"\nDone.  Processed data saved to: {processed_dir}")
