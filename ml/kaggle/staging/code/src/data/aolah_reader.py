import numpy as np
from pathlib import Path
from typing import NamedTuple, Union

# Maps every folder-name variant (both Format A and Format B) to its Arabic Unicode char.
FOLDER_TO_ARABIC: dict[str, str] = {
    # Format A names
    'ai':    'ع',  # U+0639
    'alif':  'ا',  # U+0627
    'ba':    'ب',  # U+0628
    'dal':   'د',  # U+062F
    'dhad':  'ض',  # U+0636
    'geem':  'ج',  # U+062C
    'ha':    'ه',  # U+0647
    'hha':   'ح',  # U+062D
    'kaf':   'ك',  # U+0643
    'kha':   'خ',  # U+062E
    'lam':   'ل',  # U+0644
    'non':   'ن',  # U+0646
    'ra':    'ر',  # U+0631
    'sad':   'ص',  # U+0635
    'seen':  'س',  # U+0633
    'sheen': 'ش',  # U+0634
    'ta':    'ت',  # U+062A
    'tha':   'ث',  # U+062B
    'thal':  'ذ',  # U+0630
    'tta':   'ط',  # U+0637
    'waw':   'و',  # U+0648
    'ya':    'ي',  # U+064A
    'zay':   'ز',  # U+0632
    'zha':   'ظ',  # U+0638
    # Format B alternative spellings for the same chars
    'ain':   'ع',  # U+0639  (same as 'ai')
    'jeem':  'ج',  # U+062C  (same as 'geem')
    'noon':  'ن',  # U+0646  (same as 'non')
    # Format B exclusive chars (not present in Format A)
    'fa':    'ف',  # U+0641
    'ghain': 'غ',  # U+063A
    'meem':  'م',  # U+0645
    'qaf':   'ق',  # U+0642
}

# All 28 Arabic characters present in the dataset, sorted by Unicode codepoint.
ARABIC_ALPHABET: list[str] = sorted(set(FOLDER_TO_ARABIC.values()))
# = ['ا','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي']


class CharacterSample(NamedTuple):
    char: str           # Arabic Unicode character (e.g. 'ا')
    writer_id: int      # unique integer writer ID
    strokes: np.ndarray # shape (N, 3): columns are x, y, eos  (float32)
    source_file: str    # path to originating CSV, for debugging


def _stroke_id_to_eos(raw: np.ndarray) -> np.ndarray:
    """
    Convert AOLAH raw CSV data (x, y, stroke_id) to (x, y, eos).

    The AOLAH dataset stores each pen-down segment as a block of consecutive rows
    sharing the same stroke_id value (1, 2, 3, ...).  When the stroke_id changes
    between two consecutive rows the pen was lifted, so the earlier row marks the
    end of a stroke (eos = 1).  The final row always gets eos = 1.

    Args:
        raw: float array, shape (N, 3+), columns: x, y, stroke_id, ...

    Returns:
        float32 array, shape (N, 3), columns: x, y, eos
    """
    xy = raw[:, :2].astype(np.float32)
    stroke_ids = raw[:, 2]
    eos = np.zeros(len(stroke_ids), dtype=np.float32)

    # Mark the last point of every stroke segment.
    for i in range(len(stroke_ids) - 1):
        if stroke_ids[i] != stroke_ids[i + 1]:
            eos[i] = 1.0
    eos[-1] = 1.0  # last point always ends the character

    return np.column_stack([xy, eos])


def _load_csv(csv_path: Path, min_points: int = 5) -> np.ndarray | None:
    """
    Load a single AOLAH stroke CSV and return (x, y, eos) or None on failure.

    Args:
        csv_path:   path to a '*strokes.csv' file
        min_points: reject files shorter than this (likely corrupt / calibration)

    Returns:
        float32 array shape (N, 3) or None
    """
    try:
        raw = np.loadtxt(str(csv_path), delimiter=',', dtype=np.float64)
    except Exception as e:
        print(f"  Warning: could not read {csv_path}: {e}")
        return None

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.ndim != 2 or raw.shape[1] < 3:
        print(f"  Warning: unexpected shape {raw.shape} in {csv_path}, skipping.")
        return None

    if len(raw) < min_points:
        return None

    return _stroke_id_to_eos(raw)


def _collect_format_a(data_root: Path) -> list[CharacterSample]:
    """
    Collect samples from Format A:
        data_root/.../USER NNN/{char name} Strokes/{char name} strokes.csv

    Writer IDs come from 'USER NNN' → N  (range 1–70).
    """
    samples: list[CharacterSample] = []

    user_dirs = sorted(
        p for p in data_root.rglob('USER *')
        if p.is_dir() and p.name.startswith('USER ')
    )

    for user_dir in user_dirs:
        try:
            writer_id = int(user_dir.name.split()[1])
        except (IndexError, ValueError):
            continue

        for strokes_dir in sorted(user_dir.iterdir()):
            # Only process immediate '{char} Strokes' subdirectories.
            if not strokes_dir.is_dir():
                continue
            if not strokes_dir.name.endswith(' Strokes'):
                continue

            char_name = strokes_dir.name[: -len(' Strokes')].strip().lower()
            if char_name not in FOLDER_TO_ARABIC:
                continue

            csv_files = list(strokes_dir.glob('*strokes.csv'))
            if not csv_files:
                continue

            strokes = _load_csv(csv_files[0])
            if strokes is None:
                continue

            samples.append(CharacterSample(
                char=FOLDER_TO_ARABIC[char_name],
                writer_id=writer_id,
                strokes=strokes,
                source_file=str(csv_files[0]),
            ))

    return samples


def _collect_format_b(data_root: Path) -> list[CharacterSample]:
    """
    Collect samples from Format B:
        data_root/.../USER NNN/user M/{char name} strokes.csv   (flat, no subdir)

    Writer IDs come from 'user M' → M + 200 to avoid collision with Format A IDs.
    This format has all 28 Arabic characters (including fa, ghain, meem, qaf).
    """
    samples: list[CharacterSample] = []

    # Find lowercase 'user N' directories (with a space) nested inside USER dirs.
    nested_dirs = sorted(
        p for p in data_root.rglob('user *')
        if p.is_dir() and p.name.startswith('user ') and p.name[5:].strip().isdigit()
    )

    for user_dir in nested_dirs:
        try:
            writer_id = int(user_dir.name.split()[1]) + 200
        except (IndexError, ValueError):
            continue

        for csv_file in sorted(user_dir.glob('*strokes.csv')):
            # Filename pattern: '{char name} strokes.csv'
            stem = csv_file.stem  # e.g. 'alif strokes'
            if not stem.endswith(' strokes'):
                continue
            char_name = stem[: -len(' strokes')].strip().lower()
            if char_name not in FOLDER_TO_ARABIC:
                continue

            strokes = _load_csv(csv_file)
            if strokes is None:
                continue

            samples.append(CharacterSample(
                char=FOLDER_TO_ARABIC[char_name],
                writer_id=writer_id,
                strokes=strokes,
                source_file=str(csv_file),
            ))

    return samples


def collect_samples(data_root: Union[str, Path]) -> list[CharacterSample]:
    """
    Collect all character stroke samples from the AOLAH dataset.

    Handles two directory layouts present in the dataset:
      Format A – 70 users × 24 chars (top-level USER NNN subdirs)
      Format B – 69 users × 28 chars (nested lowercase 'user N' subdirs,
                 includes the 4 chars missing from Format A: fa, ghain, meem, qaf)

    Combined yield: 3 612 samples across all 28 Arabic characters.

    Args:
        data_root: path to the AOLAH root directory
                   (e.g. 'data/raw/aolah' — the directory that contains the
                    'AOLAH CHARACTERS TRAINING (1)' folder or the USER dirs)

    Returns:
        List of CharacterSample named tuples, sorted by (writer_id, char).
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    fmt_a = _collect_format_a(data_root)
    fmt_b = _collect_format_b(data_root)

    all_samples = fmt_a + fmt_b
    all_samples.sort(key=lambda s: (s.writer_id, s.char))

    chars_found = sorted(set(s.char for s in all_samples))
    print(
        f"Collected {len(all_samples)} samples "
        f"({len(fmt_a)} Format-A, {len(fmt_b)} Format-B) "
        f"covering {len(chars_found)}/28 Arabic characters."
    )
    return all_samples
