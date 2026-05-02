import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class ProcessedHandwritingDataset(Dataset):
    """
    PyTorch Dataset that loads one pre-processed split (train / val / test).

    Expected directory layout (produced by aolah_dataset.py):
        processed_dir/
            train/
                strokes.npy          float32  (N, MAX_STROKE_LEN, 3)
                strokes_len.npy      int32    (N,)
                chars.npy            int64    (N, MAX_TEXT_LEN)
                chars_len.npy        int32    (N,)
                writer_ids.npy       int32    (N,)
                source_files.txt
            val/   (same)
            test/  (same)
            alphabet.txt             28 Arabic chars, no null, single line

    Args:
        processed_dir: path to the top-level processed directory
                       (e.g. 'data/processed').  The split subfolder is
                       appended automatically.
        split:         one of 'train', 'val', 'test'
    """

    def __init__(self, processed_dir: Path, split: str = 'train'):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        split_dir = Path(processed_dir) / split

        self.strokes     = np.load(split_dir / 'strokes.npy')
        self.strokes_len = np.load(split_dir / 'strokes_len.npy')
        self.chars       = np.load(split_dir / 'chars.npy')
        self.chars_len   = np.load(split_dir / 'chars_len.npy')
        self.writer_ids  = np.load(split_dir / 'writer_ids.npy')

        source_path = split_dir / 'source_files.txt'
        if not source_path.exists():
            raise FileNotFoundError(f"source_files.txt not found in {split_dir}")
        with open(source_path, 'r', encoding='utf-8') as f:
            self.source_files = [line.strip() for line in f if line.strip()]
        if len(self.source_files) != len(self.strokes):
            raise ValueError(
                f"source_files.txt has {len(self.source_files)} entries but "
                f"strokes.npy has {len(self.strokes)} rows in {split_dir}"
            )

        # Load alphabet from the parent processed directory.
        alphabet_path = Path(processed_dir) / 'alphabet.txt'
        if not alphabet_path.exists():
            raise FileNotFoundError(
                f"alphabet.txt not found at {alphabet_path}.  "
                "Run aolah_dataset.py first."
            )
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            alphabet_string = f.read().strip()
        # Index 0 is reserved for the null/EOS token; actual chars start at 1.
        self._alphabet = ['\x00'] + list(alphabet_string)

        print(f"[{split}] Loaded {len(self.strokes)} samples.  "
              f"Alphabet size: {len(self._alphabet)} (incl. null).")

    # ------------------------------------------------------------------
    # Alphabet helpers
    # ------------------------------------------------------------------

    def get_alphabet(self) -> list[str]:
        return self._alphabet

    def get_alphabet_size(self) -> int:
        return len(self._alphabet)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.strokes.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            'stroke':          torch.tensor(self.strokes[idx],    dtype=torch.float32),
            'stroke_len':      torch.tensor(self.strokes_len[idx], dtype=torch.int),
            'chars':           torch.tensor(self.chars[idx],       dtype=torch.long),
            'chars_len':       torch.tensor(self.chars_len[idx],   dtype=torch.int),
            'writer_id':       torch.tensor(self.writer_ids[idx],  dtype=torch.int),
            'stroke_filename': self.source_files[idx],
        }


if __name__ == '__main__':
    from config.config import load_config
    config = load_config()
    processed_dir = config.paths.processed_data_dir

    for split in ('train', 'val', 'test'):
        ds = ProcessedHandwritingDataset(processed_dir=processed_dir, split=split)
        dl = DataLoader(ds, batch_size=32, shuffle=(split == 'train'))
        batch = next(iter(dl))
        print(f"{split} — stroke batch: {batch['stroke'].shape}, "
              f"chars batch: {batch['chars'].shape}")
