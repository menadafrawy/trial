"""Persistent storage for human feedback samples (accepted / rejected strokes)."""
import json
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

_FEEDBACK_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "data" / "feedback"


class FeedbackStore:
    def __init__(self):
        _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        self._data_file = _FEEDBACK_DIR / "feedback_data.json"
        self._data: dict = {"samples": []}
        self._load()

    def _load(self):
        if self._data_file.exists():
            try:
                with open(self._data_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {"samples": []}

    def _save(self):
        with open(self._data_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    def save_sample(
        self,
        char: str,
        strokes: list[list[float]],
        bias: float,
        accepted: bool,
    ) -> str:
        sample_id = str(uuid.uuid4())
        self._data["samples"].append({
            "id": sample_id,
            "char": char,
            "strokes": strokes,
            "bias": bias,
            "timestamp": datetime.utcnow().isoformat(),
            "accepted": accepted,
        })
        self._save()
        return sample_id

    def get_accepted(self) -> list[dict]:
        return [s for s in self._data["samples"] if s["accepted"]]

    def get_stats(self) -> dict[str, int]:
        return dict(Counter(s["char"] for s in self.get_accepted()))

    def total_accepted(self) -> int:
        return sum(1 for s in self._data["samples"] if s["accepted"])
