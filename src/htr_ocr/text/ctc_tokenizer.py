import json
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CTCTokenizer:
    id2char: list[str]  # не включает бланк; id 1 соответствует id2char[0]

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return 1 + len(self.id2char)

    def char2id(self) -> dict[str, int]:
        return {ch: i + 1 for i, ch in enumerate(self.id2char)}

    def encode(self, text: str) -> list[int]:
        m = self.char2id()
        ids: list[int] = []
        for ch in text:
            if ch in m:
                ids.append(m[ch])
            else:
                raise ValueError(f"Unknown character {ch!r}")
        return ids

    def decode_greedy(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        for i in ids:
            if i == self.blank_id:
                continue
            j = int(i) - 1
            if 0 <= j < len(self.id2char):
                out.append(self.id2char[j])
        return "".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {"blank_id": 0, "id2char": self.id2char}
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def from_dict(obj: dict) -> "CTCTokenizer":
        id2char = obj["id2char"]
        if not isinstance(id2char, list):
            raise ValueError("id2char should be a list")
        return CTCTokenizer(id2char=[str(x) for x in id2char])

    @staticmethod
    def load(path: str | Path) -> "CTCTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return CTCTokenizer.from_dict(obj)


def build_charset(texts: Iterable[str]) -> list[str]:
    chars: set[str] = set()
    for t in texts:
        chars.update(list(t))
    return sorted(chars)

def build_or_load_vocab(cfg) -> CTCTokenizer:
    vocab_path = Path(getattr(cfg.train, "vocab_path", Path(cfg.data.processed_dir) / "vocab_ctc.json"))
    if vocab_path.exists():
        return CTCTokenizer.load(vocab_path)

    train_csv = Path(cfg.data.processed_dir) / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError("train.csv not found")

    df = pd.read_csv(train_csv)
    charset = build_charset(df["text"].astype(str).tolist())
    tokenizer = CTCTokenizer(id2char=charset)
    tokenizer.save(vocab_path)
    return tokenizer
