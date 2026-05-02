from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass(slots=True)
class SparseHit:
    idx: int
    score: float


class SparseIndex:
    def __init__(self, texts: list[str]):
        self.texts = texts
        self.tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 20) -> list[SparseHit]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [SparseHit(idx=i, score=float(s)) for i, s in ranked]


def persist_sparse_corpus(cache_dir: Path, chunks: list[dict]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "sparse_corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return path


def load_sparse_corpus(cache_dir: Path) -> list[dict]:
    path = cache_dir / "sparse_corpus.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
