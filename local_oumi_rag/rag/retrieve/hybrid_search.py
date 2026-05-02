from __future__ import annotations

from dataclasses import dataclass

from rag.index.embedder import Embedder
from rag.index.milvus_client import MilvusIndex
from rag.index.sparse_index import SparseIndex


@dataclass(slots=True)
class RetrievedChunk:
    rel_path: str
    doc_type: str
    start_line: int
    end_line: int
    text: str
    score: float


def hybrid_search(
    query: str,
    *,
    embedder: Embedder,
    milvus: MilvusIndex,
    sparse_docs: list[dict],
    top_k_dense: int,
    top_k_sparse: int,
    top_k_final: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[RetrievedChunk]:
    qv = embedder.encode([query])[0]
    dense_hits = milvus.search(query_vector=qv, top_k=top_k_dense)

    combined: dict[tuple[str, int, int], RetrievedChunk] = {}
    for h in dense_hits:
        ent = h.entity
        key = (ent.get("rel_path"), int(ent.get("start_line") or 0), int(ent.get("end_line") or 0))
        combined[key] = RetrievedChunk(
            rel_path=ent.get("rel_path"),
            doc_type=ent.get("doc_type"),
            start_line=int(ent.get("start_line") or 0),
            end_line=int(ent.get("end_line") or 0),
            text=ent.get("text"),
            score=dense_weight * float(h.score),
        )

    if sparse_docs:
        sparse = SparseIndex([d["text"] for d in sparse_docs])
        sparse_hits = sparse.search(query, top_k=top_k_sparse)
        max_sparse = max([h.score for h in sparse_hits], default=1.0)
        for hit in sparse_hits:
            d = sparse_docs[hit.idx]
            key = (d["rel_path"], int(d.get("start_line", 0)), int(d.get("end_line", 0)))
            s = sparse_weight * (hit.score / max_sparse if max_sparse else 0.0)
            if key in combined:
                combined[key].score += s
            else:
                combined[key] = RetrievedChunk(
                    rel_path=d["rel_path"],
                    doc_type=d.get("doc_type", "unknown"),
                    start_line=int(d.get("start_line", 0)),
                    end_line=int(d.get("end_line", 0)),
                    text=d["text"],
                    score=s,
                )

    return sorted(combined.values(), key=lambda x: x.score, reverse=True)[:top_k_final]
