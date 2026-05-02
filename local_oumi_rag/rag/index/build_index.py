from __future__ import annotations

from pathlib import Path

from rag.core.settings import Settings
from rag.ingest.scan_repo import scan_repository
from rag.ingest.parsers import parse_file
from rag.ingest.chunkers import chunk_text
from rag.index.embedder import Embedder
from rag.index.milvus_client import MilvusIndex
from rag.index.sparse_index import persist_sparse_corpus


def build_index(settings: Settings, repo_root: Path) -> dict:
    files = scan_repository(repo_root=repo_root, max_file_size_bytes=settings.max_file_size_bytes)

    chunks = []
    for f in files:
        for doc in parse_file(f.path, f.rel_path, f.suffix):
            chunks.extend(chunk_text(doc.text, doc.rel_path, doc.doc_type))

    if not chunks:
        return {"files": 0, "chunks": 0}

    embedder = Embedder(settings.embedding_model)
    vectors = embedder.encode([c.text for c in chunks])
    dim = len(vectors[0])

    milvus = MilvusIndex(settings.milvus_uri, settings.milvus_collection, dim)
    milvus.reset()
    milvus.insert(
        ids=[c.id for c in chunks],
        rel_paths=[c.rel_path for c in chunks],
        doc_types=[c.doc_type for c in chunks],
        start_lines=[c.start_line for c in chunks],
        end_lines=[c.end_line for c in chunks],
        texts=[c.text for c in chunks],
        vectors=vectors,
    )

    sparse_path = persist_sparse_corpus(
        settings.cache_dir,
        [
            {
                "id": c.id,
                "rel_path": c.rel_path,
                "doc_type": c.doc_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "text": c.text,
            }
            for c in chunks
        ],
    )

    return {"files": len(files), "chunks": len(chunks), "vector_dim": dim, "sparse_path": str(sparse_path)}
