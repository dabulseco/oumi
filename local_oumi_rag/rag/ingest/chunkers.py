from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(slots=True)
class Chunk:
    id: str
    rel_path: str
    doc_type: str
    text: str
    start_line: int
    end_line: int


def chunk_text(text: str, rel_path: str, doc_type: str, max_chars: int = 2600, overlap_lines: int = 5) -> list[Chunk]:
    """Line-aware chunking for better readability and citations.

    - Preserves line boundaries to keep config/code blocks coherent.
    - Produces line-range metadata for source citation in answers.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    i = 0
    n = len(lines)

    while i < n:
        start = i
        cur_chars = 0
        buf: list[str] = []
        while i < n:
            next_len = len(lines[i]) + 1
            if buf and cur_chars + next_len > max_chars:
                break
            buf.append(lines[i])
            cur_chars += next_len
            i += 1

        chunk_text_value = "\n".join(buf).strip()
        if chunk_text_value:
            digest = hashlib.sha1(f"{rel_path}:{doc_type}:{start}:{chunk_text_value}".encode()).hexdigest()[:20]
            chunks.append(
                Chunk(
                    id=digest,
                    rel_path=rel_path,
                    doc_type=doc_type,
                    text=chunk_text_value,
                    start_line=start + 1,
                    end_line=start + len(buf),
                )
            )

        if i >= n:
            break
        i = max(start + 1, i - overlap_lines)

    return chunks
