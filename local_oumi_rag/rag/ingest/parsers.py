from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import nbformat


@dataclass(slots=True)
class ParsedDoc:
    rel_path: str
    doc_type: str
    text: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_file(path: Path, rel_path: str, suffix: str) -> list[ParsedDoc]:
    if suffix == ".ipynb":
        return parse_notebook(path, rel_path)
    if suffix in {".md", ".rst", ".txt", ".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ""}:
        return [ParsedDoc(rel_path=rel_path, doc_type=f"text:{suffix or 'noext'}", text=_read_text(path))]
    return []


def parse_notebook(path: Path, rel_path: str) -> list[ParsedDoc]:
    nb = nbformat.read(path, as_version=4)
    out: list[ParsedDoc] = []
    for idx, cell in enumerate(nb.cells):
        kind = cell.get("cell_type", "unknown")
        src = cell.get("source", "")
        if not src:
            continue
        out.append(
            ParsedDoc(
                rel_path=rel_path,
                doc_type=f"notebook:{kind}",
                text=f"[cell {idx}]\n{src}",
            )
        )
    return out
