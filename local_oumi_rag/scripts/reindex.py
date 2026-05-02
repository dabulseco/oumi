from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.core.settings import get_settings
from rag.index.build_index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild local Oumi RAG index")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root to ingest")
    args = parser.parse_args()

    settings = get_settings()
    stats = build_index(settings, repo_root=args.repo_root.resolve())
    print(f"Indexed files={stats.get('files', 0)} chunks={stats.get('chunks', 0)} dim={stats.get('vector_dim', 'n/a')}")


if __name__ == "__main__":
    main()
