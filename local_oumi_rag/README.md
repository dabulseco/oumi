# Local Oumi RAG (Option A)

Local-first RAG assistant for the Oumi repository with:
- Streamlit UI
- Milvus vector database
- Hybrid retrieval (dense + sparse)
- Ollama for local generation
- Source-aware citations with line ranges

## Scope
This app ingests repository content broadly (docs, code, config, notebooks, text-like files), excluding noisy/cache paths and likely binaries by default (`.git`, virtual envs, caches, large files, media blobs).

## Quick start
```bash
conda env create -f local_oumi_rag/environment.yml
conda activate local-oumi-rag
python -m pip install -r local_oumi_rag/requirements.txt

# Build/rebuild index and sparse corpus cache
python local_oumi_rag/scripts/reindex.py --repo-root .

# Launch app
streamlit run local_oumi_rag/app/streamlit_app.py
```

## Retrieval design
- Dense retrieval from Milvus vectors.
- Sparse retrieval from persisted local corpus (`local_oumi_rag/.cache/sparse_corpus.jsonl`).
- Hybrid score fusion with configurable dense/sparse weights.

## Robustness notes
- UI catches retrieval/generation exceptions and surfaces actionable errors.
- Ollama model dropdown is populated from local `ollama list`.
