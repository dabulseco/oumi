from __future__ import annotations

import streamlit as st

from rag.core.settings import get_settings
from rag.index.embedder import Embedder
from rag.index.milvus_client import MilvusIndex
from rag.index.sparse_index import load_sparse_corpus
from rag.retrieve.hybrid_search import hybrid_search
from rag.llm.ollama_client import answer_with_context, list_models


st.set_page_config(page_title="Local Oumi RAG", layout="wide")
st.title("Local Oumi RAG Assistant")

settings = get_settings()

with st.sidebar:
    st.header("Settings")
    models = list_models()
    if not models:
        st.warning("No Ollama models found from `ollama list`.")
    model_name = st.selectbox("Ollama model", options=models or [settings.default_ollama_model])
    top_k = st.slider("Top K final", min_value=3, max_value=25, value=settings.top_k_final)

question = st.text_area("Ask anything about Oumi features and usage", height=120)

if st.button("Ask") and question.strip():
    try:
        with st.spinner("Retrieving context and generating answer..."):
            embedder = Embedder(settings.embedding_model)
            emb_dim = len(embedder.encode(["dimension probe"])[0])
            milvus = MilvusIndex(settings.milvus_uri, settings.milvus_collection, dim=emb_dim)
            sparse_docs = load_sparse_corpus(settings.cache_dir)

            hits = hybrid_search(
                question,
                embedder=embedder,
                milvus=milvus,
                sparse_docs=sparse_docs,
                top_k_dense=settings.top_k_dense,
                top_k_sparse=settings.top_k_sparse,
                top_k_final=top_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
            )
            context = [
                {
                    "rel_path": h.rel_path,
                    "start_line": h.start_line,
                    "end_line": h.end_line,
                    "text": h.text,
                }
                for h in hits
            ]
            response = answer_with_context(model_name, question, context)

        st.subheader("Answer")
        st.write(response)

        st.subheader("Sources")
        for i, h in enumerate(hits, 1):
            with st.expander(f"{i}. {h.rel_path}:{h.start_line}-{h.end_line} ({h.doc_type})"):
                st.caption(f"score={h.score:.4f}")
                st.code(h.text[:2000])
    except Exception as exc:
        st.error(f"Query failed: {exc}")
