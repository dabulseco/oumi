from __future__ import annotations

import subprocess
from ollama import Client


def list_models() -> list[str]:
    try:
        res = subprocess.run(["ollama", "list"], check=True, text=True, capture_output=True)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        return [ln.split()[0] for ln in lines[1:]] if len(lines) > 1 else []
    except Exception:
        return []


def answer_with_context(model: str, question: str, context_chunks: list[dict]) -> str:
    client = Client()
    context_text = "\n\n".join(
        f"[source: {c['rel_path']}:{c.get('start_line', 0)}-{c.get('end_line', 0)}]\n{c['text']}" for c in context_chunks
    )
    prompt = (
        "You are a repository-grounded assistant for Oumi. "
        "Answer ONLY from context. Include inline citations like [path:start-end]. "
        "If not in context, clearly say so.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n"
    )
    return client.generate(model=model, prompt=prompt).get("response", "")
