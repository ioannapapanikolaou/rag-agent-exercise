from __future__ import annotations

import os
from typing import List, Dict
import requests

try:  # optional hosted provider
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def provider() -> str:
    return os.getenv("LLM_PROVIDER", os.getenv("OPENAI_PROVIDER", "ollama")).lower()


def model_name() -> str:
    if provider() == "ollama":
        return os.getenv("LLM_MODEL", "llama3.2:3b")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def is_configured() -> bool:
    # Local mode is always considered available (no credits needed)
    if provider() == "ollama":
        return True
    # Hosted OpenAI requires key
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None


def _openai_client() -> "OpenAI":
    assert OpenAI is not None, "openai package not installed"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _ollama_chat(system_prompt: str, user_prompt: str) -> str | None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model_name(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # spec: {message: {content: "..."}}
    return data.get("message", {}).get("content")


def synthesize_answer(system_prompt: str, question: str, contexts: List[Dict[str, str]]) -> str:
    """Synthesize an answer using configured provider (Ollama local by default).

    contexts: list of {"tag": "doc@start:end", "text": "..."}
    """
    user_content_lines = [
        "You are a strict, citation-first assistant.",
        "Use ONLY the provided context. If the answer is not in context, say you don't know.",
        "Every claim MUST include a citation tag copied EXACTLY from the allowed tags below.",
        "Be concise (<= 5 sentences).",
        f"Question: {question}",
        "Allowed citation tags (use EXACTLY as-is; do NOT invent new tags):",
    ]
    for c in contexts:
        user_content_lines.append(f"- [{c['tag']}]")
    user_content_lines.append("Context chunks:")
    for c in contexts:
        user_content_lines.append(f"[{c['tag']}]")
        user_content_lines.append(c["text"])
    user_content = "\n".join(user_content_lines)

    if provider() == "ollama":
        out = _ollama_chat(system_prompt, user_content)
        return (out or "").strip()

    # Hosted OpenAI fallback
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model_name(),
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


