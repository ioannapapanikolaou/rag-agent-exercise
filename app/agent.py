from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .models import AnswerResponse, Citation, Metrics
from .observability import timed, record_metric
from .retriever import HybridRetriever
from .tools import prices as prices_tool
from . import llm


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "prompts" / "answer_system.txt"

# load the system prompt from the prompts/answer_system.txt file
def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return "Answer using provided context only; cite chunks like [doc@start:end]."


SYSTEM_PROMPT = _load_system_prompt()

# check if the query is a price query
def _is_price_query(q: str) -> bool:
    ql = q.lower()
    # route to price tool when price-related intent is explicit
    return bool(re.search(r"\b(price|close|open|high|low|last|latest|compare|performance|return|percentage|pct)\b", ql))

# extract the symbols from the query
def _extract_symbols(q: str) -> List[str]:
    known = set(prices_tool.list_symbols())
    # Symbols are uppercase letters/numbers typically 2-6 chars; include FX like EURUSD
    candidates = set(re.findall(r"\b[A-Z]{2,10}\b", q))
    return [s for s in candidates if s in known]


_RETRIEVER: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever(alpha=0.65)
    return _RETRIEVER


def answer(question: str, k: int = 5) -> AnswerResponse:
    if _is_price_query(question):
        extra = {"q": question, "used_tools": ["prices_tool"], "route": "price"}
        with timed("price_answer", extra):
            resp = _answer_price(question)
            if resp.metrics and resp.metrics.extra is not None:
                resp.metrics.extra.update({"used_tools": extra["used_tools"]})
            return resp
    extra = {"q": question, "k": k, "used_tools": ["retriever"], "route": "rag"}
    with timed("rag_answer", extra):
        retriever = _get_retriever()
        results = retriever.search(question, k=k)
        if not results:
            return AnswerResponse(
                answer="I couldn't find relevant context in the provided documents.",
                citations=[],
                sources=[],
                metrics=Metrics(latency_ms=0.0, retrieved_k=0, route="rag", extra={"used_tools": extra["used_tools"]}),
            )
        # If LLM configured, synthesize; otherwise do extractive join
        if llm.is_configured():
            contexts = [
                {"tag": f"{r['source']}@{r['start']}:{r['end']}", "text": r["text"]}
                for r in results
            ]
            model = llm.model_name()
            answer_text = llm.synthesize_answer(SYSTEM_PROMPT, question, contexts)
            # Sanitize citations to only allowed tags; append defaults if missing
            allowed_tags = [c["tag"] for c in contexts]
            answer_text = _clean_citations(answer_text, allowed_tags, fallback_tags=allowed_tags[:3])
            extra["used_tools"].append(f"{llm.provider()}:{model}")
            extra["model"] = model
            extra["route"] = "rag+llm"
            route = "rag+llm"
            record_metric("llm_call", {"provider": llm.provider(), "model": model, "q": question})
        else:
            citation_strs = [f"[{r['source']}@{r['start']}:{r['end']}]" for r in results]
            composed = "\n\n".join([r["text"] for r in results])
            answer_text = f"{SYSTEM_PROMPT}\n\n" + composed + "\n\n" + " ".join(citation_strs)
            route = "rag"
        citations = [Citation(source=r["source"], start=r["start"], end=r["end"]) for r in results]
        sources = sorted({r["source"] for r in results})
        return AnswerResponse(
            answer=answer_text,
            citations=citations,
            sources=sources,
            metrics=Metrics(latency_ms=0.0, retrieved_k=len(results), route=route, extra={"used_tools": extra["used_tools"], "model": extra.get("model")}),
        )


def _clean_citations(text: str, allowed_tags: List[str], fallback_tags: List[str]) -> str:
    """Ensure only allowed [tag] citations remain; add fallbacks if none.

    - Removes any [..] that aren't exactly in allowed_tags
    - If no allowed citations remain, appends fallback tags at the end
    """
    if not text:
        return text
    # find bracketed tokens
    found = list(re.finditer(r"\[([^\]]+)\]", text))
    keep_spans = []
    for m in found:
        inner = m.group(1)
        if inner in allowed_tags:
            keep_spans.append(m.span())
        else:
            # remove invalid citation
            text = text[: m.start()] + text[m.end() :]
    # check if any allowed citations remain after removals
    if not any(True for _ in re.finditer(r"\[(?:" + "|".join(map(re.escape, allowed_tags)) + ")\]", text)):
        if fallback_tags:
            text = text.rstrip() + " " + " ".join(f"[{t}]" for t in fallback_tags)
    return text


def _answer_price(question: str) -> AnswerResponse:
    q = question.strip()
    syms = _extract_symbols(q)
    citations: List[Citation] = []
    sources: List[str] = ["prices_stub/prices.json"]

    # simple patterns to extract the symbols from the query
    m_recent = re.search(r"most recent close for (\b[A-Z]{2,10}\b)", q, re.IGNORECASE)
    m_last_price = re.search(r"last price for (\b[A-Z]{2,10}\b)", q, re.IGNORECASE)
    m_compare = re.search(r"compare (\b[A-Z]{2,10}\b) performance to (\b[A-Z]{2,10}\b).*?(\d+)\s*day", q, re.IGNORECASE)

    if m_recent or m_last_price:
        sym = (m_recent.group(1) if m_recent else m_last_price.group(1)).upper()
        latest = prices_tool.get_latest_close(sym)
        if latest is None:
            return AnswerResponse(
                answer=f"No price data available for {sym}.",
                citations=[],
                sources=sources,
                metrics=Metrics(latency_ms=0.0, retrieved_k=0, route="price", extra={}),
            )
        citations.append(Citation(source=sources[0], start=0, end=0))
        return AnswerResponse(
            answer=f"Latest close for {sym} was {latest['close']} on {latest['date']}. [prices_stub/prices.json@0:0]",
            citations=citations,
            sources=sources,
            metrics=Metrics(latency_ms=0.0, retrieved_k=1, route="price", extra={}),
        )

    # compare the performance of two symbols over a given number of points
    if m_compare:
        a, b, days = m_compare.group(1).upper(), m_compare.group(2).upper(), int(m_compare.group(3))
        perf = prices_tool.compare_performance(a, b, points=days)
        if perf is None:
            msg = []
            if prices_tool.get_latest_n(a, 2) is None:
                msg.append(f"{a} not available")
            if prices_tool.get_latest_n(b, 2) is None:
                msg.append(f"{b} not available")
            detail = ", ".join(msg) if msg else "insufficient data"
            return AnswerResponse(
                answer=f"Cannot compare {a} vs {b}: {detail}.",
                citations=[],
                sources=sources,
                metrics=Metrics(latency_ms=0.0, retrieved_k=0, route="price", extra={}),
            )
        citations.append(Citation(source=sources[0], start=0, end=0))
        a_pct = round(perf["a_return"] * 100, 2)
        b_pct = round(perf["b_return"] * 100, 2)
        rel = round(perf["relative"] * 100, 2)
        return AnswerResponse(
            answer=(
                f"Over ~{days} points, {a} returned {a_pct}%, {b} returned {b_pct}% (relative {rel}% for {a}). "
                "[prices_stub/prices.json@0:0]"
            ),
            citations=citations,
            sources=sources,
            metrics=Metrics(latency_ms=0.0, retrieved_k=1, route="price", extra={}),
        )

    # fallback: if a single known symbol is present, treat as latest
    if syms:
        sym = syms[0]
        latest = prices_tool.get_latest_close(sym)
        if latest is None:
            return AnswerResponse(
                answer=f"No price data available for {sym}.",
                citations=[],
                sources=sources,
                metrics=Metrics(latency_ms=0.0, retrieved_k=0, route="price", extra={}),
            )
        return AnswerResponse(
            answer=f"Latest close for {sym} was {latest['close']} on {latest['date']}. [prices_stub/prices.json@0:0]",
            citations=[Citation(source=sources[0], start=0, end=0)],
            sources=sources,
            metrics=Metrics(latency_ms=0.0, retrieved_k=1, route="price", extra={}),
        )

    return AnswerResponse(
        answer="I can answer price questions for available symbols: " + ", ".join(prices_tool.list_symbols()) + ".",
        citations=[],
        sources=["prices_stub/prices.json"],
        metrics=Metrics(latency_ms=0.0, retrieved_k=0, route="price", extra={}),
    )
