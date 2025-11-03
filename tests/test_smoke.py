import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Ensure local tests don't try to call any LLM provider
os.environ.setdefault("LLM_PROVIDER", "off")


@pytest.fixture(scope="session")
def run_ingestion():
    from app.ingest import ingest

    stats = ingest()
    return stats


def test_ingest_outputs(run_ingestion):
    repo = Path(__file__).resolve().parents[1]
    corpus = repo / "data" / "index" / "corpus.jsonl"
    assert corpus.exists(), "corpus.jsonl should be created by ingest"
    assert run_ingestion.documents >= 1
    assert run_ingestion.chunks > 0


def test_retriever_search(run_ingestion):
    from app.retriever import HybridRetriever

    r = HybridRetriever()
    results = r.search("SPY", k=3)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert {"source", "start", "end", "text"}.issubset(results[0].keys())


def test_agent_price_query_latest():
    from app.agent import answer

    resp = answer("What is the most recent close for MSFT?")
    assert "MSFT" in resp.answer
    assert any("prices_stub/prices.json" in s for s in resp.sources)
    assert resp.metrics is not None
    assert resp.metrics.route in ("price", "rag", "rag+llm")


def test_agent_rag_query_with_citations(run_ingestion):
    from app.agent import answer

    resp = answer("Did the Q2 letter reference SPY?")
    assert resp.citations, "RAG answer should include citations"
    # Accept any citation coming from the ingested corpus under data/
    assert any(c.source.startswith("data/") for c in resp.citations)


def test_api_endpoints(run_ingestion):
    from app.main import app

    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    r = client.post("/answer", json={"question": "What was the last price for EURUSD?", "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data

