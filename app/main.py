from __future__ import annotations

import sys
import time
from typing import Optional

from fastapi import FastAPI

from .agent import answer as agent_answer
from .ingest import ingest as run_ingest
from .models import AnswerResponse, QueryRequest


app = FastAPI(title="Front-Office RAG Agent")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_endpoint():
    stats = run_ingest()
    return stats.model_dump()


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: QueryRequest) -> AnswerResponse:
    t0 = time.perf_counter()
    resp = agent_answer(req.question, k=req.k)
    dur_ms = (time.perf_counter() - t0) * 1000.0
    # Fill observed latency into metrics
    if resp.metrics is None:
        from .models import Metrics

        resp.metrics = Metrics(latency_ms=dur_ms, retrieved_k=0, route="unknown", extra={})
    else:
        resp.metrics.latency_ms = dur_ms
    return resp


def _cli(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        # no args -> start server
        import uvicorn

        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
        return 0
    cmd = argv[0]
    if cmd == "ingest":
        stats = run_ingest()
        import json

        print(json.dumps(stats.model_dump(), indent=2))
        return 0
    if cmd == "query":
        # naive parser: expect -q "question" and optional -k N
        q = ""
        k = 5
        i = 1
        while i < len(argv):
            if argv[i] in ("-q", "--question") and i + 1 < len(argv):
                q = argv[i + 1]
                i += 2
            elif argv[i] in ("-k", "--k") and i + 1 < len(argv):
                try:
                    k = int(argv[i + 1])
                except Exception:
                    k = 5
                i += 2
            else:
                i += 1
        if not q:
            print("Provide a question with -q '...'")
            return 2
        resp = agent_answer(q, k=k)
        print(resp.answer)
        return 0
    print("Unknown command. Use one of: ingest, query")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
