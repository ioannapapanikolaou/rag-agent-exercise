from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi.testclient import TestClient

from app.main import app


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_QUERIES = REPO_ROOT / "eval" / "queries.jsonl"
RESULTS_PATH = REPO_ROOT / "data" / "eval_results.jsonl"


def load_queries() -> List[str]:
    qs: List[str] = []
    with EVAL_QUERIES.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qs.append(obj["q"])
    return qs


def main() -> None:
    client = TestClient(app)
    # Ensure ingestion has been run
    client.post("/ingest")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as out:
        for q in load_queries():
            resp = client.post("/answer", json={"question": q, "k": 5})
            resp.raise_for_status()
            out.write(json.dumps({"q": q, "resp": resp.json()}) + "\n")
            print(f"Q: {q}\nA: {resp.json().get('answer','')[:200]}...\n")


if __name__ == "__main__":
    main()