# Front-Office RAG Agent

A small **Retrieval-Augmented Generation (RAG)** service that answers questions grounded in a local dataset of financial research documents.

---

## Overview

This system ingests the following sources:

- **HTML**: `data/fund_letters/q2_letter.html`
- **PDF**: `data/fund_letters/q2_macro_addendum.pdf`
- **Chat CSV**: `data/chat_logs/desk_chat.csv`
- **Prices**: `prices_stub/prices.json`

It retrieves information from these files and produces **grounded, cited answers** using either:
- a **local hybrid sparse retriever** (BM25 + TF-IDF), and optionally
- a **language model (LLM)** such as Ollama or OpenAI.

All actions and timings are logged to `data/metrics.jsonl`.

---

## Architecture

```
Question
   ↓
Retriever  ← corpus.jsonl (chunked documents)
   ↓ top-k chunks
LLM (optional)
   ↓
Cited answer → [doc@start:end]
   ↓
Metrics emitted to data/metrics.jsonl
```

### Key modules
| File | Purpose |
|------|----------|
| `app/ingest.py` | Reads HTML, PDF, and CSV files → normalizes and chunks text into overlapping passages |
| `app/retriever.py` | Performs **fully local hybrid sparse retrieval** (BM25 + TF-IDF) |
| `app/agent.py` | Routes between price tool, retriever, and LLM; sanitizes citations |
| `app/llm.py` | Handles LLM provider selection (off / Ollama / OpenAI) |
| `app/observability.py` | Records latency and metadata to `data/metrics.jsonl` |
| `app/main.py` | FastAPI app exposing `/ingest`, `/answer`, `/healthz` |
| `prices_stub/prices.py` | Local tool for price lookup and comparison |

---

## Key Concepts

- **Chunking** – splits large documents into smaller overlapping text windows so each snippet can be retrieved efficiently.
- **Fully local retrieval** – all search operations run offline using BM25 & TF-IDF (no external API).
- **Tokenization** – breaking text into lowercase words for counting and comparison.
- **Regex** – pattern matching to detect price queries and symbols.
- **System prompt** – fixed LLM instruction:  
  *“Answer using provided context only; cite chunks like [doc@start:end].”*
- **Zero-embeddings retrieval** – keyword-based; contrast to vector-based (embedding) retrieval in Qdrant.

---

## Running it

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optionally start Qdrant (for later use dense retrieval):
```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

---

### Ingest data
```bash
make ingest
```
This reads the HTML, PDF, and CSV sources, chunks them, and writes to:
```
data/index/corpus.jsonl
```

---

### Run the API
```bash
make run
```

Endpoints:
- `POST /ingest` – trigger ingestion
- `POST /answer` – ask a question  
  Example:
  ```bash
  curl -X POST localhost:8000/answer        -H 'content-type: application/json'        -d '{"question":"What did the Q2 letter say about Momentum?","k":5}'
  ```
- `GET /healthz` – check service health

---

## Running Modes

### A. Extractive (no LLM)
Fully offline - returns retrieved text directly.
```bash
export LLM_PROVIDER=off
make ingest && make run
```

### B. Local LLM via Ollama
Install [Ollama](https://ollama.ai) and pull a small model (e.g. `llama3.2:3b`):
```bash
ollama pull llama3.2:3b
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:3b
make ingest && make run
```

### C. Hosted LLM via OpenAI
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
make ingest && make run
```

---

## Evaluation

Run automated evaluation:
```bash
make eval
```
This reads `eval/queries.jsonl`, queries the agent, and logs answers to `data/eval_results.jsonl`.

---

## Metrics

Metrics are appended to `data/metrics.jsonl` automatically, e.g.:

```json
{"ts": 1730483912.51,
 "event": "rag_answer",
 "latency_ms": 145.7,
 "q": "What did the Q2 letter say about Momentum?",
 "k": 5,
 "used_tools": ["retriever", "ollama:llama3.2:3b"],
 "route": "rag+llm"}
```

Inspect them:
```bash
tail -n 5 data/metrics.jsonl | jq .
```

---

## Extending the System

- **Add embeddings + Qdrant** for semantic (dense) retrieval.  
- **Add reranking** to reorder retrieved chunks by relevance.  
- **Add more tools** (e.g., portfolio analyzer, risk stats).  
- **Integrate tracing dashboards** from metrics JSONL for performance monitoring.

---

## In Summary

> The Front-Office RAG Agent performs **fully local hybrid sparse retrieval** over ingested financial documents.  
> It can optionally use an LLM for generation or operate entirely extractively, and logs every query to metrics for transparent evaluation.


