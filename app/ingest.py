from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from typing import List, Tuple
from bs4 import BeautifulSoup
from pypdf import PdfReader
from .models import IngestStats


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
CORPUS_PATH = INDEX_DIR / "corpus.jsonl"


def read_html(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def read_chat_csv(path: Path) -> str:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg = row.get("message") or row.get("text") or " ".join(row.values())
            if msg:
                lines.append(str(msg))
    return "\n".join(lines)


def chunk_text(text: str, window: int = 600, overlap: int = 120) -> List[Tuple[int, int, str]]:
    clean = re.sub(r"\s+", " ", text).strip()
    chunks: List[Tuple[int, int, str]] = []
    if not clean:
        return chunks
    start = 0
    while start < len(clean):
        end = min(len(clean), start + window)
        boundary = clean.rfind(".", start, end)
        if boundary != -1 and boundary > start + 0.5 * window:
            end = boundary + 1
        chunk_text_value = clean[start:end].strip()
        if chunk_text_value:
            chunks.append((start, end, chunk_text_value))
        if end == len(clean):
            break
        start = max(end - overlap, 0)
    return chunks


def ingest() -> IngestStats:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "fund_letter": DATA_DIR / "fund_letters" / "q2_letter.html",
        "addendum": DATA_DIR / "fund_letters" / "q2_macro_addendum.pdf",
        "chat": DATA_DIR / "chat_logs" / "desk_chat.csv",
    }

    documents = 0
    chunks_count = 0
    bytes_read = 0
    written_sources: List[str] = []

    with CORPUS_PATH.open("w", encoding="utf-8") as out:
        if sources["fund_letter"].exists():
            txt = read_html(sources["fund_letter"])
            bytes_read += len(txt.encode("utf-8"))
            for s, e, t in chunk_text(txt):
                out.write(json.dumps({
                    "source": str(sources["fund_letter"].relative_to(REPO_ROOT)),
                    "start": s,
                    "end": e,
                    "text": t,
                }) + "\n")
                chunks_count += 1
            documents += 1
            written_sources.append(str(sources["fund_letter"].relative_to(REPO_ROOT)))

        if sources["addendum"].exists():
            txt = read_pdf(sources["addendum"])
            bytes_read += len(txt.encode("utf-8"))
            for s, e, t in chunk_text(txt):
                out.write(json.dumps({
                    "source": str(sources["addendum"].relative_to(REPO_ROOT)),
                    "start": s,
                    "end": e,
                    "text": t,
                }) + "\n")
                chunks_count += 1
            documents += 1
            written_sources.append(str(sources["addendum"].relative_to(REPO_ROOT)))

        if sources["chat"].exists():
            txt = read_chat_csv(sources["chat"])
            bytes_read += len(txt.encode("utf-8"))
            for s, e, t in chunk_text(txt):
                out.write(json.dumps({
                    "source": str(sources["chat"].relative_to(REPO_ROOT)),
                    "start": s,
                    "end": e,
                    "text": t,
                }) + "\n")
                chunks_count += 1
            documents += 1
            written_sources.append(str(sources["chat"].relative_to(REPO_ROOT)))

    return IngestStats(documents=documents, chunks=chunks_count, bytes_read=bytes_read, sources=written_sources)


if __name__ == "__main__":
    stats = ingest()
    print(json.dumps(stats.model_dump(), indent=2))

