from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "data" / "index" / "corpus.jsonl"


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


class HybridRetriever:
    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_tfidf: List[Dict[str, float]] = []
        self._load()

    def _load(self) -> None:
        if not CORPUS_PATH.exists():
            self.chunks = []
            self.tokenized_corpus = []
            return
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        self.tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._build_tfidf()

    def _build_tfidf(self) -> None:
        # build vocabulary and inverse document frequency (IDF)
        df_counter: Counter[str] = Counter()
        for toks in self.tokenized_corpus:
            df_counter.update(set(toks))
        n_docs = max(1, len(self.tokenized_corpus))
        self.idf = {term: math.log((n_docs + 1) / (df + 1)) + 1.0 for term, df in df_counter.items()}
        self.vocab = {term: i for i, term in enumerate(self.idf.keys())}

        # compute normalized TF-IDF per doc as sparse dicts (sparse vectors)
        self.doc_tfidf = []
        for toks in self.tokenized_corpus:
            tf = Counter(toks)
            vec: Dict[str, float] = {}
            for term, freq in tf.items():
                if term in self.idf:
                    vec[term] = (freq / len(toks)) * self.idf[term]
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for k in list(vec.keys()):
                vec[k] /= norm
            self.doc_tfidf.append(vec)

    # compute TF-IDF vector for a query
    def _tfidf_query(self, query: str) -> Dict[str, float]:
        toks = tokenize(query)
        if not toks:
            return {}
        tf = Counter(toks)
        vec: Dict[str, float] = {}
        for term, freq in tf.items():
            if term in self.idf:
                vec[term] = (freq / len(toks)) * self.idf[term]
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for k in list(vec.keys()):
            vec[k] /= norm
        return vec
        
    # compute cosine similarity between two sparse vectors
    @staticmethod
    def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = 0.0
        for term, w in a.items():
            bw = b.get(term)
            if bw is not None:
                dot += w * bw
        return dot

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.chunks:
            return []

        # BM25 scores
        bm25_scores = self.bm25.get_scores(tokenize(query)) if self.bm25 else np.zeros(len(self.chunks))
        # TF-IDF cosine
        q_vec = self._tfidf_query(query)
        tfidf_scores = np.array([self._cosine_sparse(q_vec, dvec) for dvec in self.doc_tfidf])

        # Combine BM25 and TF-IDF scores
        # normalize scores to 0..1 for combination
        def normalize(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            mn, mx = float(np.min(arr)), float(np.max(arr))
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        bm25_n = normalize(np.array(bm25_scores, dtype=float))
        tfidf_n = normalize(tfidf_scores)
        hybrid = self.alpha * bm25_n + (1.0 - self.alpha) * tfidf_n

        idxs = np.argsort(-hybrid)[:k]
        results: List[Dict] = []
        for i in idxs:
            c = self.chunks[int(i)]
            results.append({
                "source": c["source"],
                "start": c["start"],
                "end": c["end"],
                "text": c["text"],
                "score": float(hybrid[int(i)]),
                "bm25": float(bm25_n[int(i)]),
                "tfidf": float(tfidf_n[int(i)]),
            })
        return results
