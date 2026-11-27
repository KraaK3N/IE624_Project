"""Hybrid retrieval: dense (Chroma + bge) + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

    python -m src.retriever "Who works on stochastic control?"
    python -m src.retriever "SC 625 credits" --mode bm25 --no-rerank        # ablation
    python -m src.retriever "..." --tiers core,research                      # include papers/books

Modes (for the ablation study): dense | bm25 | hybrid, each with or without reranking.
"""

import argparse
import math
import re
from dataclasses import dataclass, field

import chromadb
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import CHUNKS_PATH, COLLECTION_NAME, DEVICE, RERANKER_MODEL, VECTOR_STORE_PATH
from src.index_artifacts import read_chunks, verify_index

RRF_K = 60
STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it its me my of on or our "
    "please s tell that the their there this to was what when where which who whom why will with you your".split()
)


def bm25_tokens(text: str) -> list[str]:
    t = text.lower()
    t = re.sub(r"\b(m|b|ph|m\.?\s?s)\s*\.\s*(tech|d|sc|phil)\b", r"\1\2", t)       # m.tech -> mtech, ph.d -> phd
    t = re.sub(r"\b([a-z]{2,3})\s*-?\s*(\d{3,4})\b", r"\1\2 \1 \2", t)              # sc 625 -> sc625 sc 625
    return [w for w in re.findall(r"[a-z0-9]+", t) if w not in STOPWORDS]


@dataclass
class Hit:
    id: str
    text: str
    embed_text: str
    meta: dict
    ranks: dict = field(default_factory=dict)
    rrf: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf

    def citation(self) -> str:
        section = f" - {self.meta['section_path']}" if self.meta.get("section_path") else ""
        return f"{self.meta['title']}{section} ({self.meta['url']})"


class Retriever:
    def __init__(self, reranker_model: str = RERANKER_MODEL, device: str = DEVICE, load_reranker: bool = True):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunks, chunks_hash = read_chunks(CHUNKS_PATH)
        if not (VECTOR_STORE_PATH / "chroma.sqlite3").is_file():
            raise FileNotFoundError(f"Chroma database is missing from {VECTOR_STORE_PATH}")
        client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_PATH), settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.col = client.get_collection(COLLECTION_NAME)
        verify_index(self.col, self.chunks, chunks_hash)
        cmeta = self.col.metadata or {}
        self.query_instruction = cmeta.get("query_instruction", "")
        self.embedder = SentenceTransformer(cmeta["embedding_model"], device=device)
        self.reranker = CrossEncoder(reranker_model, max_length=512, device=device) if load_reranker else None

        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([bm25_tokens(c["embed_text"]) for c in self.chunks])

    # ---------- candidate generators ----------
    def dense(self, query: str, k: int, tiers: set[str]) -> list[str]:
        qv = self.embedder.encode([self.query_instruction + query], normalize_embeddings=True).tolist()
        where = {"tier": {"$in": sorted(tiers)}} if len(tiers) > 1 else {"tier": next(iter(tiers))}
        res = self.col.query(query_embeddings=qv, n_results=k, where=where, include=[])
        return res["ids"][0]

    def sparse(self, query: str, k: int, tiers: set[str]) -> list[str]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [self.chunks[i]["id"] for i in order if scores[i] > 0 and self.chunks[i]["tier"] in tiers][:k]

    def doc_chunks(self, doc_id: str, limit: int = 15) -> list[Hit]:
        """All chunks of one document, in reading order (for list-style questions)."""
        cs = sorted((c for c in self.chunks if c["doc_id"] == doc_id), key=lambda c: c["chunk_index"])[:limit]
        return [Hit(c["id"], c["text"], c["embed_text"], {k: v for k, v in c.items() if k not in {"text", "embed_text"}})
                for c in cs]

    # ---------- main entry ----------
    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        rerank: bool = True,
        tiers: tuple[str, ...] = ("core",),
        n_candidates: int = 30,
        rerank_top: int = 20,
        max_per_doc: int = 2,
    ) -> list[Hit]:
        tiers = set(tiers)
        lists = {}
        if mode in {"dense", "hybrid"}:
            lists["dense"] = self.dense(query, n_candidates, tiers)
        if mode in {"bm25", "hybrid"}:
            lists["bm25"] = self.sparse(query, n_candidates, tiers)

        hits: dict[str, Hit] = {}
        for name, ids in lists.items():
            for rank, cid in enumerate(ids, 1):
                c = self.by_id[cid]
                h = hits.setdefault(cid, Hit(cid, c["text"], c["embed_text"], {k_: v for k_, v in c.items() if k_ not in {"text", "embed_text"}}))
                h.ranks[name] = rank
                h.rrf += 1 / (RRF_K + rank)
        ranked = sorted(hits.values(), key=lambda h: h.rrf, reverse=True)

        if rerank and self.reranker is not None and ranked:
            pool = ranked[:rerank_top]
            raw = self.reranker.predict([(query, h.embed_text) for h in pool], batch_size=8, show_progress_bar=False)
            raw = [float(s) for s in raw]
            probs = raw if all(0.0 <= s <= 1.0 for s in raw) else [1 / (1 + math.exp(-s)) for s in raw]
            for h, s in zip(pool, probs):
                h.rerank_score = s
            ranked = sorted(pool, key=lambda h: h.rerank_score, reverse=True)

        out, per_doc = [], {}
        for h in ranked:
            d = h.meta["doc_id"]
            if per_doc.get(d, 0) < max_per_doc:
                out.append(h)
                per_doc[d] = per_doc.get(d, 0) + 1
            if len(out) == k:
                break
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--tiers", default="core")
    a = ap.parse_args()

    r = Retriever(load_reranker=not a.no_rerank)
    queries = [" ".join(a.query)] if a.query else [
        "Who works on stochastic control?",
        "What is the eligibility for M.Tech admission?",
        "How many credits are required for the PhD course work?",
        "What is SC 625 about?",
        "How do I apply for leave as a PhD student?",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        for h in r.search(q, k=a.k, mode=a.mode, rerank=not a.no_rerank, tiers=tuple(a.tiers.split(","))):
            ranks = " ".join(f"{k}#{v}" for k, v in h.ranks.items())
            print(f"  {h.score:.3f} [{ranks:>17}] {h.citation()[:110]}")
