"""Stage 3 - chunk data/processed/corpus.jsonl and index it into ChromaDB.

    python scripts/index.py                    # full rebuild (core + research tiers)
    python scripts/index.py --tiers core       # department content only (much faster)

Every chunk carries a `tier`:
    core      department pages + department documents + faculty homepages
    research  faculty papers/books/theses/slides and seminar slide decks
Retrieval should filter to tier=core by default (see the test queries at the end).

Outputs (data/processed/):
    chunks.jsonl        all chunks + metadata (also the input for BM25 later)
    chroma_db/          vector store, collection "syscon"
    index_report.json   chunk statistics
"""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
PROC = ROOT / "data" / "processed"
CORPUS, CHUNKS, DB_DIR, REPORT = PROC / "corpus.jsonl", PROC / "chunks.jsonl", PROC / "chroma_db", PROC / "index_report.json"

COLLECTION = "syscon"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "  # bge: prefix queries only

TARGET_TOKENS = 350      # pack blocks up to this size
MAX_BLOCK_TOKENS = 400   # larger paragraphs/tables are split
OVERLAP_TOKENS = 50      # sentence overlap between consecutive chunks of one section
MIN_SECTION_TOKENS = 120  # smaller sections are merged with the next one
MIN_CHUNK_TOKENS = 20
MIN_ALNUM_RATIO = 0.5    # drops OCR garbage / symbol soup

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
SENT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[\"'])")
GENERIC_CTX_RE = re.compile(r"^\W*(click\s*here|here|link|download|more|read\s*more|details?|view|pdf)\W*$", re.I)
RESEARCH_PAGE_RE = re.compile(r"pub|paper|gallery|lecture|thesis|slides", re.I)


# ---------------- text prep ----------------
def clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = text.replace("�", "").replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tier_of(doc: dict) -> str:
    url = doc["url"]
    if "/archiveTalks/" in url:
        return "research"
    if "/~" in url:
        if doc["source_kind"] != "html" or RESEARCH_PAGE_RE.search(url):
            return "research"
    return "core"


def blocks(text: str):
    """Yield ('heading', level, text) | ('table', 0, text) | ('para', 0, text)."""
    para, table = [], []

    def flush():
        if para:
            yield "para", 0, " ".join(para) if not any(l.lstrip().startswith(("-", "*", "+")) or re.match(r"\s*\d+[.)]\s", l) for l in para) else "\n".join(para)
            para.clear()
        if table:
            yield "table", 0, "\n".join(table)
            table.clear()

    for line in text.splitlines():
        s = line.strip()
        h = HEADING_RE.match(s)
        if h:
            yield from flush()
            title = re.sub(r"<[^>]+>|\*\*|__", "", h.group(2)).strip("* ")
            if title:
                yield "heading", len(h.group(1)), title
        elif s.startswith("|"):
            if para:
                yield from flush()
            table.append(s)
        elif not s:
            yield from flush()
        else:
            if table:
                yield from flush()
            para.append(s)
    yield from flush()


# ---------------- chunker ----------------
class Chunker:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def n(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def window_split(self, text: str, limit: int) -> list[str]:
        ids = self.tok.encode(text, add_special_tokens=False)
        return [self.tok.decode(ids[i:i + limit]) for i in range(0, len(ids), limit)]

    def split_block(self, kind: str, text: str) -> list[str]:
        if self.n(text) <= MAX_BLOCK_TOKENS:
            return [text]
        if kind == "table":
            rows = text.splitlines()
            head = rows[:2] if len(rows) > 1 and SEP_RE.match(rows[1]) else []
            body, out, cur = rows[len(head):], [], []
            used = head_tok = self.n("\n".join(head))
            for row in body:
                t = self.n(row) + 1
                if cur and used + t > MAX_BLOCK_TOKENS:
                    out.append("\n".join(head + cur))
                    cur, used = [], head_tok
                cur.append(row)
                used += t
            if cur:
                out.append("\n".join(head + cur))
            return [p for piece in out for p in (self.window_split(piece, MAX_BLOCK_TOKENS) if self.n(piece) > MAX_BLOCK_TOKENS else [piece])]
        out, cur, used = [], "", 0
        for sent in SENT_RE.split(text):
            t = self.n(sent)
            if t > MAX_BLOCK_TOKENS:
                if cur:
                    out.append(cur)
                    cur, used = "", 0
                out.extend(self.window_split(sent, MAX_BLOCK_TOKENS))
            elif cur and used + t > MAX_BLOCK_TOKENS:
                out.append(cur)
                cur, used = sent, t
            else:
                cur, used = f"{cur} {sent}".strip(), used + t
        if cur:
            out.append(cur)
        return out

    def tail(self, piece: str) -> str:
        if piece.lstrip().startswith(("|", "#")):
            return ""
        out = ""
        for sent in reversed(SENT_RE.split(piece)):
            cand = f"{sent} {out}".strip()
            if self.n(cand) > OVERLAP_TOKENS:
                break
            out = cand
        return out

    def chunk(self, text: str) -> list[tuple[str, str]]:
        """Return [(section_path, body)]."""
        chunks, cur, cur_tok, cur_path, stack = [], [], 0, "", []

        def emit(overlap: bool):
            nonlocal cur, cur_tok
            if any(not p.startswith("#") for p in cur):
                chunks.append((cur_path, "\n\n".join(cur)))
            carry = self.tail(cur[-1]) if overlap and cur else ""
            cur, cur_tok = ([carry], self.n(carry)) if carry else ([], 0)

        for kind, level, content in blocks(text):
            if kind == "heading":
                if cur_tok >= MIN_SECTION_TOKENS:
                    emit(overlap=False)
                stack[:] = [s for s in stack if s[0] < level] + [(level, content)]
                if not cur:
                    cur_path = " > ".join(t for _, t in stack)
                cur.append(f"{'#' * level} {content}")
                cur_tok += self.n(content)
                continue
            for piece in self.split_block(kind, content):
                t = self.n(piece)
                if cur and cur_tok + t > TARGET_TOKENS:
                    emit(overlap=kind == "para")
                if not cur:
                    cur_path = " > ".join(s for _, s in stack)
                cur.append(piece)
                cur_tok += t
        if cur:
            emit(overlap=False)
        return chunks


# ---------------- main ----------------
def alnum_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalnum() for c in chars) / max(len(chars), 1)


def build_chunks(docs: list[dict], chunker: Chunker, tiers: set[str]) -> tuple[list[dict], Counter]:
    out, seen, stats = [], set(), Counter()
    for doc in docs:
        tier = tier_of(doc)
        if tier not in tiers:
            stats[f"skipped_tier_{tier}"] += 1
            continue
        title = doc["title"].strip()
        ctx = next((c for c in doc.get("link_context") or []
                    if c and c.lower() != title.lower() and not GENERIC_CTX_RE.match(c)), "")
        pieces = chunker.chunk(clean(doc["text"]))
        kept = []
        for path, body in pieces:
            norm = re.sub(r"\W+", " ", body).lower().strip()
            key = hashlib.sha1(norm.encode()).hexdigest()
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            if chunker.n(body) < MIN_CHUNK_TOKENS or alnum_ratio(body) < MIN_ALNUM_RATIO:
                stats["dropped_junk"] += 1
                continue
            seen.add(key)
            kept.append((path, body))
        for i, (path, body) in enumerate(kept):
            parts = path.split(" > ") if path else []
            while parts and parts[0].lower() == title.lower():
                parts.pop(0)
            header = " > ".join([title, *parts])
            if ctx and doc["source_kind"] != "html":
                header += f"\n(Linked on the website as: {ctx[:200]})"
            out.append({
                "id": f"{doc['id']}-{i:04d}",
                "doc_id": doc["id"],
                "url": doc["url"],
                "title": title,
                "section_path": path,
                "doc_type": doc["doc_type"],
                "tier": tier,
                "source_kind": doc["source_kind"],
                "chunk_index": i,
                "n_chunks": len(kept),
                "latest_year": doc.get("latest_year") or -1,
                "last_modified": doc.get("last_modified") or "",
                "fetched_at": doc.get("fetched_at") or "",
                "n_tokens": chunker.n(body),
                "text": body,
                "embed_text": f"{header}\n\n{body}",
            })
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tiers", default="core,research", help="comma list: core,research")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    tiers = {t.strip() for t in args.tiers.split(",")}

    print(f"Loading {args.model} on {args.device} ...")
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = 512

    docs = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    t0 = time.time()
    chunks, stats = build_chunks(docs, Chunker(model.tokenizer), tiers)
    with open(CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks from {len(docs)} docs in {time.time() - t0:.0f}s -> {CHUNKS.relative_to(ROOT)}")

    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]["embed_text"]))  # length-sorted = faster batching
    emb = model.encode([chunks[i]["embed_text"] for i in order], batch_size=args.batch_size,
                       normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    vectors = [None] * len(chunks)
    for pos, i in enumerate(order):
        vectors[i] = emb[pos].tolist()

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - collection did not exist
        pass
    col = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model, "query_instruction": QUERY_INSTRUCTION},
    )
    meta_keys = ["doc_id", "url", "title", "section_path", "doc_type", "tier", "source_kind",
                 "chunk_index", "n_chunks", "latest_year", "last_modified", "fetched_at", "n_tokens"]
    step = min(client.get_max_batch_size(), 2000)
    for s in range(0, len(chunks), step):
        batch = chunks[s:s + step]
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors[s:s + step],
            documents=[c["embed_text"] for c in batch],
            metadatas=[{k: c[k] for k in meta_keys} for c in batch],
        )
    print(f"Indexed {col.count()} chunks -> {DB_DIR.relative_to(ROOT)} (collection '{COLLECTION}')")

    toks = [c["n_tokens"] for c in chunks]
    REPORT.write_text(json.dumps({
        "model": args.model,
        "chunks": len(chunks),
        "by_tier": dict(Counter(c["tier"] for c in chunks)),
        "by_doc_type": dict(Counter(c["doc_type"] for c in chunks).most_common()),
        "tokens": {"mean": round(sum(toks) / max(len(toks), 1)), "max": max(toks, default=0),
                   "over_512": sum(t > 512 for t in toks)},
        **stats,
        "largest_docs": Counter(c["url"] for c in chunks).most_common(10),
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nSanity check (tier=core):")
    for q in ["Who works on stochastic control?", "What is the eligibility for M.Tech admission?",
              "How many credits are required for the PhD course work?"]:
        qv = model.encode([QUERY_INSTRUCTION + q], normalize_embeddings=True).tolist()
        where = {"tier": "core"} if "core" in tiers else None
        res = col.query(query_embeddings=qv, n_results=3, where=where)
        print(f"\nQ: {q}")
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            print(f"  {1 - dist:.3f}  {meta['title'][:60]} | {meta['section_path'][:40]} | {meta['url']}")


if __name__ == "__main__":
    main()
