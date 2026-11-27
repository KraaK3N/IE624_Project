"""Build a fresh SysBot index from a baseline corpus and locally curated files."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.chunking import Chunker, build_chunks
from src.index_artifacts import read_chunks, verify_index
from src.knowledge import load_documents

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / "data" / "processed" / "corpus.jsonl"
DEFAULT_SOURCES = ROOT / "data" / "knowledge" / "sources.json"
BUILDS = ROOT / "data" / "builds"
COLLECTION = "syscon"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SUPPORTED_TIERS = {"core", "research"}
MODEL_TOKEN_LIMIT = 512


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_index(docs: list[dict], output: Path, args: argparse.Namespace, tiers: set[str]) -> None:
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} from the local model cache", flush=True)
    model = SentenceTransformer(args.model, device=device, local_files_only=True)
    model.max_seq_length = min(model.max_seq_length, MODEL_TOKEN_LIMIT)
    chunker = Chunker(model.tokenizer, model.max_seq_length)
    chunks, stats = build_chunks(docs, chunker, tiers)
    if not chunks:
        raise ValueError(f"No searchable chunks in the selected tiers ({', '.join(sorted(tiers))}).")

    print(f"Embedding {len(chunks):,} chunks from {len(docs):,} documents", flush=True)
    write_jsonl(output / "corpus.jsonl", docs)
    chunks_path = output / "chunks.jsonl"
    write_jsonl(chunks_path, chunks)
    written_chunks, chunks_hash = read_chunks(chunks_path)
    if any(chunk["embedding_tokens"] > model.max_seq_length for chunk in written_chunks):
        raise ValueError(f"Embedding text exceeds the model limit of {model.max_seq_length} tokens.")

    client = chromadb.PersistentClient(
        path=str(output / "chroma_db"), settings=chromadb.Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.create_collection(
            args.collection,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": args.model,
                "query_instruction": QUERY_INSTRUCTION,
                "chunks_sha256": chunks_hash,
            },
        )
        written_chunks.sort(key=lambda chunk: chunk["embedding_tokens"])
        step = min(client.get_max_batch_size(), 1024)
        for start in range(0, len(written_chunks), step):
            batch = written_chunks[start:start + step]
            documents = [chunk["embed_text"] for chunk in batch]
            vectors = model.encode(
                documents, batch_size=args.batch_size, normalize_embeddings=True,
                show_progress_bar=True, convert_to_numpy=True,
            )
            collection.add(
                ids=[chunk["id"] for chunk in batch],
                embeddings=vectors.tolist(),
                documents=documents,
                metadatas=[{key: value for key, value in chunk.items() if key not in {"id", "text", "embed_text"}}
                           for chunk in batch],
            )
            print(f"Indexed {start + len(batch):,}/{len(written_chunks):,} chunks", flush=True)

        verify_index(collection, written_chunks, chunks_hash)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    tokens = [chunk["n_tokens"] for chunk in written_chunks]
    embed_tokens = [chunk["embedding_tokens"] for chunk in written_chunks]
    report = {
        "model": args.model,
        "collection": args.collection,
        "documents": len(docs),
        "indexed_documents": len({chunk["doc_id"] for chunk in written_chunks}),
        "chunks": len(written_chunks),
        "by_tier": dict(Counter(chunk["tier"] for chunk in written_chunks)),
        "by_doc_type": dict(Counter(chunk["doc_type"] for chunk in written_chunks).most_common()),
        "tokens": {"mean": round(sum(tokens) / len(tokens)), "max": max(tokens)},
        "embedding_tokens": {"max": max(embed_tokens), "limit": model.max_seq_length},
        "verified_chunk_ids": len(written_chunks),
        "chunks_sha256": chunks_hash,
        **stats,
        "largest_docs": Counter(chunk["url"] for chunk in written_chunks).most_common(10),
    }
    (output / "index_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    baseline_options = parser.add_mutually_exclusive_group()
    baseline_options.add_argument("--baseline", type=Path,
                                  help="baseline corpus.jsonl (defaults to data/processed/corpus.jsonl if present)")
    baseline_options.add_argument("--no-baseline", action="store_true",
                                  help="build from the local source manifest only")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="path to the local sources.json manifest")
    parser.add_argument("--output-dir", type=Path, help="new build directory (defaults to data/builds/<UTC timestamp>)")
    parser.add_argument("--collection", default=COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="cached embedding model ID or local model directory")
    parser.add_argument("--tiers", default="core,research", help="comma-separated tiers to index")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="torch device, defaults to CUDA when available or CPU")
    args = parser.parse_args()
    tiers = {tier.strip() for tier in args.tiers.split(",")}
    if not tiers <= SUPPORTED_TIERS:
        parser.error("--tiers must contain core, research, or both")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if Path(args.model).is_dir():
        args.model = str(Path(args.model).resolve())

    baseline = args.baseline
    if baseline is None and not args.no_baseline and DEFAULT_BASELINE.exists():
        baseline = DEFAULT_BASELINE
    output = args.output_dir or BUILDS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists() or output.is_symlink():
        parser.error(f"Build output already exists: {output}. Choose a fresh directory.")

    try:
        docs = load_documents(baseline, args.sources)
        if not docs:
            raise ValueError("No documents supplied. Register local sources or pass --baseline corpus.jsonl.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            stage = Path(temporary) / "build"
            stage.mkdir()
            build_index(docs, stage, args, tiers)
            # Serialize publication of the same output name without exposing a partial build.
            publication_lock = output.parent / f".{output.name}.publish"
            publication_lock.mkdir()
            try:
                if output.exists() or output.is_symlink():
                    raise FileExistsError(f"Build output already exists: {output}")
                stage.rename(output)
            finally:
                publication_lock.rmdir()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")

    print(f"Verified build: {output.resolve()}")
    print(f"VECTOR_STORE_PATH={output.resolve() / 'chroma_db'}")
    print(f"CHUNKS_PATH={output.resolve() / 'chunks.jsonl'}")
    print(f"COLLECTION_NAME={args.collection}")


if __name__ == "__main__":
    main()
