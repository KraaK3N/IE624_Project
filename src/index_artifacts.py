"""Check that the vector store and BM25 input belong to the same build."""

import hashlib
import json
from pathlib import Path


def read_chunks(path: Path) -> tuple[list[dict], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file is missing: {path}")
    contents = path.read_bytes()
    chunks = []
    for number, line in enumerate(contents.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
            if not isinstance(chunk, dict) or not isinstance(chunk.get("id"), str) or not chunk["id"]:
                raise ValueError("expected a chunk object with a nonempty ID")
        except ValueError as exc:
            raise ValueError(f"Invalid chunk at {path}:{number}: {exc}") from exc
        chunks.append(chunk)
    return chunks, hashlib.sha256(contents).hexdigest()


def verify_index(collection, chunks: list[dict], chunks_sha256: str) -> None:
    chunk_ids = {chunk["id"] for chunk in chunks}
    indexed_ids = set(collection.get(include=[])["ids"])
    if not chunk_ids or len(chunk_ids) != len(chunks):
        raise ValueError("chunks.jsonl must contain at least one chunk and have unique IDs")
    if chunk_ids != indexed_ids:
        raise ValueError(
            "Chroma and chunks.jsonl have different IDs "
            f"({len(indexed_ids - chunk_ids)} only in Chroma, {len(chunk_ids - indexed_ids)} only in chunks.jsonl). "
            "Set VECTOR_STORE_PATH and CHUNKS_PATH to the same build."
        )
    expected_hash = (collection.metadata or {}).get("chunks_sha256")
    if expected_hash is not None and chunks_sha256 != expected_hash:
        raise ValueError(
            "chunks.jsonl does not match the file used to build Chroma. "
            "Set VECTOR_STORE_PATH and CHUNKS_PATH to the same build."
        )
