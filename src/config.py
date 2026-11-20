import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

load_dotenv(BASE_DIR / ".env")


def _path(env: str, default: Path) -> Path:
    p = Path(os.getenv(env, str(default)))
    return p if p.is_absolute() else BASE_DIR / p


# LLM: "ollama" (local Llama, default) | "anthropic" (Claude) | "openai" (ChatGPT) | "gemini" (Google)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
MODEL_NAME = os.getenv("MODEL_NAME", {"ollama": "llama3.1:8b", "anthropic": "claude-haiku-5-5", "openai": "gpt-5-mini", "gemini": "gemini-3.8-flash"}.get(LLM_PROVIDER, ""))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY= os.getenv("GEMINI_API_KEY", "")
# Answering
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.2"))  # below this the bot abstains

# Retrieval
VECTOR_STORE_PATH = _path("VECTOR_STORE_PATH", PROCESSED_DATA_DIR / "chroma_db")
CHUNKS_PATH = _path("CHUNKS_PATH", PROCESSED_DATA_DIR / "chunks.jsonl")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "syscon")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE", "")  # "", "cpu" or "cuda"; empty = auto
