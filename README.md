# SysBot

SysBot is a retrieval-augmented assistant for the Centre for Systems and Control at IIT Bombay. A router handles small talk and rewrites follow-up questions for search. For department questions, SysBot combines dense ChromaDB retrieval with BM25 using reciprocal rank fusion, then uses a cross-encoder to rerank the results before generating a cited answer. Chainlit is the primary interface.

## Quick start

From the repository root, set up **Python 3.13** and copy the example configuration. Skip the copy if you already have a `.env` file.

```sh
python3.13 -m venv .venv313
.venv313/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

The default provider is Ollama. Install and start Ollama, then run `ollama pull llama3.1:8b`. To use Gemini, OpenAI, or Anthropic, set `LLM_PROVIDER`, `MODEL_NAME`, and the corresponding API key in `.env`. For Anthropic, also run `.venv313/bin/python -m pip install anthropic`. [Ollama installation guide](https://docs.ollama.com/linux)

SysBot needs a matching `chroma_db/` and `chunks.jsonl` from the same build. If you already have both in `data/processed/`, the example configuration points there. For a fresh clone, get a built index from the team or follow [Build the index](#build-the-index). Update both paths in `.env` if the build is elsewhere.

```sh
.venv313/bin/python -m chainlit run src/chainlit_app.py --host 127.0.0.1 --port 8000
```

When Chainlit prints that the app is available, open [http://127.0.0.1:8000](http://127.0.0.1:8000). If the page is blank, check that you used `.venv313/bin/python` and installed the requirements above. The previous Python 3.14 and Chainlit 2.11 setup returned errors for the page's JavaScript and CSS.

The optional Streamlit interface uses the same index and configuration. Run `.venv313/bin/python -m streamlit run app.py`.

## Build the index

Use a saved `corpus.jsonl` from the team, or create one from the department website.

```sh
.venv313/bin/python -m pip install -r crawler/requirements.txt
.venv313/bin/python -m crawler.crawl
.venv313/bin/python -m crawler.parse
```

The crawler saves pages under `data/raw/`. To reuse an existing crawl, run the parser with `--raw-dir PATH`. The parser writes `data/crawler/corpus.jsonl` by default.

Build from the crawler output, or pass the path to a saved snapshot to `--baseline`.

```sh
.venv313/bin/python index.py --baseline data/crawler/corpus.jsonl
```

The first build downloads the embedding model. The indexer creates a timestamped directory under `data/builds/` and prints the settings to copy into `.env`. Set `VECTOR_STORE_PATH` and `CHUNKS_PATH` together, then restart SysBot. The first online start also downloads the reranker.

To check retrieval without generating an answer, run `.venv313/bin/python -m src.retriever "What is SC 625 about?"`.

## Add local sources

1. Put each Markdown or text-based PDF in `data/knowledge/`. For PDF support, install `requirements-ingest.txt`.
2. Add a record to [the source manifest](data/knowledge/sources.json) with the relative `file` path, `title`, official citation `url`, `doc_type`, and `tier`. Use `core` for information such as courses, admissions, and department rules. Use `research` for publications and similar faculty material.
3. Rebuild with your existing corpus as `--baseline`, or use `--no-baseline` to index only the registered sources. Copy the printed paths to `.env` and restart SysBot.

A local source with the same normalized URL replaces the corresponding baseline record. For an image-only scan, add a checked Markdown transcription with the official source URL.

## Evaluation

The course presentation evaluated **25 questions** (20 answerable and 5 unanswerable) on a crawl snapshot with 271 documents and 8,248 chunks. On the 20 answerable questions, hybrid retrieval with reranking achieved **95% Hit@1, 100% Hit@5, and 0.975 MRR@10**. The two final Llama 3.1 8B runs scored **88–92% overall correctness**. Both declined all five unanswerable questions. [Course presentation](report/presentation.pdf#page=12) · [Presented results](evaluation/RESULTS.md)

The repository ships a separate 60-question fixture for testing the current code. [Evaluation guide](evaluation/README.md)
