# IE624 Course Project: SysCon Department Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to answer academic, curriculum, and administrative queries for the Systems & Control Engineering (SysCon) department using verified official documentation.

## 1. Core Architecture Pillars

1. **Knowledge Base Definition:** Lock down exact department data sources—course handbooks, credit requirements, faculty research directories, lab manuals, and academic calendars.
2. **Data Ingestion & Chunking:** Department documents are heavy on PDFs and complex tables (syllabi, timetables, grading rules). Use a layout-aware parser that preserves table structures (such as `Docling` or `PyMuPDF4LLM`) rather than plain text splitting.
3. **Hybrid Retrieval Pipeline:** Combine dense vector embeddings (for conceptual queries like *"Who works on stochastic control?"*) with keyword search/BM25 (for exact string matches like course codes *"IE624"*, *"SC625"*, or room numbers).
4. **Guardrails & Citations:** Force the LLM to cite the exact document/page and explicitly state *"I don't have that information"* when a query falls outside official documents to prevent hallucinated academic policies.

## 2. Core Project Deliverables

* **Curated SysCon Dataset:** A clean, structured repository of raw and processed data (M.Tech/Ph.D. curriculum rules, course syllabi, faculty profiles, and lab details) tagged with metadata (`doc_type`, `source`, `year`).
* **End-to-End RAG Codebase:** Modular implementation covering table-aware PDF ingestion, hybrid indexing (Vector Store + BM25), cross-encoder reranking, and an LLM generation layer with strict citation prompts.
* **Benchmark "Golden" QA Dataset:** A manually verified test set of 50–100 realistic SysCon queries paired with ground-truth answers and exact source documents.
* **Quantitative Evaluation & Ablation Study:** Automated evaluation using RAGAS metrics (*Context Recall*, *Faithfulness*, and *Answer Relevance*) comparing baseline dense vector search against hybrid search and reranking.
* **Lightweight Demo & Final Report:** An interactive Streamlit or Gradio web interface displaying answers alongside expandable source chunks, accompanied by a technical report analyzing system architecture and failure cases.

## 3. Execution Roadmap

### Stage 1: Scope & Knowledge Base Curation
Collect and organize all official SysCon department data, including M.Tech/Ph.D. rulebooks, course syllabi, faculty directories, and lab webpages, tagging each file with source and year metadata. Simultaneously, create a manually verified "Golden QA Dataset" of 50 to 75 realistic student and administrative queries paired with ground-truth answers to serve as the final evaluation benchmark.

### Stage 2: Table-Aware Ingestion & Chunking
Convert raw PDFs and web pages into clean Markdown using layout-aware parsers like `Docling` or `PyMuPDF4LLM` so complex curriculum tables, credit structures, and grading rules remain intact. Split these processed documents into 300–500 token chunks with a 10% overlap, embedding parent-document metadata directly into every chunk to preserve context during lookup.

### Stage 3: Hybrid Retrieval & Reranking
Build a dual-index search pipeline combining dense vector embeddings (via `ChromaDB` or `Qdrant`) for conceptual queries with sparse BM25 keyword search for exact string matches like course codes and room numbers. Merge the retrieved results using Reciprocal Rank Fusion (RRF) and pass the top candidates through a cross-encoder reranker (`bge-reranker-base`) to feed only the top 5 most relevant chunks to the generator.

### Stage 4: LLM Synthesis & Guardrailing
Integrate an instruction-tuned LLM (such as `Llama-3.1-8B` or `Qwen-2.5-7B`) governed by a strict system prompt to synthesize retrieved chunks into concise, accurate responses with inline source citations. Implement confidence thresholds and abstention guardrails so the bot explicitly declines to answer when retrieval scores are low, eliminating hallucinated academic policies.

### Stage 5: Ablation Evaluation & UI Deployment
Benchmark the pipeline against the Golden QA Dataset using RAGAS metrics (*Context Recall*, *Faithfulness*, and *Answer Relevance*) and run an ablation study comparing dense-only search, hybrid search, and reranking. Wrap the finalized pipeline in a clean Streamlit or Gradio web interface that displays the curated answer alongside expandable source snippets for user verification.