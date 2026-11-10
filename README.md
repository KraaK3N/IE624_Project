# IE624 Course Project: Department AI Chatbot

An AI-powered conversational assistant designed for department students, faculty, and visitors. The chatbot helps answer queries regarding department courses, faculty profiles, curriculum, FAQs, academic deadlines, and department resources.

---

## 📌 Project Overview

- **Course:** IE624
- **Project Title:** Department AI Chatbot
- **Goal:** Develop an intelligent, retrieval-augmented conversational agent tailored to the department's knowledge base.

---

## 📁 Repository Structure

```text
IE624_Project/
├── data/
│   ├── raw/                 # Department handbooks, FAQs, syllabi, web scrapes
│   └── processed/           # Processed chunks, vector store indexes (ChromaDB/FAISS)
├── src/
│   ├── __init__.py
│   ├── config.py            # Project configurations and environment variable loader
│   ├── bot.py               # Chatbot logic, RAG retrieval, and LLM integration
│   └── app.py               # Streamlit web interface
├── .env.example             # Template for API keys and environment variables
├── .gitignore               # Git ignore rules for Python, secrets, and caches
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Set Up Virtual Environment
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API key:
```bash
cp .env.example .env
```

### 5. Run the Chatbot UI
```bash
streamlit run src/app.py
```

---

## 🗺️ Roadmap & Next Steps

1. **Data Ingestion & Preprocessing:**
   - Add department PDFs, course brochures, and FAQ documents into `data/raw/`.
   - Chunk text and generate vector embeddings into `data/processed/`.
2. **RAG Pipeline Implementation:**
   - Implement vector search (e.g. ChromaDB / FAISS) in `src/`.
   - Integrate retriever with LLM (e.g. Gemini / OpenAI / open-source models).
3. **UI / Evaluation:**
   - Enhance the Streamlit frontend with source citations and filters.
   - Run evaluation on sample department queries.
