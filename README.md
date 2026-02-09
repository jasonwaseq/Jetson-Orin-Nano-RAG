# Orin Nano Offline RAG (Qwen2.5 + FAISS + OCR)

A fully offline Retrieval-Augmented Generation (RAG) system designed for NVIDIA Jetson Orin Nano.  
Upload PDFs → build a local vector index → ask questions → get answers with citations (doc + page), **without any cloud dependencies**.

## Features
- **Fully offline**: generation via `llama.cpp` (GGUF), retrieval via FAISS
- **PDF ingestion**: text extraction with **OCR fallback** for scanned PDFs (Tesseract)
- **Incremental indexing**: SHA256 hashing skips unchanged PDFs (fast rebuilds)
- **Document registry**: tracks indexed docs + stats (`registry.json`)
- **Better retrieval**: MMR selection for diverse, high-signal context
- **Hallucination guard**: low-evidence threshold returns “I don’t know”
- **Streamlit UI**: upload PDFs, build index, query, view sources/snippets

## Architecture (High-Level)
1. **Ingest**
   - PDF → text extraction (`pypdf`)
   - If little/no text → OCR (`pdf2image` + `tesseract`)
   - Chunking (overlap) → Embeddings (`bge-small-en-v1.5`) → FAISS index
   - Store metadata per chunk (doc/page/text) + registry (hashes/stats)

2. **Query**
   - Embed question
   - Retrieve candidate chunks (FAISS)
   - Select final chunks using **MMR**
   - Prompt Qwen2.5 (GGUF) via `llama.cpp`
   - Display answer + citations + sources

## Requirements
### Hardware
- NVIDIA Jetson Orin Nano (8GB recommended)
- SSD strongly recommended for models + indexing

### System packages
```bash
sudo apt update
sudo apt install -y git cmake build-essential python3-venv
sudo apt install -y tesseract-ocr poppler-utils

