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
```
## Setup   
1) Build llama.cpp (with CUDA)   
cd ~   
git clone https://github.com/ggml-org/llama.cpp.git   
cd llama.cpp   
cmake -B build -DGGML_CUDA=ON   
cmake --build build -j   

Verify:   
ls -l ~/llama.cpp/build/bin/llama-cli   

## 2) Create Python environment + install dependencies   

From the repo root:   

python3 -m venv .venv   
source .venv/bin/activate   
   
pip install --upgrade pip   
pip install faiss-cpu sentence-transformers pypdf streamlit numpy   
pip install pytesseract pdf2image pillow   

## 3) Add your GGUF model   

Place your Qwen2.5 7B instruct GGUF here:   

models/llm/qwen2.5-7b-instruct-q4_k_m.gguf   


If your filename differs, update LLM_MODEL in app/app.py.   

## 4) Run the app   
source .venv/bin/activate   
streamlit run app/app.py   

Usage   

Upload PDF(s) in the sidebar   

Click Build / Rebuild (incremental by default)   

Ask a question   

Expand Sources Used to see cited context (doc + page)   

Notes on scanned PDFs   

If the PDF is scanned (images), OCR is used automatically:   

Requires: tesseract-ocr, poppler-utils, pytesseract, pdf2image   

Project Structure   
orin-rag/      
   app/      
       app.py          # Streamlit UI   
       rag.py          # ingestion + retrieval + generation   
   data/      
   raw/            # uploaded PDFs (ignored by git)   
   index/          # FAISS index + metadata + registry (ignored by git)   
   models/   
      llm/            # GGUF model(s) (ignored by git)      

Configuration / Tuning   

Inside the UI under “Advanced”:   

Top-k: number of final chunks passed to the model   

Candidate chunks: pre-MMR pool size   

MMR lambda: higher = more relevant, lower = more diverse   

Evidence threshold: if best similarity is below this, answer returns “I don’t know”   

n_ctx: context window (try 4096 first)   

gpu_layers (-ngl): 999 offloads as much as possible   

Troubleshooting   
“Index not found”   

Upload PDFs and click Build / Rebuild first.   
Expected files (ignored by git):   

data/index/faiss.index   

data/index/meta.json   

data/index/registry.json   

OCR not working / empty extraction   

Make sure these are installed:   

sudo apt install -y tesseract-ocr poppler-utils   
pip install pytesseract pdf2image pillow   

