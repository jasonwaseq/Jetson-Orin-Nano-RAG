import os
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
import requests

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# OCR deps (optional)
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str


# -----------------------------
# Index filenames
# -----------------------------
INDEX_FILE = "faiss.index"
META_FILE = "meta.json"
REGISTRY_FILE = "registry.json"


# -----------------------------
# Simple in-process caches
# -----------------------------
_EMBED_CACHE: Dict[str, SentenceTransformer] = {}
_INDEX_CACHE = {"index_dir": None, "index": None, "meta": None, "meta_mtime": None}


def get_embed_model(name: str) -> SentenceTransformer:
    m = _EMBED_CACHE.get(name)
    if m is None:
        m = SentenceTransformer(name, device="cpu")
        _EMBED_CACHE[name] = m
    return m


def load_index_cached(index_dir: str):
    meta_path = os.path.join(index_dir, META_FILE)
    mtime = os.path.getmtime(meta_path) if os.path.exists(meta_path) else None
    if (
        _INDEX_CACHE["index_dir"] == index_dir
        and _INDEX_CACHE["index"] is not None
        and _INDEX_CACHE["meta"] is not None
        and _INDEX_CACHE["meta_mtime"] == mtime
    ):
        return _INDEX_CACHE["index"], _INDEX_CACHE["meta"]

    index, meta = load_index(index_dir)
    _INDEX_CACHE.update({"index_dir": index_dir, "index": index, "meta": meta, "meta_mtime": mtime})
    return index, meta


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = " ".join(s.split())
    return s.strip()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_pdfs(raw_dir: str) -> List[str]:
    if not os.path.isdir(raw_dir):
        return []
    return sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")])


# -----------------------------
# Text extraction
# -----------------------------
def extract_text_pypdf(pdf_path: str, max_pages: Optional[int] = None) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    n = len(reader.pages)
    if max_pages is not None:
        n = min(n, max_pages)

    out: List[Tuple[int, str]] = []
    for i in range(n):
        t = (reader.pages[i].extract_text() or "")
        out.append((i + 1, _clean_text(t)))
    return out


def extract_text_ocr(pdf_path: str, dpi: int = 200, max_pages: Optional[int] = None) -> List[Tuple[int, str]]:
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR requested but OCR deps not available. Install:\n"
            "sudo apt install -y tesseract-ocr poppler-utils\n"
            "pip install pytesseract pdf2image pillow"
        )
    images = convert_from_path(pdf_path, dpi=dpi)
    if max_pages is not None:
        images = images[:max_pages]

    out: List[Tuple[int, str]] = []
    for idx, img in enumerate(images):
        txt = pytesseract.image_to_string(img)
        out.append((idx + 1, _clean_text(txt)))
    return out


def pages_to_chunks(
    doc_id: str,
    pages_text: List[Tuple[int, str]],
    chunk_chars: int = 2200,
    overlap_chars: int = 200,
    min_chunk_chars: int = 200,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, text in pages_text:
        if not text:
            continue

        start = 0
        L = len(text)
        while start < L:
            end = min(L, start + chunk_chars)
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= min_chunk_chars:
                cid_src = f"{doc_id}|{page_num}|{start}|{end}"
                chunk_id = hashlib.sha256(cid_src.encode("utf-8")).hexdigest()[:16]
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, page=page_num, text=chunk_text))
            if end == L:
                break
            start = max(0, end - overlap_chars)
    return chunks


# -----------------------------
# Embedding + FAISS
# -----------------------------
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embs, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("No embeddings to index (0 vectors).")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, meta: List[Dict[str, Any]], index_dir: str) -> None:
    ensure_dir(index_dir)
    faiss.write_index(index, os.path.join(index_dir, INDEX_FILE))
    save_json(os.path.join(index_dir, META_FILE), meta)


def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_path = os.path.join(index_dir, INDEX_FILE)
    meta_path = os.path.join(index_dir, META_FILE)
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Index not found. Click 'Build / Rebuild' first.")
    index = faiss.read_index(index_path)
    meta = load_json(meta_path, [])
    return index, meta


# -----------------------------
# Registry (incremental indexing)
# -----------------------------
def load_registry(index_dir: str) -> Dict[str, Any]:
    reg = load_json(os.path.join(index_dir, REGISTRY_FILE), {"docs": {}})
    reg.setdefault("docs", {})
    return reg


def save_registry(index_dir: str, reg: Dict[str, Any]) -> None:
    ensure_dir(index_dir)
    save_json(os.path.join(index_dir, REGISTRY_FILE), reg)


def ingest_pdfs(
    raw_dir: str,
    index_dir: str,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    chunk_chars: int = 2200,
    overlap_chars: int = 200,
    ocr_dpi: int = 200,
    max_pages: Optional[int] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    ensure_dir(raw_dir)
    ensure_dir(index_dir)

    pdfs = list_pdfs(raw_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    reg = load_registry(index_dir)
    docs_reg: Dict[str, Any] = reg["docs"]

    # Load existing meta so we can reuse unchanged docs
    existing_meta = load_json(os.path.join(index_dir, META_FILE), [])
    existing_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for m in existing_meta:
        existing_by_doc.setdefault(m["doc_id"], []).append(m)

    plan = []
    for p in pdfs:
        doc_id = os.path.basename(p)
        h = sha256_file(p)
        prev = docs_reg.get(doc_id)
        if prev and prev.get("sha256") == h and not force_rebuild:
            status = "unchanged"
        elif prev and not force_rebuild:
            status = "changed"
        else:
            status = "new" if not prev else "rebuild"
        plan.append({"doc_id": doc_id, "path": p, "sha256": h, "status": status})

    embed_model = get_embed_model(embed_model_name)

    new_meta: List[Dict[str, Any]] = []
    per_doc_stats = []
    reused_docs = 0
    processed_docs = 0

    for item in plan:
        doc_id = item["doc_id"]
        status = item["status"]
        pdf_path = item["path"]
        h = item["sha256"]

        if status == "unchanged" and doc_id in existing_by_doc:
            new_meta.extend(existing_by_doc[doc_id])
            reused_docs += 1
            per_doc_stats.append({"doc_id": doc_id, "status": "reused", "chunks": len(existing_by_doc[doc_id])})
            continue

        processed_docs += 1

        pages = extract_text_pypdf(pdf_path, max_pages=max_pages)
        chars = sum(len(t) for _, t in pages)

        used_ocr = False
        if chars < 500:
            used_ocr = True
            pages = extract_text_ocr(pdf_path, dpi=ocr_dpi, max_pages=max_pages)

        chunks = pages_to_chunks(
            doc_id=doc_id,
            pages_text=pages,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=200,
        )

        if not chunks:
            raise ValueError(f"0 chunks produced for {doc_id} (OCR may be missing or empty).")

        for c in chunks:
            new_meta.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page, "text": c.text})

        docs_reg[doc_id] = {
            "sha256": h,
            "indexed_at": now_iso(),
            "used_ocr": used_ocr,
            "page_count": len(pages),
            "extracted_chars": sum(len(t) for _, t in pages),
            "chunks": len(chunks),
        }

        per_doc_stats.append({"doc_id": doc_id, "status": status, "used_ocr": used_ocr, "chunks": len(chunks)})

    # Rebuild FAISS index from all chunks (simple + reliable)
    texts = [m["text"] for m in new_meta]
    embs = embed_texts(embed_model, texts, batch_size=32)
    index = build_faiss_index(embs)
    save_index(index, new_meta, index_dir)

    save_registry(index_dir, reg)

    return {
        "pdf_count": len(pdfs),
        "processed_docs": processed_docs,
        "reused_docs": reused_docs,
        "total_chunks": len(new_meta),
        "per_doc": per_doc_stats,
        "ocr_available": OCR_AVAILABLE,
        "force_rebuild": force_rebuild,
    }


def delete_doc_and_rebuild(index_dir: str, doc_id: str) -> Dict[str, Any]:
    meta = load_json(os.path.join(index_dir, META_FILE), [])
    new_meta = [m for m in meta if m.get("doc_id") != doc_id]

    reg = load_registry(index_dir)
    reg["docs"].pop(doc_id, None)
    save_registry(index_dir, reg)

    if not new_meta:
        # wipe index files
        for fn in [INDEX_FILE, META_FILE]:
            p = os.path.join(index_dir, fn)
            if os.path.exists(p):
                os.remove(p)
        return {"deleted": doc_id, "remaining_chunks": 0, "note": "Index cleared (no docs left)."}

    embed_model = get_embed_model("BAAI/bge-small-en-v1.5")
    texts = [m["text"] for m in new_meta]
    embs = embed_texts(embed_model, texts, batch_size=32)
    index = build_faiss_index(embs)
    save_index(index, new_meta, index_dir)

    return {"deleted": doc_id, "remaining_chunks": len(new_meta)}


# -----------------------------
# Retrieval (simple + stable; no per-query re-embedding of chunks)
# -----------------------------
def retrieve(
    embed_model: SentenceTransformer,
    index: faiss.Index,
    meta: List[Dict[str, Any]],
    query: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    q = embed_texts(embed_model, [query], batch_size=1)
    scores, ids = index.search(q, top_k)
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        item = dict(meta[idx])
        item["score"] = float(score)
        results.append(item)
    return results


# -----------------------------
# LLM call via llama-server (OpenAI-compatible)
# -----------------------------
def make_messages(query: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[SOURCE: {r['doc_id']} p.{r['page']}]\n{r['text']}")
    context = "\n\n".join(context_blocks)

    system = (
        "You are an assistant doing Retrieval-Augmented Generation (RAG) fully offline.\n"
        "Rules:\n"
        "1) Answer ONLY using the provided SOURCES.\n"
        "2) If the SOURCES do not contain enough info, say you don't know.\n"
        "3) Always cite sources inline like: (doc p.X).\n"
        "4) Be concise and technical.\n"
    )

    user = f"Question: {query}\n\nSOURCES:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def llama_server_chat(
    server_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout_s: int = 60,
) -> str:
    # llama-server supports OpenAI-style endpoint on many builds:
    # POST /v1/chat/completions
    url = server_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    # OpenAI-style response
    return (data["choices"][0]["message"]["content"] or "").strip()


def answer_query(
    index_dir: str,
    query: str,
    llama_server_url: str,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    top_k: int = 6,
    evidence_threshold: float = 0.35,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    t0 = time.time()
    index, meta = load_index_cached(index_dir)
    embed_model = get_embed_model(embed_model_name)
    t1 = time.time()

    retrieved = retrieve(embed_model, index, meta, query, top_k=top_k)
    t2 = time.time()

    if not retrieved:
        return {
            "answer": "I don't know — I couldn't retrieve relevant context from the indexed documents.",
            "retrieved": [],
            "timing": {"load_s": round(t1 - t0, 3), "retrieve_s": round(t2 - t1, 3), "generate_s": 0.0, "total_s": round(t2 - t0, 3)},
        }

    best_score = float(retrieved[0].get("score", 0.0))
    if best_score < evidence_threshold:
        return {
            "answer": "I don't know — the documents I have indexed don’t provide strong enough evidence to answer that confidently.",
            "retrieved": retrieved,
            "timing": {"load_s": round(t1 - t0, 3), "retrieve_s": round(t2 - t1, 3), "generate_s": 0.0, "total_s": round(t2 - t0, 3)},
            "note": f"best_score={best_score:.3f} < threshold={evidence_threshold:.3f}",
        }

    messages = make_messages(query, retrieved)
    out = llama_server_chat(
        server_url=llama_server_url,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_s=60,
    )
    # Safety: cap returned content so Streamlit never freezes
    MAX_CHARS = 8000
    if len(out) > MAX_CHARS:
        out = out[:MAX_CHARS] + "\n\n[Output truncated]"

    t3 = time.time()
    return {
        "answer": out,
        "retrieved": retrieved,
        "timing": {"load_s": round(t1 - t0, 3), "retrieve_s": round(t2 - t1, 3), "generate_s": round(t3 - t2, 3), "total_s": round(t3 - t0, 3)},
        "best_score": best_score,
    }
