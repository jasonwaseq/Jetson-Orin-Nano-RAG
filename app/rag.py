import os
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# OCR deps (optional at runtime; handled gracefully)
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
# Paths / filenames (within index_dir)
# -----------------------------
INDEX_FILE = "faiss.index"
META_FILE = "meta.json"
REGISTRY_FILE = "registry.json"


# -----------------------------
# Utilities
# -----------------------------
def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = " ".join(s.split())
    return s.strip()


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


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


# -----------------------------
# PDF text extraction
# -----------------------------
def extract_text_pypdf(pdf_path: str, max_pages: Optional[int] = None) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    n = len(reader.pages)
    if max_pages is not None:
        n = min(n, max_pages)

    out: List[Tuple[int, str]] = []
    for i in range(n):
        t = (reader.pages[i].extract_text() or "")
        t = _clean_text(t)
        out.append((i + 1, t))
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
        txt = _clean_text(txt)
        out.append((idx + 1, txt))
    return out


def pages_to_chunks(
    doc_id: str,
    pages_text: List[Tuple[int, str]],
    chunk_chars: int = 2800,
    overlap_chars: int = 300,
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
                # stable chunk_id derived from doc/page/start/end
                cid_src = f"{doc_id}|{page_num}|{start}|{end}"
                chunk_id = hashlib.sha256(cid_src.encode("utf-8")).hexdigest()[:16]
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, page=page_num, text=chunk_text))
            if end == L:
                break
            start = max(0, end - overlap_chars)
    return chunks


# -----------------------------
# Embeddings / FAISS
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
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized dot product
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
        raise FileNotFoundError("Index not found. Click 'Build / Rebuild Index' first.")
    index = faiss.read_index(index_path)
    meta = load_json(meta_path, [])
    return index, meta


# -----------------------------
# Registry (doc hashes + stats)
# -----------------------------
def load_registry(index_dir: str) -> Dict[str, Any]:
    path = os.path.join(index_dir, REGISTRY_FILE)
    reg = load_json(path, {"docs": {}})
    if "docs" not in reg:
        reg["docs"] = {}
    return reg


def save_registry(index_dir: str, reg: Dict[str, Any]) -> None:
    ensure_dir(index_dir)
    save_json(os.path.join(index_dir, REGISTRY_FILE), reg)


def list_pdfs(raw_dir: str) -> List[str]:
    if not os.path.isdir(raw_dir):
        return []
    return sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")])


# -----------------------------
# Retrieval improvements: MMR + evidence gate
# -----------------------------
def mmr_select(
    query_emb: np.ndarray,
    cand_embs: np.ndarray,
    cand_meta: List[Dict[str, Any]],
    k: int,
    lambda_mult: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance selection.
    query_emb: (d,)
    cand_embs: (n,d) normalized
    """
    if cand_embs.shape[0] == 0:
        return []

    # similarity to query
    sim_q = cand_embs @ query_emb.reshape(-1, 1)
    sim_q = sim_q.reshape(-1)  # (n,)

    selected = []
    selected_idx = []

    # start with best
    first = int(np.argmax(sim_q))
    selected_idx.append(first)

    while len(selected_idx) < min(k, cand_embs.shape[0]):
        best_score = -1e9
        best_i = None

        sel_embs = cand_embs[selected_idx]  # (m,d)
        # similarity to selected set: max cosine to any selected
        sim_sel = cand_embs @ sel_embs.T  # (n,m)
        max_sim_sel = sim_sel.max(axis=1) if sim_sel.ndim == 2 else sim_sel

        for i in range(cand_embs.shape[0]):
            if i in selected_idx:
                continue
            score = lambda_mult * sim_q[i] - (1.0 - lambda_mult) * max_sim_sel[i]
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break
        selected_idx.append(int(best_i))

    for i in selected_idx[:k]:
        item = dict(cand_meta[i])
        item["score"] = float(sim_q[i])
        selected.append(item)

    # sort by score desc for nicer display
    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected


def retrieve_mmr(
    embed_model: SentenceTransformer,
    index: faiss.Index,
    meta: List[Dict[str, Any]],
    query: str,
    top_k: int = 8,
    cand_k: int = 30,
    lambda_mult: float = 0.7,
) -> List[Dict[str, Any]]:
    q_emb = embed_texts(embed_model, [query], batch_size=1)  # (1,d)
    scores, ids = index.search(q_emb, cand_k)
    ids0 = ids[0].tolist()

    cand_meta = []
    cand_texts = []
    for idx in ids0:
        if idx == -1:
            continue
        cand_meta.append(meta[idx])
        cand_texts.append(meta[idx]["text"])

    if not cand_meta:
        return []

    cand_embs = embed_texts(embed_model, cand_texts, batch_size=32)  # (n,d) normalized
    return mmr_select(q_emb[0], cand_embs, cand_meta, k=top_k, lambda_mult=lambda_mult)


# -----------------------------
# Prompt + llama.cpp call
# -----------------------------
def make_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
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

    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {query}\n\n"
        f"SOURCES:\n{context}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def run_llama_cli(
    llama_cli_path: str,
    model_path: str,
    prompt: str,
    n_ctx: int = 4096,
    max_tokens: int = 512,
    temp: float = 0.2,
    top_p: float = 0.9,
    threads: int = 6,
    gpu_layers: int = 999,
) -> str:
    if not os.path.exists(llama_cli_path):
        raise FileNotFoundError(f"llama-cli not found at: {llama_cli_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF model not found at: {model_path}")

    cmd = [
        llama_cli_path,
        "-m", model_path,
        "-c", str(n_ctx),
        "--temp", str(temp),
        "--top-p", str(top_p),
        "-n", str(max_tokens),
        "-t", str(threads),
        "-ngl", str(gpu_layers),
        "--prompt", prompt,
        "--no-display-prompt",
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"llama-cli failed:\nSTDERR:\n{p.stderr}\n\nSTDOUT:\n{p.stdout}")

    return p.stdout.strip()


# -----------------------------
# Ingestion (incremental default)
# -----------------------------
def ingest_pdfs(
    raw_dir: str,
    index_dir: str,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    chunk_chars: int = 2800,
    overlap_chars: int = 300,
    ocr_dpi: int = 200,
    max_pages: Optional[int] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Incremental ingest by default:
    - hashes PDFs
    - skips unchanged PDFs if already indexed
    - rebuilds index from all indexed chunks (simple & reliable)
    force_rebuild=True ignores hashes and rebuilds everything from raw_dir.
    """
    ensure_dir(raw_dir)
    ensure_dir(index_dir)

    pdfs = list_pdfs(raw_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    reg = load_registry(index_dir)
    docs_reg: Dict[str, Any] = reg.get("docs", {})

    # Determine which docs are new/changed
    plan = []
    for p in pdfs:
        doc_id = os.path.basename(p)
        h = sha256_file(p)
        prev = docs_reg.get(doc_id)
        status = "new"
        if prev and prev.get("sha256") == h and not force_rebuild:
            status = "unchanged"
        elif prev and not force_rebuild:
            status = "changed"
        elif prev and force_rebuild:
            status = "rebuild"
        plan.append({"doc_id": doc_id, "path": p, "sha256": h, "status": status})

    # Process docs: for unchanged docs, we can reuse old chunks from meta.json
    existing_meta = load_json(os.path.join(index_dir, META_FILE), [])
    existing_chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for m in existing_meta:
        existing_chunks_by_doc.setdefault(m["doc_id"], []).append(m)

    embed_model = SentenceTransformer(embed_model_name, device="cpu")

    new_meta: List[Dict[str, Any]] = []
    per_doc_stats = []
    reused_docs = 0
    processed_docs = 0

    for item in plan:
        doc_id = item["doc_id"]
        status = item["status"]
        pdf_path = item["path"]
        h = item["sha256"]

        if status == "unchanged" and doc_id in existing_chunks_by_doc:
            # reuse old chunks
            new_meta.extend(existing_chunks_by_doc[doc_id])
            per_doc_stats.append({
                "doc_id": doc_id,
                "status": "reused",
                "used_ocr": docs_reg.get(doc_id, {}).get("used_ocr", False),
                "chunks": len(existing_chunks_by_doc[doc_id]),
                "sha256": h[:12],
            })
            reused_docs += 1
            continue

        # Otherwise, re-process this doc (pypdf -> OCR fallback)
        processed_docs += 1

        pages = extract_text_pypdf(pdf_path, max_pages=max_pages)
        text_chars = sum(len(t) for _, t in pages)

        used_ocr = False
        if text_chars < 500:
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
            raise ValueError(
                f"Produced 0 chunks for {doc_id}. "
                f"If scanned, OCR may be missing or returning empty text."
            )

        # Add to meta
        for c in chunks:
            new_meta.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page": c.page,
                "text": c.text,
            })

        # Update registry entry
        docs_reg[doc_id] = {
            "sha256": h,
            "indexed_at": now_iso(),
            "used_ocr": used_ocr,
            "page_count": len(pages),
            "extracted_chars": sum(len(t) for _, t in pages),
            "chunks": len(chunks),
        }

        per_doc_stats.append({
            "doc_id": doc_id,
            "status": status,
            "used_ocr": used_ocr,
            "chunks": len(chunks),
            "sha256": h[:12],
        })

    # Build FAISS index from all meta (simple & reliable)
    texts = [m["text"] for m in new_meta]
    embs = embed_texts(embed_model, texts, batch_size=32)
    index = build_faiss_index(embs)

    save_index(index, new_meta, index_dir)
    reg["docs"] = docs_reg
    save_registry(index_dir, reg)

    return {
        "pdf_count": len(pdfs),
        "processed_docs": processed_docs,
        "reused_docs": reused_docs,
        "total_chunks": len(new_meta),
        "per_doc": per_doc_stats,
        "index_dir": index_dir,
        "ocr_available": OCR_AVAILABLE,
        "force_rebuild": force_rebuild,
    }


def delete_doc_and_rebuild(index_dir: str, doc_id: str) -> Dict[str, Any]:
    """
    Deletes a doc from registry and rebuilds FAISS index from remaining meta chunks.
    (simple approach; fine for portfolio-sized libraries)
    """
    ensure_dir(index_dir)
    meta_path = os.path.join(index_dir, META_FILE)
    meta = load_json(meta_path, [])
    new_meta = [m for m in meta if m.get("doc_id") != doc_id]

    reg = load_registry(index_dir)
    docs = reg.get("docs", {})
    if doc_id in docs:
        del docs[doc_id]
    reg["docs"] = docs
    save_registry(index_dir, reg)

    if not new_meta:
        # remove index files if empty
        for fn in [INDEX_FILE, META_FILE]:
            p = os.path.join(index_dir, fn)
            if os.path.exists(p):
                os.remove(p)
        return {"deleted": doc_id, "remaining_chunks": 0, "note": "Index cleared (no docs left)."}

    # Rebuild embeddings + FAISS
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    texts = [m["text"] for m in new_meta]
    embs = embed_texts(embed_model, texts, batch_size=32)
    index = build_faiss_index(embs)
    save_index(index, new_meta, index_dir)

    return {"deleted": doc_id, "remaining_chunks": len(new_meta)}


# -----------------------------
# Query
# -----------------------------
def answer_query(
    index_dir: str,
    query: str,
    llama_cli_path: str,
    llm_model_path: str,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    top_k: int = 8,
    cand_k: int = 30,
    mmr_lambda: float = 0.7,
    evidence_threshold: float = 0.35,
    n_ctx: int = 4096,
    max_tokens: int = 512,
    temp: float = 0.2,
    top_p: float = 0.9,
    threads: int = 6,
    gpu_layers: int = 999,
) -> Dict[str, Any]:
    t0 = time.time()
    index, meta = load_index(index_dir)
    embed_model = SentenceTransformer(embed_model_name, device="cpu")
    t1 = time.time()

    retrieved = retrieve_mmr(
        embed_model=embed_model,
        index=index,
        meta=meta,
        query=query,
        top_k=top_k,
        cand_k=cand_k,
        lambda_mult=mmr_lambda,
    )
    t2 = time.time()

    if not retrieved:
        return {
            "answer": "I don't know — I couldn't retrieve any relevant context from the indexed documents.",
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

    prompt = make_prompt(query, retrieved)
    out = run_llama_cli(
        llama_cli_path=llama_cli_path,
        model_path=llm_model_path,
        prompt=prompt,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        threads=threads,
        gpu_layers=gpu_layers,
    )
    t3 = time.time()

    return {
        "answer": out,
        "retrieved": retrieved,
        "timing": {"load_s": round(t1 - t0, 3), "retrieve_s": round(t2 - t1, 3), "generate_s": round(t3 - t2, 3), "total_s": round(t3 - t0, 3)},
        "best_score": best_score,
    }
