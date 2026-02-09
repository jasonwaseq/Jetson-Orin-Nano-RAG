import os
import shutil
import streamlit as st

from rag import (
    ingest_pdfs,
    answer_query,
    load_registry,
    delete_doc_and_rebuild,
    INDEX_FILE,
    META_FILE,
)

st.set_page_config(page_title="Orin Nano Offline RAG", layout="wide")
st.title("Orin Nano Offline RAG (Offline Qwen2.5 + FAISS + Incremental Indexing)")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE, "data", "raw")
INDEX_DIR = os.path.join(BASE, "data", "index")

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
LLM_MODEL = os.path.join(BASE, "models", "llm", "qwen2.5-7b-instruct-q4_k_m.gguf")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------- Sidebar: status + uploads ----------
st.sidebar.header("Paths")
st.sidebar.code(
    f"RAW_DIR: {RAW_DIR}\n"
    f"INDEX_DIR: {INDEX_DIR}\n"
    f"llama-cli: {LLAMA_CLI}\n"
    f"GGUF: {LLM_MODEL}\n"
)

st.sidebar.divider()
st.sidebar.header("Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        out_path = os.path.join(RAW_DIR, f.name)
        with open(out_path, "wb") as w:
            w.write(f.read())
    st.sidebar.success(f"Saved {len(uploaded)} PDF(s) to data/raw/")

st.sidebar.divider()
st.sidebar.header("Library")

try:
    reg = load_registry(INDEX_DIR)
    docs = reg.get("docs", {})
except Exception:
    docs = {}

if docs:
    st.sidebar.write(f"Indexed docs: **{len(docs)}**")
    doc_list = sorted(docs.keys())
    selected_doc = st.sidebar.selectbox("Select doc", ["(none)"] + doc_list)
    if selected_doc != "(none)":
        st.sidebar.json(docs[selected_doc])
        if st.sidebar.button("Delete selected doc (rebuild index)"):
            with st.spinner("Deleting doc and rebuilding index..."):
                res = delete_doc_and_rebuild(INDEX_DIR, selected_doc)
            st.sidebar.success(str(res))
            st.rerun()
else:
    st.sidebar.write("No indexed docs yet.")

st.sidebar.divider()
st.sidebar.header("Index controls")

force_rebuild = st.sidebar.checkbox("Force full rebuild", value=False)

colA, colB = st.sidebar.columns(2)
build = colA.button("Build / Rebuild")
clear = colB.button("Clear ALL")

if clear:
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    st.success("Cleared PDFs + index + registry.")
    st.stop()

if build:
    try:
        with st.spinner("Indexing (incremental by default)..."):
            stats = ingest_pdfs(RAW_DIR, INDEX_DIR, force_rebuild=force_rebuild)
        st.success("Index built successfully.")
        st.subheader("Ingestion stats")
        st.json(stats)
    except Exception as e:
        st.error(f"Ingest failed: {e}")
        st.exception(e)
    st.rerun()

# ---------- Guard ----------
index_ok = (
    os.path.exists(os.path.join(INDEX_DIR, INDEX_FILE))
    and os.path.exists(os.path.join(INDEX_DIR, META_FILE))
)

st.divider()

# ---------- Main: QA ----------
st.subheader("Ask")

q = st.text_input("Question about your documents:")
left, right = st.columns(2)
with left:
    top_k = st.slider("Top-k final chunks", 4, 16, 8)
with right:
    n_ctx = st.selectbox("Context size (n_ctx)", [2048, 3072, 4096, 8192], index=2)

adv = st.expander("Advanced retrieval + generation")
with adv:
    cand_k = st.slider("Candidate chunks (pre-MMR)", 10, 80, 30, step=5)
    mmr_lambda = st.slider("MMR lambda (higher = more relevant, lower = more diverse)", 0.3, 0.9, 0.7, step=0.05)
    evidence_threshold = st.slider("Evidence threshold (best similarity)", 0.10, 0.80, 0.35, step=0.01)
    max_tokens = st.slider("Max new tokens", 64, 1024, 384, step=64)
    temp = st.slider("Temperature", 0.0, 1.5, 0.2, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    threads = st.slider("CPU threads", 1, 16, 6)
    gpu_layers = st.slider("GPU layers (-ngl)", 0, 999, 999)

ask = st.button("Answer")

if ask:
    if not index_ok:
        st.error("No index found yet. Upload PDFs, then click 'Build / Rebuild' first.")
    elif not q.strip():
        st.error("Type a question.")
    else:
        try:
            with st.spinner("Retrieving + generating (offline)..."):
                result = answer_query(
                    index_dir=INDEX_DIR,
                    query=q.strip(),
                    llama_cli_path=LLAMA_CLI,
                    llm_model_path=LLM_MODEL,
                    top_k=top_k,
                    cand_k=cand_k,
                    mmr_lambda=mmr_lambda,
                    evidence_threshold=evidence_threshold,
                    n_ctx=n_ctx,
                    max_tokens=max_tokens,
                    temp=temp,
                    top_p=top_p,
                    threads=threads,
                    gpu_layers=gpu_layers,
                )

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Timing")
            st.json(result["timing"])

            if "note" in result:
                st.info(result["note"])

            st.subheader("Sources Used")
            for i, r in enumerate(result["retrieved"], 1):
                score = r.get("score", 0.0)
                with st.expander(f"{i}) {r['doc_id']} p.{r['page']} (score={score:.3f})"):
                    st.write(r["text"])
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.exception(e)

st.divider()
st.caption("Incremental indexing is default. Use 'Force full rebuild' only if needed.")
