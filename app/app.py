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
st.title("Orin Nano Offline RAG (llama-server + FAISS + incremental indexing)")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE, "data", "raw")
INDEX_DIR = os.path.join(BASE, "data", "index")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Server URL (local only)
LLAMA_SERVER_URL = "http://127.0.0.1:8080"

st.sidebar.header("Paths")
st.sidebar.code(
    f"RAW_DIR: {RAW_DIR}\n"
    f"INDEX_DIR: {INDEX_DIR}\n"
    f"LLAMA_SERVER_URL: {LLAMA_SERVER_URL}\n"
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

st.divider()

index_ok = (
    os.path.exists(os.path.join(INDEX_DIR, INDEX_FILE))
    and os.path.exists(os.path.join(INDEX_DIR, META_FILE))
)

st.subheader("Ask")

q = st.text_input("Question about your documents:")

left, right = st.columns(2)
with left:
    top_k = st.slider("Top-k chunks", 3, 12, 6)
with right:
    evidence_threshold = st.slider("Evidence threshold", 0.10, 0.80, 0.35, step=0.01)

adv = st.expander("Advanced generation")
with adv:
    max_tokens = st.slider("Max new tokens", 32, 512, 256, step=32)
    temp = st.slider("Temperature", 0.0, 1.5, 0.2, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)

ask = st.button("Answer")

if ask:
    if not index_ok:
        st.error("No index found yet. Upload PDFs, then click 'Build / Rebuild' first.")
    elif not q.strip():
        st.error("Type a question.")
    else:
        try:
            with st.spinner("Retrieving + generating (offline via llama-server)..."):
                result = answer_query(
                    index_dir=INDEX_DIR,
                    query=q.strip(),
                    llama_server_url=LLAMA_SERVER_URL,
                    top_k=top_k,
                    evidence_threshold=evidence_threshold,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
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
                    txt = r["text"]
                    st.write(txt[:2000] + ("..." if len(txt) > 2000 else ""))
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.exception(e)

st.divider()
st.caption("Run llama-server separately so the model stays loaded (faster + no Streamlit hangs).")
