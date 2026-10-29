"""Streamlit interface for local PDF question answering."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import streamlit as st
from chromadb.errors import ChromaError

from rag_pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TOP_K,
    DocumentProcessingError,
    RAGServiceError,
    create_vector_collection,
    extractive_answer,
    generate_local_answer,
    index_document,
    list_indexed_sources,
    retrieve_chunks,
)


st.set_page_config(
    page_title="RAG PDF Question Answering",
    page_icon=":material/description:",
    layout="wide",
)


@st.cache_resource
def get_collection() -> Any:
    return create_vector_collection()


st.title("RAG PDF Question Answering")
st.caption(
    "Upload documents, retrieve the most relevant passages, and ask grounded questions. "
    "Your Chroma index stays on this computer."
)

try:
    collection = get_collection()
except (ChromaError, OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
    st.error(f"Could not open the local Chroma index: {exc}")
    st.stop()

index_result = st.session_state.pop("index_result", None)
if index_result:
    if index_result["level"] == "warning":
        st.warning(index_result["message"])
    else:
        st.success(index_result["message"])

with st.sidebar:
    st.header("Documents")
    chunk_size = st.number_input(
        "Words per chunk",
        min_value=20,
        max_value=1000,
        value=DEFAULT_CHUNK_SIZE,
        step=20,
        help="Smaller chunks are more precise; larger chunks preserve more context.",
    )
    chunk_overlap = st.number_input(
        "Overlapping words",
        min_value=0,
        max_value=int(chunk_size) - 1,
        value=min(DEFAULT_CHUNK_OVERLAP, int(chunk_size) - 1),
        step=5,
        help="A small overlap keeps context that crosses a chunk boundary.",
    )
    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    st.subheader("Answer mode")
    answer_mode = st.selectbox(
        "Choose how answers are written",
        ["Extractive (no chat model required)", "Ollama local LLM"],
    )
    ollama_url = st.text_input(
        "Ollama URL",
        value=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        disabled=answer_mode != "Ollama local LLM",
    )
    ollama_model = st.text_input(
        "Ollama model",
        value=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        disabled=answer_mode != "Ollama local LLM",
        help="Example: qwen2.5:3b. Install it with `ollama pull qwen2.5:3b`.",
    )
    st.caption(
        "Extractive mode needs no chat model. Ollama mode runs generation locally."
    )

    st.divider()
    st.subheader("Indexed documents")
    try:
        indexed_sources = list_indexed_sources(collection)
    except (ChromaError, sqlite3.Error) as exc:
        st.error(f"Could not read the local vector index: {exc}")
        indexed_sources = []

    if indexed_sources:
        st.metric(
            "Indexed passages",
            sum(chunk_count for _, chunk_count in indexed_sources),
        )
        for source, chunk_count in indexed_sources:
            st.text(f"{source} · {chunk_count} passages")
    else:
        st.caption("No documents indexed yet.")


st.subheader("Add documents")
st.caption(
    "The first index may download Chroma's approximately 79 MB local embedding model."
)
uploaded_files = st.file_uploader(
    "Choose PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help=(
        "PDFs must contain selectable text. Scanned PDFs need OCR first. "
        "Maximum file size: 25 MB."
    ),
)

upload_col, sample_col = st.columns(2, vertical_alignment="bottom")
with upload_col:
    if st.button(
        "Index selected documents",
        type="primary",
        disabled=not uploaded_files,
        width="stretch",
    ):
        indexed_count = 0
        failures: list[str] = []
        with st.status("Reading and indexing documents", expanded=True) as status:
            for uploaded_file in uploaded_files:
                try:
                    chunk_count = index_document(
                        collection,
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                except (
                    DocumentProcessingError,
                    RAGServiceError,
                    ChromaError,
                    OSError,
                    sqlite3.Error,
                    ValueError,
                ) as exc:
                    failures.append(f"{uploaded_file.name}: {exc}")
                    st.write(f"Could not index {uploaded_file.name}: {exc}")
                else:
                    indexed_count += chunk_count
                    st.write(
                        f"Indexed {uploaded_file.name} into {chunk_count} passages."
                    )
            if failures:
                status.update(label="Indexing finished with errors", state="error")
            elif indexed_count:
                status.update(label="Documents indexed", state="complete")
            else:
                status.update(label="No documents were indexed", state="error")
        if indexed_count:
            if failures:
                st.session_state["index_result"] = {
                    "level": "warning",
                    "message": (
                        f"Indexed {indexed_count} passages; "
                        f"{len(failures)} file(s) could not be indexed:\n"
                        + "\n".join(failures)
                    ),
                }
            else:
                st.session_state["index_result"] = {
                    "level": "success",
                    "message": f"Indexed {indexed_count} passages.",
                }
            st.rerun()

with sample_col:
    sample_path = Path(__file__).parent / "documents" / "leave_policy.txt"
    if st.button(
        "Index the sample leave policy",
        disabled=not sample_path.is_file(),
        width="stretch",
    ):
        try:
            sample_bytes = sample_path.read_bytes()
            chunk_count = index_document(
                collection,
                sample_path.name,
                sample_bytes,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )
        except (
            DocumentProcessingError,
            RAGServiceError,
            ChromaError,
            OSError,
            sqlite3.Error,
            ValueError,
        ) as exc:
            st.error(f"Could not index the sample policy: {exc}")
        else:
            st.session_state["index_result"] = {
                "level": "success",
                "message": (
                    f"Indexed the sample leave policy into {chunk_count} passages."
                ),
            }
            st.rerun()

st.divider()
st.subheader("Ask your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for index, source in enumerate(message["sources"], start=1):
                    location = source["source"]
                    if source["page_number"] is not None:
                        location += f" · page {source['page_number']}"
                    st.markdown(f"**[{index}] {location}**")
                    st.caption(source["text"])
                    if source["distance"] is not None:
                        st.caption(f"Vector distance: {source['distance']:.3f}")

if not indexed_sources:
    st.info("Index a document above to start asking questions.")

question = st.chat_input("Ask a question about your indexed documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            matches = retrieve_chunks(collection, question, top_k=int(top_k))
            if not matches:
                answer = (
                    "मुझे indexed documents में इस प्रश्न से संबंधित कोई passage नहीं मिला।"
                )
            elif answer_mode == "Ollama local LLM":
                answer = generate_local_answer(
                    question,
                    matches,
                    model=ollama_model,
                    base_url=ollama_url,
                )
            else:
                answer = extractive_answer(matches)
        except (ChromaError, RAGServiceError, ValueError) as exc:
            st.error(f"Question answering failed: {exc}")
        else:
            st.markdown(answer)
            source_records = [
                {
                    "source": match.source,
                    "page_number": match.page_number,
                    "distance": match.distance,
                    "text": match.text,
                }
                for match in matches
            ]
            with st.expander(f"Sources ({len(source_records)})"):
                for index, source in enumerate(source_records, start=1):
                    location = source["source"]
                    if source["page_number"] is not None:
                        location += f" · page {source['page_number']}"
                    st.markdown(f"**[{index}] {location}**")
                    st.caption(source["text"])
                    if source["distance"] is not None:
                        st.caption(f"Vector distance: {source['distance']:.3f}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_records,
                }
            )

if st.session_state.messages and st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
