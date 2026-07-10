import streamlit as st

from src.embedders.local_embedder import LocalEmbedder
from src.vectorstores.chroma_vectorstore import ChromaVectorStore
from src.llms.groq_llm import GroqLLM
from src.rerankers.flashrank_reranker import FlashRankReranker
from src.pipeline import RAGPipeline


@st.cache_resource
def load_pipeline():
    """
    Cached so the pipeline (and its ingested documents) only load once per
    server session, not on every question asked.
    """
    pipeline = RAGPipeline(
        embedder=LocalEmbedder(),
        vectorstore=ChromaVectorStore(),
        llm=GroqLLM(),
        reranker=FlashRankReranker(),
    )
    chunk_count = pipeline.ingest_directory("data/sample_docs/true_data")
    return pipeline, chunk_count


st.set_page_config(page_title="RAG Pipeline Demo", layout="wide")
st.title("RAG Pipeline — Ask a Question")

pipeline, chunk_count = load_pipeline()
st.caption(f"Knowledge base loaded: {chunk_count} chunks ingested.")

# Chat history persists across reruns via session_state — without this,
# Streamlit would forget every previous question/answer on each new input.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render the full conversation so far, on every rerun. This is what
# makes old Q&As stay visible instead of being replaced by the new one.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander(f"Retrieved chunks ({len(msg['chunks'])})"):
                for i, chunk in enumerate(msg["chunks"], start=1):
                    st.text(f"Chunk {i}\n{chunk}")

# st.chat_input auto-clears itself after submit — no manual clearing needed,
# and it stays pinned to the bottom of the page like ChatGPT/Claude's input.
question = st.chat_input("Ask a question about the ingested documents:")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            answer, retrieved_chunks = pipeline.query(question, return_context=True)
        st.write(answer)
        with st.expander(f"Retrieved chunks ({len(retrieved_chunks)})"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.text(f"Chunk {i}\n{chunk}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": retrieved_chunks,
    })