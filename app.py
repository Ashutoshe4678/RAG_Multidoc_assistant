import os
import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import extract_documents_from_pdf
from src.chunker import create_chunks
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.generator import RAGGenerator

# Load environment variables from .env file
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multi-Document Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []


# Lazy singletons for backend components
@st.cache_resource
def get_vector_store():
    embedding_manager = EmbeddingManager()
    return VectorStoreManager(embedding_manager=embedding_manager)


vector_store = get_vector_store()
retriever_manager = RAGRetriever(vector_store)
generator = RAGGenerator()

# Sidebar UI
with st.sidebar:
    st.title("📂 Document Manager")
    st.markdown("Upload multiple PDF documents and index them for intelligent search.")

    uploaded_files = st.file_uploader(
        "Select PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚡ Process Documents", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear Documents", use_container_width=True)

    st.divider()

    # Configurable retrieval depth parameter
    top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=5)

    # Handle Clear Documents Action
    if clear_btn:
        vector_store.clear_store()
        st.session_state.uploaded_filenames = []
        st.session_state.messages = []
        st.success("Vector store cleared successfully!")

    # Handle Process Documents Action
    if process_btn:
        if not uploaded_files:
            st.warning("Please select at least one PDF file first.")
        else:
            all_documents = []
            processed_names = []

            with st.spinner("Parsing PDFs and generating embeddings..."):
                for pdf_file in uploaded_files:
                    file_bytes = pdf_file.getvalue()
                    filename = pdf_file.name

                    docs = extract_documents_from_pdf(file_bytes, filename)
                    if docs:
                        all_documents.extend(docs)
                        processed_names.append(filename)
                    else:
                        st.warning(f"Could not extract text from '{filename}' (it may be empty or image-only).")

                if all_documents:
                    chunks = create_chunks(all_documents)
                    added_count = vector_store.add_documents(chunks)

                    # Update session state file tracking
                    for fname in processed_names:
                        if fname not in st.session_state.uploaded_filenames:
                            st.session_state.uploaded_filenames.append(fname)

                    st.success(f"Successfully indexed {len(processed_names)} PDFs ({added_count} new chunks)!")
                else:
                    st.error("No valid text extracted from uploaded PDFs.")

    # Collection statistics
    chunk_count = vector_store.get_count()
    st.metric("Total Indexed Chunks", chunk_count)

    if st.session_state.uploaded_filenames:
        st.subheader("Uploaded Documents:")
        for name in st.session_state.uploaded_filenames:
            st.text(f"📄 {name}")

# Main Application Layout
st.title("Multi-Document Research Assistant 🤖")
st.caption("Ask questions about your uploaded documents. Answers are derived strictly from retrieved text chunks with exact page citations.")

# Display Chat Transcript
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            st.caption("📌 **Sources Used:**")
            for cit in msg["citations"]:
                st.markdown(f"- Source: `{cit['source']}` — Page `{cit['page']}`")

# Chat Prompt Input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    if vector_store.get_count() == 0:
        st.warning("Please upload and process PDF documents before asking questions.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve & generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating response..."):
                langchain_retriever = retriever_manager.get_langchain_retriever(top_k=top_k)
                answer, citations = generator.generate_answer(query=prompt, retriever_obj=langchain_retriever)

                st.markdown(answer)
                if citations:
                    st.caption("📌 **Sources Used:**")
                    for cit in citations:
                        st.markdown(f"- Source: `{cit['source']}` — Page `{cit['page']}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })
