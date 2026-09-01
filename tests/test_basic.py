import os
import shutil
import pytest
import pymupdf  # PyMuPDF
from langchain_core.documents import Document

from src.pdf_processor import extract_documents_from_pdf
from src.chunker import create_chunks
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.generator import RAGGenerator, FALLBACK_RESPONSE

TEST_DB_PATH = "data/chroma_test"


@pytest.fixture(autouse=True)
def cleanup_test_db():
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
    yield
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


def create_sample_pdf_bytes(text_content: str) -> bytes:
    """Helper to generate in-memory sample PDF bytes using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text_content)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_pdf_extraction_to_langchain_documents():
    """Verify PyMuPDF text extraction into LangChain Document objects."""
    sample_text = "Supervised learning uses labeled datasets to train machine learning algorithms."
    pdf_bytes = create_sample_pdf_bytes(sample_text)

    docs = extract_documents_from_pdf(pdf_bytes, "machine_learning.pdf")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["source"] == "machine_learning.pdf"
    assert docs[0].metadata["page"] == 1
    assert "Supervised learning" in docs[0].page_content


def test_chunking_metadata_retention():
    """Verify RecursiveCharacterTextSplitter preserves source document and page metadata."""
    documents = [
        Document(
            page_content="Supervised learning trains models on labeled data. Unsupervised learning finds patterns in unlabeled data. " * 5,
            metadata={"source": "ml_guide.pdf", "page": 3}
        )
    ]

    chunks = create_chunks(documents, chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert chunk.metadata["source"] == "ml_guide.pdf"
        assert chunk.metadata["page"] == 3
        assert "chunk_index" in chunk.metadata


def test_embeddings_generation():
    """Verify LangChain embeddings vector output generation."""
    em = EmbeddingManager()
    vectors = em.embeddings.embed_documents(["Hello world", "Artificial intelligence RAG system"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # MiniLM-L6-v2 vector dimension


def test_vector_store_end_to_end():
    """Verify LangChain Chroma document indexing, deduplication, and similarity retrieval."""
    em = EmbeddingManager()
    vs = VectorStoreManager(persist_directory=TEST_DB_PATH, collection_name="test_coll", embedding_manager=em)
    vs.clear_store()

    documents = [
        Document(page_content="Deep learning relies heavily on multi-layer neural networks.", metadata={"source": "deep_learning.pdf", "page": 1}),
        Document(page_content="Natural Language Processing (NLP) enables text analysis and summarization.", metadata={"source": "nlp.pdf", "page": 2})
    ]
    chunks = create_chunks(documents)
    added_count = vs.add_documents(chunks)
    assert added_count == len(chunks)

    # Test deduplication
    readded_count = vs.add_documents(chunks)
    assert readded_count == 0

    # Test retriever lookup via LangChain BaseRetriever
    retriever_manager = RAGRetriever(vs)
    retrieved_docs = retriever_manager.retrieve_documents("neural networks", top_k=2)
    assert len(retrieved_docs) > 0
    assert retrieved_docs[0].metadata["source"] == "deep_learning.pdf"
    assert retrieved_docs[0].metadata["page"] == 1


def test_generator_fallback():
    """Verify strictly grounded fallback response when context is empty."""
    em = EmbeddingManager()
    vs = VectorStoreManager(persist_directory=TEST_DB_PATH, collection_name="test_coll", embedding_manager=em)
    retriever_manager = RAGRetriever(vs)
    langchain_retriever = retriever_manager.get_langchain_retriever(top_k=2)

    generator = RAGGenerator()
    answer, citations = generator.generate_answer(query="What is quantum computing?", retriever_obj=langchain_retriever)
    assert answer == FALLBACK_RESPONSE
    assert citations == []
