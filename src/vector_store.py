import os
from typing import List, Optional
from langchain_core.documents import Document
from src.embeddings import EmbeddingManager


class VectorStoreManager:
    """
    Manages persistent local vector database storage using LangChain's Chroma wrapper.

    LangChain Concept:
    LangChain `Chroma` wraps the underlying vector DB client.
    It takes an `embedding_function` (LangChain Embeddings) and manages document indexing,
    similarity queries, and persistence to disk (`persist_directory`).
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "pdf_rag_chain_collection",
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()

        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore = self._init_vectorstore()

    def _init_vectorstore(self):
        embeddings = self.embedding_manager.embeddings
        try:
            from langchain_chroma import Chroma
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory
            )
        except Exception:
            from langchain_community.vectorstores import Chroma
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory
            )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Adds chunk Documents to LangChain Chroma vector store with deduplication.

        Args:
            documents (List[Document]): List of chunk Documents.

        Returns:
            int: Count of new Document chunks added.
        """
        if not documents:
            return 0

        # Retrieve existing IDs from collection to prevent duplicate insertion
        try:
            existing_records = self.vectorstore._collection.get()
            existing_ids = set(existing_records["ids"]) if existing_records and "ids" in existing_records else set()
        except Exception:
            existing_ids = set()

        new_docs = []
        new_ids = []

        for doc in documents:
            doc_id = doc.metadata.get("doc_id", f"{doc.metadata.get('source')}_p{doc.metadata.get('page')}")
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)
                existing_ids.add(doc_id)

        if not new_docs:
            return 0

        # LangChain add_documents method inserts Documents and ids
        self.vectorstore.add_documents(documents=new_docs, ids=new_ids)
        return len(new_docs)

    def clear_store(self):
        """Clears all stored document vectors from ChromaDB collection."""
        try:
            self.vectorstore._client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.vectorstore = self._init_vectorstore()

    def get_count(self) -> int:
        """Returns total count of indexed document chunks."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
