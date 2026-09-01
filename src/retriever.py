from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.vector_store import VectorStoreManager


class RAGRetriever:
    """
    LangChain VectorStoreRetriever module.

    LangChain Concept:
    LangChain vector stores expose a standardized `as_retriever()` method that returns
    a `BaseRetriever` runnable object. It accepts `search_kwargs={"k": top_k}` to control
    how many relevant Document snippets are retrieved for a query.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def get_langchain_retriever(self, top_k: int = 5) -> BaseRetriever:
        """
        Returns a native LangChain BaseRetriever configured with top_k depth.

        Args:
            top_k (int): Number of top matching chunks to retrieve.

        Returns:
            BaseRetriever: Configured LangChain retriever instance.
        """
        # Ensure k is bounded by available document count
        count = self.vector_store.get_count()
        k = min(top_k, max(1, count))

        return self.vector_store.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Direct document retrieval helper method.

        Args:
            query (str): User question.
            top_k (int): Number of top matching document chunks to retrieve.

        Returns:
            List[Document]: Matching LangChain Document objects.
        """
        if not query or not query.strip() or self.vector_store.get_count() == 0:
            return []

        retriever = self.get_langchain_retriever(top_k=top_k)

        # In modern LangChain, invoke() retrieves relevant documents
        try:
            return retriever.invoke(query.strip())
        except Exception:
            return retriever.get_relevant_documents(query.strip())
