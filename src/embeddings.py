from typing import List
from sentence_transformers import SentenceTransformer


class DirectSentenceTransformerEmbeddings:
    """Fallback embedding class implementing LangChain Embeddings interface."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()


class EmbeddingManager:
    """
    LangChain HuggingFace Embeddings Manager.

    LangChain Concept:
    LangChain unifies embedding models under the `Embeddings` base class with two main methods:
    1. `embed_documents(List[str])`: Generates vectors for a list of document chunks.
    2. `embed_query(str)`: Generates a vector for a single query.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings_instance = None

    @property
    def embeddings(self):
        """Returns a LangChain-compatible Embeddings object."""
        if self._embeddings_instance is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings_instance = HuggingFaceEmbeddings(model_name=self.model_name)
            except Exception:
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self._embeddings_instance = HuggingFaceEmbeddings(model_name=self.model_name)
                except Exception:
                    self._embeddings_instance = DirectSentenceTransformerEmbeddings(self.model_name)
        return self._embeddings_instance
