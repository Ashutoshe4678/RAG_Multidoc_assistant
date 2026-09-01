from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits LangChain Document objects into smaller overlapping Document chunks using RecursiveCharacterTextSplitter.

    LangChain Concept:
    `RecursiveCharacterTextSplitter` recursively splits text on paragraph ('\\n\\n'), line ('\\n'),
    space (' '), and character ('') boundaries to preserve semantic coherence.
    LangChain automatically propagates the parent Document metadata to every child chunk.

    Args:
        documents (List[Document]): List of input page Documents.
        chunk_size (int): Target character length per chunk.
        chunk_overlap (int): Overlap character length between adjacent chunks.

    Returns:
        List[Document]: List of chunked Document objects.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    # LangChain split_documents method automatically preserves and propagates metadata
    chunks = splitter.split_documents(documents)

    # Add chunk_index and deterministic chunk ID for deduplication
    page_chunk_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 1)

        key = (source, page)
        idx = page_chunk_counts.get(key, 0)
        page_chunk_counts[key] = idx + 1

        chunk.metadata["chunk_index"] = idx
        chunk.metadata["doc_id"] = f"{source}_p{page}_c{idx}"

    return chunks
