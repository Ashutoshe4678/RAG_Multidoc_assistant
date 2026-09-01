from typing import List
import pymupdf  # PyMuPDF
from langchain_core.documents import Document


def extract_documents_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extracts text page-by-page from PDF file bytes and returns LangChain Document objects.

    LangChain Concept:
    LangChain represents textual data using the `Document` object, which encapsulates:
    1. `page_content` (str): The raw text snippet.
    2. `metadata` (dict): Contextual metadata key-value pairs (e.g. source filename, page number).

    Args:
        file_bytes (bytes): Raw bytes of the uploaded PDF.
        filename (str): Name of the source PDF document.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    documents = []

    if not file_bytes:
        print(f"[Warning] Empty file bytes received for: {filename}")
        return documents

    try:
        # Open PDF document using PyMuPDF stream
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()

            if text:
                # Wrap each page in a LangChain Document with metadata
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_idx + 1  # 1-indexed page number
                        }
                    )
                )

        doc.close()
    except Exception as e:
        print(f"[Error] Failed to parse PDF '{filename}': {str(e)}")

    return documents
