# Multi-Document Research Assistant 🤖

A simple, clean, and complete **Retrieval-Augmented Generation (RAG)** application built in Python that enables users to upload multiple PDF documents and ask questions about their content with grounded answers and page-level citations.

---

## 1. Project Overview

The **Multi-Document Research Assistant** lets users upload multiple PDF files (e.g., textbook chapters, research papers, resumes, reports) and query them through a natural language chat interface. Rather than relying on a general-purpose Large Language Model's parametric memory (which can hallucinate), this system extracts text from the uploaded PDFs, indexes it in a local vector database, retrieves relevant passages, and forces the LLM to generate answers strictly grounded in the retrieved facts.

---

## 2. What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI architecture pattern designed to combine the strengths of information retrieval systems with generative LLMs:

- **Retrieval**: Search an external knowledge base (e.g., custom PDFs, corporate docs) to locate relevant information for a user query.
- **Augmented Generation**: Supply those retrieved snippets as context to an LLM so it generates accurate, verifiable answers grounded in facts.

---

## 3. How the RAG Pipeline Works

```
📄 Upload PDFs ──> PyMuPDF Stream ──> Document Parsing ──> Recursive Text Chunking
                                                                    │
                                                                    ▼
User Question ──> Similarity Retriever ◄── Persistent Chroma Vector Store ┘
                       │
                       ▼
          LLM Prompting (Groq Llama 3.3 / OpenAI)
                       │
                       ▼
         Answer + Source & Page Citations
```

1. **Document Loading**: PyMuPDF extracts text per page into document objects with metadata (`source`, `page`).
2. **Chunking**: `RecursiveCharacterTextSplitter` breaks documents into ~1000 character snippets with 200 character overlap, preserving metadata.
3. **Embeddings**: `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) generates 384-dimensional vector embeddings.
4. **Vector Storage**: `Chroma` stores vectors, text, and metadata locally (`data/chroma`). Deterministic IDs prevent duplicate indexing.
5. **Retrieval**: Cosine similarity search retrieves top-K relevant chunks for the user question.
6. **Generation**: Groq API (`Llama 3.3` / `openai/gpt-oss-20b`) or OpenAI API generates answers strictly grounded in the retrieved context snippets.
7. **Citations**: Page and document citations are extracted from retrieved document metadata and rendered in the UI.

---

## 4. Technology Stack

- **Python 3.11+**
- **Streamlit**: Interactive chat interface.
- **PyMuPDF (`pymupdf`)**: PDF text and page extraction.
- **LangChain Ecosystem**: `langchain`, `langchain-community`, `langchain-core`, `langchain-text-splitters`, `langchain-chroma`, `langchain-groq`, `langchain-openai`
- **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Local semantic text embeddings.
- **ChromaDB**: Local persistent vector database.
- **Groq API**: High-speed free LLM text generation (`Llama 3.3` / `openai/gpt-oss-20b`).
- **OpenAI API**: Optional fallback LLM.
- **python-dotenv**: Environment variable management.
- **pytest**: Automated testing framework.

---

## 5. Project Structure

```
RAG_multidoc_assistant/
│
├── app.py                  # Streamlit UI & application controller
├── requirements.txt        # Package dependencies
├── .env                    # Environment variables (Groq / OpenAI API keys)
├── .env.example            # Environment template file
├── .gitignore              # Files excluded from git
├── README.md               # Documentation & setup guide
│
├── data/
│   └── chroma/             # Local persistent ChromaDB database folder
│
├── src/
│   ├── __init__.py         # Package initialization
│   ├── pdf_processor.py    # PyMuPDF extraction returning document objects
│   ├── chunker.py          # Recursive text splitter with metadata preservation
│   ├── embeddings.py       # Embeddings manager
│   ├── vector_store.py     # Chroma vector collection manager & deduplication
│   ├── retriever.py        # Vector store retriever builder
│   └── generator.py        # RAG answer generation & citations
│
└── tests/
    └── test_basic.py       # Pytest unit tests for RAG pipeline
```

---

## 6. Installation

1. **Navigate to the Project Directory**:
   ```bash
   cd C:\Users\ashuk\Downloads\RAG_multidoc_assistant
   ```

2. **Create and Activate a Virtual Environment** *(recommended)*:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. Environment Variable Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and add your **Groq API Key** (100% FREE from [console.groq.com/keys](https://console.groq.com/keys)):
   ```ini
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

---

## 8. How to Run the Application

1. **Start the Streamlit Server**:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser at `http://localhost:8501`.
3. Upload one or more PDF files via the sidebar.
4. Click **⚡ Process Documents**.
5. Type your question in the chat input!

---

## 9. Run Automated Tests

To run the pipeline test suite:
```bash
pytest tests/test_basic.py -v
```

---

## 10. Example Questions

Given sample AI/ML documents (`machine_learning.pdf`, `deep_learning.pdf`, `nlp.pdf`):

- *"What are the main differences between supervised and unsupervised learning?"*
- *"What is a neural network and how is it used in deep learning?"*
- *"What techniques are used for Natural Language Processing?"*
