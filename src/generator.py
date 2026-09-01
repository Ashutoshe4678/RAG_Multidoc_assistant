import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

FALLBACK_RESPONSE = "I couldn't find enough information in the uploaded documents to answer this question."

# Active supported models on Groq API (openai/gpt-oss-20b supports large context windows without 413 errors)
GROQ_MODELS_TO_TRY = [
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "groq/compound"
]


class RAGGenerator:
    """
    Generates grounded answers using LangChain Expression Language (LCEL) chain pipeline:
    `ChatPromptTemplate | ChatGroq / ChatOpenAI | StrOutputParser()`

    LangChain Concepts Used:
    1. `ChatGroq` / `ChatOpenAI`: LangChain chat model wrapper instances.
    2. `ChatPromptTemplate`: System and human message prompt templates.
    3. `StrOutputParser`: LangChain output parser converting AIMessage into clean string output.
    4. `LCEL Pipe (|)`: Composition operator chaining prompt | llm | output_parser.
    """

    def __init__(self):
        pass

    def _get_langchain_llms(self, groq_key: str = None, openai_key: str = None):
        """Returns a list of candidate LangChain ChatGroq or ChatOpenAI model instances."""
        candidates = []

        if groq_key and groq_key.strip() and not groq_key.startswith("your_"):
            try:
                from langchain_groq import ChatGroq
                for model_name in GROQ_MODELS_TO_TRY:
                    try:
                        llm = ChatGroq(api_key=groq_key, model_name=model_name, temperature=0.0)
                        candidates.append(llm)
                    except Exception:
                        continue
            except Exception:
                pass

        if openai_key and openai_key.strip() and not openai_key.startswith("your_"):
            if openai_key.startswith("gsk_"):
                from langchain_groq import ChatGroq
                candidates.append(ChatGroq(api_key=openai_key, model_name="openai/gpt-oss-20b", temperature=0.0))
            else:
                from langchain_openai import ChatOpenAI
                candidates.append(ChatOpenAI(api_key=openai_key, model_name="gpt-4o-mini", temperature=0.0))

        return candidates

    def generate_answer(
        self,
        query: str,
        retriever_obj,
        api_key: str = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes the LCEL retrieval chain to generate a grounded answer with page citations.

        Args:
            query (str): User question string.
            retriever_obj: LangChain BaseRetriever object.
            api_key (str, optional): API key override.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: (Answer text, List of unique citations)
        """
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = api_key or os.getenv("OPENAI_API_KEY")

        llm_candidates = self._get_langchain_llms(groq_key, openai_key)
        if not llm_candidates:
            return (
                "⚠️ **API Key Missing**: Please set `GROQ_API_KEY=gsk_...` (Free at console.groq.com) or `OPENAI_API_KEY=sk-...` in your `.env` file.",
                []
            )

        # 1. Retrieve matching LangChain Document objects using BaseRetriever
        try:
            retrieved_docs = retriever_obj.invoke(query.strip())
        except Exception:
            try:
                retrieved_docs = retriever_obj.get_relevant_documents(query.strip())
            except Exception:
                retrieved_docs = []

        if not retrieved_docs:
            return FALLBACK_RESPONSE, []

        # Extract citations & format document snippets with safety limit
        citations = []
        seen_citations = set()
        formatted_snippets = ""
        max_context_chars = 4000  # Cap context length safely

        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "Unknown PDF")
            page = doc.metadata.get("page", 1)

            snippet = f"--- Snippet {idx} [Source: {source}, Page: {page}] ---\n{doc.page_content}\n\n"
            if len(formatted_snippets) + len(snippet) > max_context_chars and idx > 2:
                break

            formatted_snippets += snippet

            citation_key = (source, page)
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                citations.append({"source": source, "page": page})

        # 2. Define System & Human ChatPromptTemplate
        system_prompt = (
            "You are a strict Multi-Document Research Assistant.\n"
            "Your task is to answer the user's question based ONLY on the provided context snippets below.\n\n"
            "STRICT RULES:\n"
            "1. Rely ONLY on the clear facts contained in the provided context.\n"
            "2. Do NOT use external prior knowledge or make assumptions beyond what is explicitly stated.\n"
            "3. If the provided context does not contain enough information to answer the question, reply EXACTLY with:\n"
            f"\"{FALLBACK_RESPONSE}\"\n"
            "4. Keep the answer clear, structured, and professional.\n\n"
            "Context Snippets:\n"
            "{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        last_error = ""
        # 3. Modern LCEL Chain Execution with candidate LLM fallbacks
        for llm in llm_candidates:
            try:
                lcel_chain = prompt_template | llm | StrOutputParser()

                answer = lcel_chain.invoke({
                    "context": formatted_snippets,
                    "question": query
                }).strip()

                if FALLBACK_RESPONSE.lower() in answer.lower():
                    return FALLBACK_RESPONSE, []

                return answer, citations
            except Exception as e:
                last_error = str(e)
                continue

        return f"❌ **LangChain LCEL Error**: {last_error}", []
