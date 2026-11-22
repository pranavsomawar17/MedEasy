"""
LangChain + FAISS Retrieval-Augmented Generation (RAG) Q&A module.
Answers follow-up questions strictly grounded in the loaded health article.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


QA_SYSTEM_PROMPT = """You are a helpful medical article Q&A assistant.
Answer the user's question using ONLY the provided excerpts from the health article.

Rules:
1. Respond in the requested Target Language in clear, simple wording.
2. Keep all numbers, dosages, thresholds, and warnings exact.
3. If the answer is NOT in the provided article excerpts, say clearly that the article does not mention it—never guess or invent medical facts.
4. Keep the answer concise and easy to read."""

QA_USER_PROMPT = """Target Language: {language}

Article Excerpts:
{context}

Question: {question}"""


def load_embeddings() -> HuggingFaceEmbeddings:
    """Load free, local HuggingFace sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss_index(article_text: str, source_title: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Split article text into overlapping chunks and index them in FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(article_text)
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": source_title, "chunk_id": idx + 1},
        )
        for idx, chunk in enumerate(chunks)
    ]
    return FAISS.from_documents(docs, embeddings)


def answer_question(
    vectorstore: FAISS,
    question: str,
    api_key: str,
    model_name: str,
    language: str,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    """Retrieve top-k chunks from FAISS and generate a grounded answer via Groq."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(
        f"[Excerpt {doc.metadata.get('chunk_id', i + 1)}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            ("human", QA_USER_PROMPT),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "language": language,
            "context": context,
            "question": question,
        }
    )

    return str(response.content).strip(), docs
