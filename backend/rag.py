import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from backend.retrieval.dense import (
    get_dense_retriever,
    dense_similarity_search
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv(
        "GOOGLE_MODEL",
        "gemini-3.1-flash-lite"
    ),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise document assistant.
Answer based ONLY on the provided context.
If the answer is not in the context, say so clearly.
Never hallucinate or make up information.

Context:
{context}

Question: {question}

Answer:"""
)


def build_chain(
    user_id: str,
    doc_ids: list[str] | None = None
):

    retriever = get_dense_retriever(
        user_id,
        doc_ids=doc_ids,
        k=5
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        k=6
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT},
        return_source_documents=True,
        verbose=False
    )

    return chain


def ask(chain, question: str) -> dict:

    result = chain({"question": question})

    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result.get("source_documents", [])
    })

    return {
        "answer": result["answer"],
        "sources": sources
    }


def compare_documents(
    user_id: str,
    doc_ids: list[str],
    question: str
) -> dict:

    results = {}

    for doc_id in doc_ids:
        chain = build_chain(user_id, [doc_id])

        results[doc_id] = ask(
            chain,
            question
        )["answer"]

    return {
        "question": question,
        "by_document": results
    }


def cross_search(
    user_id: str,
    query: str,
    k: int = 8
) -> list:

    results = dense_similarity_search(
        user_id,
        query,
        k=k
    )

    return [{
        "content": doc.page_content[:400],
        "source": doc.metadata.get("source"),
        "doc_id": doc.metadata.get("doc_id"),
        "score": round(float(score), 3)
    } for doc, score in results]