from backend.ingest import get_vectorstore


def get_dense_retriever(
    user_id: str,
    doc_ids: list[str] | None = None,
    k: int = 5
):
    vectorstore = get_vectorstore(user_id)

    search_kwargs = {"k": k}

    if doc_ids:
        search_kwargs["filter"] = {
            "doc_id": {
                "$in": doc_ids
            }
        }

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )


def dense_similarity_search(
    user_id: str,
    query: str,
    k: int = 8
):
    vectorstore = get_vectorstore(user_id)

    return vectorstore.similarity_search_with_score(
        query,
        k=k
    )