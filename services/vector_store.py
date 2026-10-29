import os

from dotenv import load_dotenv
from supabase import create_client


load_dotenv()


def get_supabase_client():
    """
    Create a connection to Supabase.
    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials are missing. "
            "Check your .env file."
        )

    return create_client(
        supabase_url,
        supabase_key,
    )


def store_document_chunks(chunks, embeddings, filename, document_id):
    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings must match."
        )

    supabase = get_supabase_client()
    records = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings), start=1
    ):
        records.append({
            "content": chunk,
            "embedding": embedding,
            "metadata": {
                "source": filename,
                "document_id": document_id,
                "chunk_index": index,
            },
        })

    batch_size = 50

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        supabase.table("document_chunks").insert(batch).execute()

    return len(records)


def search_document_chunks(
    query_embedding,
    match_count=5,
    document_id=None,
):
    supabase = get_supabase_client()

    response = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_document_id": document_id,
        },
    ).execute()

    return response.data
