import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedding_model():
    """
    Load the embedding model once and reuse it.
    """

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def generate_embeddings(texts):
    """
    Convert a list of text chunks into embeddings.
    """

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    return embeddings.tolist()