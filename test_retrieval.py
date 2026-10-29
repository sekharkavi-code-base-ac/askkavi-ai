from services.embeddings import generate_embeddings
from services.vector_store import search_document_chunks

# Ask a question related to your uploaded PDF.
question = "What is this document about?"

# Convert the question into a vector.
query_embedding = generate_embeddings([question])[0]

# Search Supabase for the five most relevant chunks.
results = search_document_chunks(query_embedding, match_count=5)

# Display the results.
if results:
    print("\nRelevant document chunks:\n")

    for index, result in enumerate(results, start=1):
        print(f"--- Result {index} ---")
        print("Similarity:", round(result["similarity"], 4))
        print("Content:", result["content"])
        print()
else:
    print("No matching document chunks found.")
