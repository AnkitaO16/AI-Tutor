import chromadb
from chromadb.utils import embedding_functions

# 1. Load existing ChromaDB
chroma_client = chromadb.PersistentClient(path="../chroma_db")

# 2. Same embedding function
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 3. Load collection
collection = chroma_client.get_or_create_collection(
    name="teacher_notes",
    embedding_function=embedding_func
)

# 4. Query
query = "What is the first law of thermodynamics?"
results = collection.query(query_texts=[query], n_results=2)

print("🔎 Query results:")
print(results)
