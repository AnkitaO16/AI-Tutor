# from llama_index.llms.ollama import Ollama
# from llama_index.core import Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import chromadb
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import VectorStoreIndex, StorageContext
#
# # 🔹 Configure Ollama
# Settings.llm = Ollama(model="llama3.1:8b", request_timeout=30.0, base_url="http://127.0.0.1:11434")
#
# # 🔹 Configure embeddings
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#
# # Chroma client
# chroma_client = chromadb.PersistentClient(path="chroma_db")
#
# def query_notes(question: str, class_name: str, subject: str, topic: str, k: int = 3):
#     try:
#         # Build collection name
#         collection_name = f"{class_name}_{subject}_{topic}".lower().replace(" ", "_")
#
#         # Get collection
#         try:
#             collection = chroma_client.get_collection(name=collection_name)
#         except Exception:
#             return {"error": f"❌ Collection '{collection_name}' not found. Did you ingest notes first?"}
#
#         # Wrap collection into LlamaIndex VectorStore
#         vector_store = ChromaVectorStore(chroma_collection=collection)
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)
#
#         # Build index from vector store
#         index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
#
#         # Query with LLM
#         query_engine = index.as_query_engine(llm=Settings.llm, similarity_top_k=k)
#         response = query_engine.query(question)
#
#         return {
#             "answer": str(response),
#             "sources": [str(n) for n in response.source_nodes] if hasattr(response, "source_nodes") else []
#         }
#
#     except Exception as e:
#         return {"error": f"Query failed: {e}"}
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# ================================
# 🔹 OPTION 1: Use Groq API (recommended: cloud, no memory issue)
# pip install groq
# ================================
from llama_index.llms.groq import Groq

Settings.llm = Groq(model="llama3-8b-8192")  # Free tier available
# Docs: https://console.groq.com/keys

# ================================
# 🔹 OPTION 2: Local Ollama (comment Groq above, uncomment below)
# Needs Ollama running locally with model pulled (e.g., `ollama pull llama3.1:8b`)
# Heavy on RAM (your error came from this type of setup)
# ================================
# from llama_index.llms.ollama import Ollama
# Settings.llm = Ollama(
#     model="llama3.1:8b",
#     request_timeout=60.0,
#     base_url="http://127.0.0.1:11434"
# )

# ✅ Configure embeddings (lightweight, stays local)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ✅ Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_db")


def query_notes(question: str, class_name: str, subject: str, topic: str, k: int = 3):
    """
    Query Chroma collection for given class/subject/topic,
    then pass results to LLM (Groq API by default).
    """
    try:
        # Build collection name
        collection_name = f"{class_name}_{subject}_{topic}".lower().replace(" ", "_")

        # Get collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            return {"error": f"❌ Collection '{collection_name}' not found. Did you ingest notes first?"}

        # Wrap collection into LlamaIndex VectorStore
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build index from vector store
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        # Query with chosen LLM
        query_engine = index.as_query_engine(llm=Settings.llm, similarity_top_k=k)
        response = query_engine.query(question)

        return {
            "answer": str(response),
            "sources": [str(n) for n in getattr(response, "source_nodes", [])]
        }

    except Exception as e:
        return {"error": f"Query failed: {e}"}
