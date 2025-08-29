import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ✅ set global embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def ingest_text_file(file_path: str, class_name: str, subject: str, topic: str) -> dict:
    """
    Ingest a teacher's text file into Chroma collection
    Collection name format: class_subject_topic
    """
    try:
        # load file into LlamaIndex docs
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        # prepare collection (class + subject + topic)
        collection_name = f"{class_name}_{subject}_{topic}".replace(" ", "_").lower()

        db = chromadb.PersistentClient(path="./chroma_db")
        collection = db.get_or_create_collection(collection_name)

        # wrap in LlamaIndex
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # build index and insert
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        return {
            "status": "success",
            "collection": collection_name,
            "docs": len(documents)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
