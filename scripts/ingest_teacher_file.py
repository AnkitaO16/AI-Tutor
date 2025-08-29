#
# import os
# import sys
# import chromadb
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#
# # 1. Check args
# if len(sys.argv) < 4:
#     print("❌ Usage: python ingest_teacher_file.py <file.txt> <subject> <topic>")
#     sys.exit(1)
#
# file_path, subject, topic = sys.argv[1], sys.argv[2], sys.argv[3]
#
# if not os.path.exists(file_path):
#     print(f"❌ File not found: {file_path}")
#     sys.exit(1)
#
# # 2. Read file content
# with open(file_path, "r", encoding="utf-8") as f:
#     content = f.read().strip()
#
# docs = [line.strip() for line in content.split("\n") if line.strip()]
#
# # 3. Init Chroma client + embedding model
# chroma_client = chromadb.PersistentClient(path="chroma_db")
#
# # sanitize collection name
# # collection_name = f"{subject.strip().lower()}_{topic.strip().lower()}".replace(" ", "_")
#
# collection_name = f"{subject.strip().lower()}_{topic.strip().lower()}".replace(" ", "_")
#
# collection = chroma_client.get_or_create_collection(name=collection_name)
#
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#
# # 4. Embed and insert
# embeddings = embed_model.get_text_embedding_batch(docs)
# ids = [f"{collection_name}_doc_{i}" for i in range(len(docs))]
#
# collection.add(
#     documents=docs,
#     embeddings=embeddings,
#     ids=ids
# )
#
# print(f"✅ Ingested {len(docs)} chunks from '{file_path}' into collection '{collection_name}' in ChromaDB!")
import os
import sys
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

if len(sys.argv) < 4:
    print("❌ Usage: python ingest_teacher_file.py <file.txt> <class> <subject> <topic>")
    sys.exit(1)

file_path, class_name, subject, topic = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    sys.exit(1)

# Read file
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read().strip()

docs = [line.strip() for line in content.split("\n") if line.strip()]

# Init Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Collection name = class_subject_topic
collection_name = f"{class_name}_{subject}_{topic}".lower().replace(" ", "_")
collection = chroma_client.get_or_create_collection(name=collection_name)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Embed & add
embeddings = embed_model.get_text_embedding_batch(docs)
ids = [f"{collection_name}_doc_{i}" for i in range(len(docs))]

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=ids
)

print(f"✅ Ingested {len(docs)} chunks into '{collection_name}'")
