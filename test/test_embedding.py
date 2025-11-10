import json
from tqdm import tqdm
from chromadb import PersistentClient        # ✅ use this class
from nomic.embed import text as nomic_text

# 1. Initialize a persistent Chroma client
# ✅ this guarantees disk persistence
chroma_client = PersistentClient(path="data/chroma_store")
collection = chroma_client.get_or_create_collection("projects_codebase")

# 2. Load parsed chunks
with open("data/chunks/parsed_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"📦 Loaded {len(chunks)} chunks to embed")

# 3. Embed and store
for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
    text = chunk["content"]
    metadata = {
        "file": chunk["file"],
        "name": chunk["name"],
        "type": chunk["type"]
    }

    try:
        response = nomic_text(texts=[text])
        emb = response["embeddings"][0]
        collection.add(
            ids=[str(i)],
            embeddings=[emb],
            metadatas=[metadata],
            documents=[text]
        )
    except Exception as e:
        print(f"⚠️ Skipping {chunk['file']}: {e}")

print(f"✅ Stored {len(chunks)} embeddings in persistent ChromaDB")
print("💾 Embeddings saved to data/chroma_store")
