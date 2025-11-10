# import json
# from tqdm import tqdm
# from chromadb import PersistentClient        # ✅ use this class
# from nomic.embed import text as nomic_text

# # 1. Initialize a persistent Chroma client
# # ✅ this guarantees disk persistence
# chroma_client = PersistentClient(path="data/chroma_store")
# collection = chroma_client.get_or_create_collection("projects_codebase")

# # 2. Load parsed chunks
# with open("chunks/parsed_chunks.json", "r", encoding="utf-8") as f:
#     chunks = json.load(f)

# print(f"📦 Loaded {len(chunks)} chunks to embed")

# # 3. Embed and store
# for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
#     text = chunk["content"]
#     metadata = {
#         "file": chunk["file"],
#         "name": chunk["name"],
#         "type": chunk["type"]
#     }

#     try:
#         response = nomic_text(texts=[text])
#         emb = response["embeddings"][0]
#         collection.add(
#             ids=[str(i)],
#             embeddings=[emb],
#             metadatas=[metadata],
#             documents=[text]
#         )
#     except Exception as e:
#         print(f"⚠️ Skipping {chunk['file']}: {e}")

# print(f"✅ Stored {len(chunks)} embeddings in persistent ChromaDB")
# print("💾 Embeddings saved to data/chroma_store")


# embed_store.py
import os
from typing import List, Dict, Any, Optional, Iterable, Tuple
import chromadb
from chromadb import Client
from chromadb.config import Settings

PERSIST_DIR = "data/chroma_store"
COLLECTION_NAME = "projects_codebase"

_client = None
_collection = None


def get_client():
    """Create or return a single global Chroma client."""
    global _client
    if _client is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
    return _client


def get_collection():
    """Create or return a single global Chroma collection."""
    global _collection
    if _collection is None:
        client = get_client()
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = client.create_collection(COLLECTION_NAME)
    return _collection


def upsert_chunks(chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Upsert a list of chunks: [{"id": str, "text": str, "metadata": {...}}, ...]
    Returns (n_ids, n_metadatas)
    """
    col = get_collection()
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]
    col.upsert(ids=ids, documents=texts, metadatas=metadatas)
    return (len(ids), len(metadatas))


def delete_where(where: Dict[str, Any]) -> int:
    """
    Delete documents by where-filter. Returns number of deleted items (best-effort).
    """
    col = get_collection()
    # Chroma doesn't return count; we estimate by prefetching matching ids
    existing = col.get(where=where, include=[])
    ids = existing.get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids or [])


def get_stats_for_project(project_id: str) -> Dict[str, Any]:
    col = get_collection()
    res = col.get(where={"project_id": project_id}, include=[])
    return {
        "project_id": project_id,
        "chunk_count": len(res.get("ids", []))
    }


def list_files_for_project(
        project_id: str, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    from fnmatch import fnmatch
    col = get_collection()
    res = col.get(where={"project_id": project_id}, include=["metadatas"])
    files = {}
    for md in res.get("metadatas", []):
        rel = md.get("rel_path")
        if not rel:
            continue
        if pattern and not fnmatch(rel, pattern):
            continue
        files.setdefault(
            rel, {
                "rel_path": rel, "filetype": md.get("filetype"), "chunks": 0})
        files[rel]["chunks"] += 1
    return sorted(files.values(), key=lambda x: x["rel_path"])


def list_documents_for_project(project_id: str) -> List[Dict[str, Any]]:
    col = get_collection()
    res = col.get(where={"project_id": project_id}, include=["metadatas"])
    docs = {}
    for md in res.get("metadatas", []):
        doc_id = md.get("doc_id")
        if not doc_id:
            continue
        entry = docs.setdefault(
            doc_id, {
                "doc_id": doc_id, "source": md.get("source"), "chunk_count": 0})
        entry["chunk_count"] += 1
    return list(docs.values())


def get_chunks(project_id: str,
               rel_path: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    col = get_collection()
    where = {"project_id": project_id}
    if rel_path:
        where["rel_path"] = rel_path
    res = col.get(where=where, include=["documents", "metadatas"])
    items = []
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    mds = res.get("metadatas", [])
    for i in range(min(len(ids), limit)):
        items.append({"id": ids[i], "text": docs[i], "metadata": mds[i]})
    return items


def query(query_embedding, top_k: int = 5, where: dict |
          None = None, include: list[str] | None = None):
    """
    Wrapper around chroma Collection.query with safe 'include' defaults.
    Valid include items for query(): 'documents', 'embeddings', 'metadatas', 'distances', 'uris', 'data'
    ('ids' is NOT valid for query()).
    """
    col = get_collection()

    # Default include set for query (NO 'ids')
    if include is None:
        include = ["documents", "metadatas", "distances"]

    # Chroma expects a list for query_embeddings, even for a single vector
    q = col.query(
        query_embeddings=[query_embedding],
        n_results=max(1, int(top_k)),
        where=where or {},
        include=include,
    )

    # Normalize result to a simple list of matches
    matches = []
    if q and q.get("documents"):
        docs = q["documents"][0]
        metas = q.get("metadatas", [[]])[0]
        dists = q.get("distances", [[]])[0]
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else None
            matches.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist,
                }
            )
    return matches
