# from chromadb import PersistentClient
# # use same model that built embeddings
# from nomic.embed import text as nomic_text
# from pprint import pprint


# # 1️⃣ Connect to your persistent store
# client = PersistentClient(path="data/chroma_store")
# collection = client.get_collection("projects_codebase")

# print(f"✅ Connected to collection: {collection.name}")
# print(f"Total documents: {collection.count()}")


# # 2️⃣ Ask a natural-language question
# query = input("Enter your question about the codebase:\n> ")

# # 3️⃣ Generate the embedding for the query
# response = nomic_text(texts=[query])
# query_embedding = response["embeddings"][0]

# # 4️⃣ Query the vector database for the top matches
# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=5,          # top-k similar chunks
#     include=["documents", "metadatas", "distances"]
# )

# # 5️⃣ Print results neatly
# print("\n🔍 Top matching code snippets:\n")
# for i in range(len(results["ids"][0])):
#     meta = results["metadatas"][0][i]
#     dist = results["distances"][0][i]
#     print(f"Result {i + 1} (distance {dist:.4f})")
#     print("File:", meta.get("file"))
#     print("Name:", meta.get("name"))
#     print("Type:", meta.get("type"))
#     print("Snippet:\n", results["documents"][0][i][:300], "…\n")


# retrieval.py
from src.pipeline.embed_store import query
from src.pipeline.answer_generation import embed_text, llm_answer


def ask_question(question: str, project_id: str |
                 None, top_k: int = 5) -> dict:
    q_emb = embed_text(question)
    where = {}
    if project_id:
        # ensure your upserts set 'project_id' in metadata
        where["project_id"] = project_id

    matches = query(q_emb, top_k=top_k, where=where)

    if not matches:
        return {
            "answer": "I couldn’t find relevant chunks for that question in the selected project.",
            "matches": [],
        }

    answer = llm_answer(question, matches)
    return {
        "answer": answer,
        "matches": matches,
    }
