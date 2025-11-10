from sentence_transformers import SentenceTransformer
from chromadb.errors import NotFoundError
import os
import textwrap
from chromadb import PersistentClient
from nomic.embed import text as nomic_text
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Global clients ---
CHROMA_PATH = "data/chroma_store"
COLLECTION_NAME = "projects_codebase"


client = PersistentClient(path="data/chroma_store")
try:
    collection = client.get_collection("projects_codebase")
except NotFoundError:
    collection = client.create_collection("projects_codebase")

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("LLM_key"),
)

# -----------------------------------------------------
# 🔹 1. Generate embedding
# -----------------------------------------------------


# 384-dimensional model (fast, small, consistent)
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()

# -----------------------------------------------------
# 🔹 2. Retrieve context chunks (standalone helper)
# -----------------------------------------------------


def retrieve_context(query_emb, top_k: int = 3) -> str:
    """
    Retrieve top chunks from the Chroma collection and build a formatted context string.
    """
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)

    context = ""
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        snippet = results["documents"][0][i]
        file = meta.get("file") or meta.get("rel_path") or "unknown_file"
        name = meta.get("name") or meta.get("project_name") or "unknown_name"

        context += f"\nFile: {file}\nProject: {name}\nSnippet:\n{snippet}\n---\n"

    return context.strip()


# -----------------------------------------------------
# 🔹 3. Generate answer with Groq (LLaMA 3 / Mixtral)
# -----------------------------------------------------
def llm_answer(question: str, matches: list[dict]) -> str:
    """
    Builds an LLM prompt from retrieved chunks and returns the model's concise answer.
    """
    # Build code context from matches
    context = ""
    for m in matches:
        md = m.get("metadata", {})
        rel_path = md.get("rel_path", md.get("file", "unknown"))
        idx = md.get("chunk_idx", 0)
        text = m.get("text", "")
        context += f"\nFile: {rel_path} (chunk {idx})\n{text}\n---\n"

    prompt = f"""
You are an AI assistant helping developers understand their private codebase.
Answer the question below using only the provided code context.
Be concise and cite file names and function names when relevant.

Question:
{question}

Relevant Code:
{context}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # or "llama3-70b"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )

    answer = response.choices[0].message.content.strip()
    return textwrap.fill(answer, width=90)


# -----------------------------------------------------
# 🔹 4. CLI mode (for quick manual testing)
# -----------------------------------------------------
if __name__ == "__main__":
    query = input("Ask a question about the codebase:\n> ")
    query_emb = embed_text(query)
    context = retrieve_context(query_emb, top_k=3)

    print("\n⏳ Generating answer with Groq ...\n")
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"""
You are an AI assistant helping developers understand their private codebase.
Answer the question below using only the provided code context.
Be concise and cite file names and function names when relevant.

Question:
{query}

Relevant Code:
{context}
""",
            }
        ],
        temperature=0.3,
        max_tokens=600,
    )

    print("\n💡 AI Answer:\n")
    print(textwrap.fill(response.choices[0].message.content.strip(), width=90))
