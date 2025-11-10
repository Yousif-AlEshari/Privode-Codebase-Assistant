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
