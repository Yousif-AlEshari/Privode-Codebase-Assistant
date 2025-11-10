# api_server.py
from src.pipeline.embed_store import get_collection
import os
import json
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from src.pipeline.embed_store import (
    get_stats_for_project,
    delete_where,
    list_files_for_project,
    list_documents_for_project,
    get_chunks,
)
from src.pipeline.ingest_repo import ingest_folder, ingest_repo, upsert_files
from src.pipeline.retrieval import ask_question

PROJECTS_FILE = "data/projects.json"
os.makedirs("data", exist_ok=True)


def _load_projects() -> Dict[str, Any]:
    if not os.path.exists(PROJECTS_FILE):
        return {"projects": []}
    try:
        with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "projects" not in data:
                data = {"projects": []}
            return data
    except json.JSONDecodeError:
        # Auto-recover from corrupt file
        return {"projects": []}


def _save_projects(data: Dict[str, Any]) -> None:
    with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_project(project_id: str) -> Optional[Dict[str, Any]]:
    db = _load_projects()
    for p in db["projects"]:
        if p["project_id"] == project_id:
            return p
    return None


class ProjectCreate(BaseModel):
    project_name: str
    repo_url: Optional[str] = None
    root_path: Optional[str] = None
    branch: Optional[str] = None


class Project(BaseModel):
    project_id: str
    project_name: str
    repo_url: Optional[str] = None
    root_path: Optional[str] = None
    branch: Optional[str] = None
    created_at: str


class IngestFolderRequest(BaseModel):
    folder_path: str
    extensions: Optional[List[str]] = None
    ignore_git: Optional[bool] = True


class IngestRepoRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = None
    dest_dir: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    project_id: Optional[str] = None
    rel_path_filter: Optional[str] = None


class ReembedRequest(BaseModel):
    strategy: str = "replace"  # "replace" or "append"


app = FastAPI(title="Codebase Assistant API", version="1.0")

# ---- Projects CRUD ----


@app.post("/projects", response_model=Project)
def create_project(req: ProjectCreate):
    if not req.repo_url and not req.root_path:
        raise HTTPException(status_code=400,
                            detail="Provide either repo_url or root_path.")
    proj = {
        "project_id": f"proj_{uuid.uuid4().hex[:8]}",
        "project_name": req.project_name,
        "repo_url": req.repo_url,
        "root_path": req.root_path,
        "branch": req.branch,
        "created_at": __import__("datetime").datetime.utcnow().isoformat()
    }
    db = _load_projects()
    db["projects"].append(proj)
    _save_projects(db)
    return proj


@app.get("/projects")
def list_projects():
    db = _load_projects()
    items = []
    for p in db["projects"]:
        stats = get_stats_for_project(p["project_id"])
        items.append({
            "project_id": p["project_id"],
            "project_name": p["project_name"],
            "repo_url": p.get("repo_url"),
            "root_path": p.get("root_path"),
            "branch": p.get("branch"),
            "created_at": p["created_at"],
            "chunk_count": stats["chunk_count"]
        })
    return items


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    stats = get_stats_for_project(project_id)
    return {**p, **stats}


@app.delete("/projects/{project_id}")
def delete_project(project_id: str):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    deleted = delete_where({"project_id": project_id})
    db = _load_projects()
    db["projects"] = [x for x in db["projects"] if x["project_id"] != project_id]
    _save_projects(db)
    return {"status": "deleted", "project_id": project_id,
            "deleted_chunks": deleted}

# ---- Ingestion / Upsert ----


@app.post("/projects/{project_id}/ingest-folder")
def api_ingest_folder(project_id: str, req: IngestFolderRequest):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    if not os.path.exists(req.folder_path):
        raise HTTPException(400, f"Folder not found: {req.folder_path}")
    result = ingest_folder(
        folder_path=req.folder_path,
        project_id=project_id,
        project_name=p["project_name"],
        repo_url=p.get("repo_url"),
        branch=p.get("branch"),
        exts=req.extensions
    )
    return result


@app.post("/projects/{project_id}/ingest-repo")
def api_ingest_repo(project_id: str, req: IngestRepoRequest):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    res = ingest_repo(
        repo_url=req.repo_url,
        dest_dir=req.dest_dir,
        branch=req.branch,
        project_id=project_id,
        project_name=p["project_name"]
    )
    # store the root_path/repo_url in the project registry if missing
    db = _load_projects()
    for proj in db["projects"]:
        if proj["project_id"] == project_id:
            proj["repo_url"] = req.repo_url
            proj["root_path"] = req.dest_dir
            proj["branch"] = req.branch
            break
    _save_projects(db)
    return res


@app.post("/projects/{project_id}/upsert-files")
async def api_upsert_files(
        project_id: str, files: List[UploadFile] = File(...)):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    os.makedirs("data/uploads", exist_ok=True)
    saved_paths = []
    for f in files:
        dest = os.path.join("data/uploads", f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved_paths.append(dest)
    res = upsert_files(
        saved_paths,
        project_id=project_id,
        project_name=p["project_name"],
        repo_url=p.get("repo_url"),
        branch=p.get("branch"))
    return res


@app.post("/projects/{project_id}/reembed")
def api_reembed_project(project_id: str, req: ReembedRequest):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    # replace: delete old vectors, then re-ingest folder if path exists
    if req.strategy not in {"replace", "append"}:
        raise HTTPException(400, "strategy must be 'replace' or 'append'")
    if req.strategy == "replace":
        delete_where({"project_id": project_id})

    root_path = p.get("root_path")
    if not root_path or not os.path.isdir(root_path):
        return {"status": "ok",
                "message": "No root_path on record, nothing to re-embed. Use ingest-folder or ingest-repo."}
    res = ingest_folder(
        folder_path=root_path,
        project_id=project_id,
        project_name=p["project_name"],
        repo_url=p.get("repo_url"),
        branch=p.get("branch")
    )
    return {"status": "ok", **res}

# ---- Browsing / Discovery ----


@app.get("/projects/{project_id}/files")
def api_list_files(project_id: str, pattern: Optional[str] = None):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return list_files_for_project(project_id, pattern=pattern)


@app.get("/projects/{project_id}/documents")
def api_list_documents(project_id: str):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return list_documents_for_project(project_id)


@app.get("/projects/{project_id}/chunks")
def api_list_chunks(
        project_id: str, rel_path: Optional[str] = None, limit: int = 200):
    p = _get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return get_chunks(project_id, rel_path=rel_path, limit=limit)

# ---- Ask / Search ----


@app.post("/ask")
def api_ask(req: AskRequest):
    return ask_question(
        question=req.question,
        top_k=req.top_k,
        project_id=req.project_id
    )


@app.get("/search")
def api_search(q: str, project_id: str):
    """Lightweight retrieval-only search without LLM."""
    try:
        from src.pipeline.embed_store import get_collection
        from src.pipeline.answer_generation import embed_text

        col = get_collection()
        q_emb = embed_text(q)
        where = {"project_id": project_id}

        # Perform similarity search
        results = col.query(
            query_embeddings=[q_emb],
            where=where,
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return {
            "count": len(docs),
            "results": [
                {
                    "rel_path": m.get("rel_path"),
                    "chunk_idx": m.get("chunk_idx"),
                    "preview": doc[:300],
                    "distance": d
                }
                for doc, m, d in zip(docs, metas, dists)
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
