# ingest_repo.py
import os
import uuid
import json
import subprocess
from typing import Dict, Any, List, Optional, Iterable, Tuple

from src.pipeline.embed_store import upsert_chunks
from src.pipeline.parse_chunk import parse_file, parse_folder

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "bin",
    "obj"}


def _should_skip_dir(dirname: str) -> bool:
    base = os.path.basename(dirname)
    return base in IGNORE_DIRS


def ingest_folder(
    folder_path: str,
    project_id: str,
    project_name: str,
    repo_url: Optional[str] = None,
    branch: Optional[str] = None,
    exts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Walk `folder_path`, parse supported files, upsert chunks.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    files = []
    for root, dirs, filenames in os.walk(folder_path):
        # prune ignored dirs
        dirs[:] = [
            d for d in dirs if not _should_skip_dir(
                os.path.join(
                    root, d))]
        for f in filenames:
            if exts:
                if any(f.endswith(e) for e in exts):
                    files.append(os.path.join(root, f))
            else:
                files.append(os.path.join(root, f))

    total_chunks = 0
    file_count = 0

    for fp in files:
        chunks = parse_file(
            fp,
            project_id=project_id,
            project_name=project_name,
            repo_url=repo_url,
            branch=branch,
        )
        if chunks:
            n_ids, _ = upsert_chunks(chunks)
            total_chunks += n_ids
            file_count += 1

    return {"project_id": project_id, "files_ingested": file_count,
            "chunks_upserted": total_chunks}


def ingest_repo(
    repo_url: str,
    dest_dir: str,
    branch: Optional[str],
    project_id: str,
    project_name: str
) -> Dict[str, Any]:
    """
    Clone (or pull) the repo into dest_dir and then call ingest_folder on it.
    """
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
    if not os.path.exists(dest_dir):
        cmd = ["git", "clone", repo_url, dest_dir]
        subprocess.run(cmd, check=True)
    # checkout / pull
    if branch:
        subprocess.run(
            ["git", "-C", dest_dir, "checkout", branch], check=False)
    subprocess.run(["git", "-C", dest_dir, "pull"], check=False)

    return ingest_folder(dest_dir, project_id=project_id,
                         project_name=project_name, repo_url=repo_url, branch=branch)


def upsert_files(
    files: List[str],
    project_id: str,
    project_name: str,
    repo_url: Optional[str] = None,
    branch: Optional[str] = None
) -> Dict[str, Any]:
    total_chunks = 0
    file_count = 0
    for fp in files:
        chunks = parse_file(
            fp,
            project_id=project_id,
            project_name=project_name,
            repo_url=repo_url,
            branch=branch,
        )
        if chunks:
            n_ids, _ = upsert_chunks(chunks)
            total_chunks += n_ids
            file_count += 1
    return {"project_id": project_id, "files_upserted": file_count,
            "chunks_upserted": total_chunks}
