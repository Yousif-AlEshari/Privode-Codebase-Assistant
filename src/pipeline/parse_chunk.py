# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pathlib import Path
# import json
# import ast

# # Simple Python-aware parser (for MVP)


# def python_function_chunker(code_text: str, file_path: str):
#     chunks = []
#     try:
#         tree = ast.parse(code_text)
#         for node in tree.body:
#             if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
#                 start_line = node.lineno - 1
#                 end_line = node.end_lineno if hasattr(
#                     node, "end_lineno") else start_line + 1
#                 lines = code_text.splitlines()[start_line:end_line]
#                 snippet = "\n".join(lines)
#                 chunks.append({
#                     "file": file_path,
#                     "type": node.__class__.__name__,
#                     "name": node.name,
#                     "content": snippet
#                 })
#     except Exception as e:
#         # fallback if AST fails
#         chunks = [{"file": file_path, "type": "raw",
#                    "name": "full", "content": code_text}]
#     return chunks


# # Generic fallback for non-Python files
# def generic_chunker(code_text: str, file_path: str):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#         separators=["\nclass ", "\ndef ", "\n\n", "\n", " "]
#     )
#     return [
#         {"file": file_path, "type": "block", "name": f"chunk_{i}", "content": chunk}
#         for i, chunk in enumerate(splitter.split_text(code_text))
#     ]


# def parse_files(docs):
#     all_chunks = []
#     for doc in docs:
#         path = doc["path"]
#         code = doc["content"]
#         ext = Path(path).suffix

#         if ext == ".py":
#             chunks = python_function_chunker(code, path)
#         else:
#             chunks = generic_chunker(code, path)

#         all_chunks.extend(chunks)
#     return all_chunks


# if __name__ == "__main__":
#     # Load docs produced by Stage 1
#     with open("ingestions/ingested_docs.json", "r", encoding="utf-8") as f:
#         docs = json.load(f)

#     chunks = parse_files(docs)
#     print(f"✅ Created {len(chunks)} chunks")
#     print(json.dumps(chunks[0], indent=2)[:500])

#     # Save for the embedding stage
#     with open("chunks/parsed_chunks.json", "w", encoding="utf-8") as f:
#         json.dump(chunks, f, ensure_ascii=False, indent=2)

import os
import uuid
import json
from datetime import datetime
from pathlib import Path


def chunk_text(text, max_length=1000):
    """
    Splits text into chunks of approximately `max_length` characters.
    Keeps your current sentence-splitting logic.
    """
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def parse_file(file_path, project_id=None, project_name=None,
               repo_url=None, branch=None):
    """
    Reads a file, splits it into chunks, and attaches extended metadata.
    """
    # ----- Core info -----
    abs_path = os.path.abspath(file_path)
    try:
        rel_path = os.path.relpath(file_path, start=os.getcwd())
    except ValueError:
        # Happens when file is on a different drive (e.g., D: vs C:)
        rel_path = abs_path
    filetype = Path(file_path).suffix
    size_bytes = os.path.getsize(file_path)
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    # doc_id stays per-file (consistent with old logic)
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, abs_path))

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text)
    results = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{project_id or 'default'}::{rel_path}::{idx}"

        metadata = {
            # ---- Existing keys ----
            "doc_id": doc_id,
            "source": os.path.basename(file_path),
            "page": 1,
            "chunk_idx": idx,
            # ---- New project-aware metadata ----
            "project_id": project_id or "default",
            "project_name": project_name or "Unnamed Project",
            "repo_url": repo_url or None,
            "branch": branch or None,
            "abs_path": abs_path,
            "rel_path": rel_path,
            "filetype": filetype,
            "size_bytes": size_bytes,
            "mtime": mtime,
        }

        results.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": metadata
        })

    return results


def parse_folder(folder_path, project_id=None,
                 project_name=None, repo_url=None, branch=None):
    """
    Recursively parse all text/code files in a folder into structured chunks.
    """
    all_chunks = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Skip binary/large or irrelevant files
            if not file.endswith(
                    (".py", ".js", ".ts", ".tsx", ".html", ".css", ".md", ".json", ".txt")):
                continue

            file_path = os.path.join(root, file)
            try:
                file_chunks = parse_file(
                    file_path,
                    project_id=project_id,
                    project_name=project_name,
                    repo_url=repo_url,
                    branch=branch
                )
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"[WARN] Failed to parse {file_path}: {e}")

    return all_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and chunk files into structured JSONL with metadata.")
    parser.add_argument("path", help="Path to a single file or folder.")
    parser.add_argument(
        "--project-id",
        default=str(
            uuid.uuid4()),
        help="Unique project ID or slug.")
    parser.add_argument(
        "--project-name",
        default="Unnamed Project",
        help="Human-readable project name.")
    parser.add_argument(
        "--repo-url",
        default=None,
        help="Optional repository URL.")
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional repository branch.")
    parser.add_argument(
        "--output",
        default="data/parsed_chunks.json",
        help="Output JSON file.")

    args = parser.parse_args()
    path = args.path

    if os.path.isdir(path):
        chunks = parse_folder(
            path,
            project_id=args.project_id,
            project_name=args.project_name,
            repo_url=args.repo_url,
            branch=args.branch
        )
    else:
        chunks = parse_file(
            path,
            project_id=args.project_id,
            project_name=args.project_name,
            repo_url=args.repo_url,
            branch=args.branch
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Parsed {len(chunks)} chunks written to {args.output}")
