# app.py
import os
import sys
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../..")))
from src.services import api_server
import uvicorn
import threading
import json
import requests
import streamlit as st



API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def run_api():
    uvicorn.run(api_server.app, port=8000)


# Run FastAPI in background when Streamlit starts
if "api_thread_started" not in st.session_state:
    thread = threading.Thread(target=run_api, daemon=True)
    thread.start()
    st.session_state["api_thread_started"] = True

st.set_page_config(page_title="Codebase Assistant", layout="wide")

st.title("💡 Private Codebase Assistant (Multi-Project)")

# ---------------- Sidebar navigation ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Ask", "Search (no LLM)", "Projects", "Ingest / Upload", "Browse"]
)

# Helper for API calls


def api_get(path, **kwargs):
    return requests.get(f"{API_BASE}{path}", **kwargs).json()


def api_post(path, **kwargs):
    resp = requests.post(f"{API_BASE}{path}", **kwargs)
    try:
        return resp.json()
    except Exception:
        return {"error": resp.text, "status_code": resp.status_code}


# ---------------- Ask tab ----------------
if menu == "Ask":
    st.header("🤖 Ask about your code")
    projects = api_get("/projects")

    if projects:
        project_options = {p["project_name"]: p["project_id"]
                           for p in projects}
        project_name = st.selectbox(
            "Select project", list(project_options.keys()), key="ask_project_select"
        )
        project_id = project_options.get(project_name)

        if project_id:
            # Unique key added to text area
            question = st.text_area(
                "Question",
                placeholder="Input your question here...",
                key="ask_question_input"
            )
            top_k = st.slider("Top-K Chunks", 1, 10, 5, key="ask_topk_slider")

            if st.button("Ask", key="ask_button"):
                if not question.strip():
                    st.warning("Please enter a question first.")
                else:
                    with st.spinner("Querying LLM..."):
                        res = api_post("/ask", json={
                            "question": question,
                            "top_k": top_k,
                            "project_id": project_id
                        })
                    st.subheader("Answer")
                    st.write(res.get("answer", "No answer"))
                    with st.expander("Show matched chunks"):
                        for m in res.get("matches", []):
                            st.markdown(
                                f"**{
                                    m['metadata'].get('rel_path')} (chunk {
                                    m['metadata'].get('chunk_idx')})**"
                            )
                            st.code(m["text"], language="python")
        else:
            st.warning("Please select a project first.")
    else:
        st.warning("No projects found. Please create one from the Projects tab.")

# ---------------- Search tab ----------------
elif menu == "Search (no LLM)":
    st.header("🔍 Quick search (no LLM)")
    projects = api_get("/projects")

    if not projects:
        st.warning(
            "⚠️ No projects found. Please create one from the **Projects** tab first.")
    else:
        project_options = {p["project_name"]: p["project_id"]
                           for p in projects}
        project_name = st.selectbox(
            "Select project", list(
                project_options.keys()))
        if project_name:
            project_id = project_options[project_name]

            q = st.text_area("Search keyword", key="search_text_area")
            if st.button("Search"):
                with st.spinner("Searching chunks..."):
                    res = api_get(
                        "/search",
                        params={
                            "q": q,
                            "project_id": project_id})
                st.write(f"Found {res.get('count', 0)} results.")
                for r in res.get("results", []):
                    st.markdown(
                        f"**{r['rel_path']}** (chunk {r['chunk_idx']})")
                    st.code(r["preview"], language="python")

    q = st.text_area("Search keyword", key="search_text_area")
    if st.button("Search"):
        with st.spinner("Searching chunks..."):
            res = api_get("/search", params={"q": q, "project_id": project_id})
        st.write(f"Found {res.get('count', 0)} results.")
        for r in res.get("results", []):
            st.markdown(f"**{r['rel_path']}** (chunk {r['chunk_idx']})")
            st.code(r["preview"], language="python")

# ---------------- Projects tab ----------------
elif menu == "Projects":
    st.header("📁 Projects")
    projects = api_get("/projects")
    st.dataframe(projects)

    st.subheader("Create new project")
    with st.form("create_project"):
        name = st.text_input("Project name")
        repo = st.text_input("Repo URL (optional)")
        path = st.text_input("Root path (optional)")
        branch = st.text_input("Branch (optional)")
        submitted = st.form_submit_button("Create")
        if submitted:
            res = api_post("/projects", json={
                "project_name": name,
                "repo_url": repo or None,
                "root_path": path or None,
                "branch": branch or None
            })
            st.write("Server response:", res)

            if "project_id" in res:
                st.success(
                    f"✅ Created project: {
                        res['project_name']} ({
                        res['project_id']})")
            else:
                st.error("❌ Failed to create project. Check details below:")
                st.json(res)

    st.subheader("Delete project")
    pid = st.text_input("Project ID to delete")
    if st.button("Delete"):
        res = requests.delete(f"{API_BASE}/projects/{pid}").json()
        st.write(res)

# ---------------- Ingest / Upload tab ----------------
elif menu == "Ingest / Upload":
    st.header("📤 Ingest or Upload")
    projects = api_get("/projects")
    if not projects:
        st.warning("Create a project first.")
    else:
        project_options = {p["project_name"]: p["project_id"]
                           for p in projects}
        project_name = st.selectbox(
            "Select project", list(
                project_options.keys()))
        project_id = project_options[project_name]

        mode = st.radio(
            "Mode", [
                "Ingest Folder", "Ingest Repo", "Upload Files", "Re-Embed"])
        if mode == "Ingest Folder":
            folder = st.text_input("Folder path")
            exts = st.text_input("Extensions (comma separated, e.g. .py,.js)")

            if not project_id:
                st.warning("Please select a project first.")
            elif st.button("Ingest Folder"):
                res = api_post(f"/projects/{project_id}/ingest-folder", json={
                    "folder_path": folder,
                    "extensions": [e.strip() for e in exts.split(",") if e.strip()]
                })
                st.json(res)
        elif mode == "Ingest Repo":
            repo = st.text_input("Repo URL")
            branch = st.text_input("Branch", "main")
            dest = st.text_input(
                "Destination directory", f"data/projects/{project_name.lower().replace(' ', '_')}")
            if not project_id:
                st.warning("Please select a project first.")

            elif st.button("Ingest Repo"):
                res = api_post(f"/projects/{project_id}/ingest-repo", json={
                    "repo_url": repo,
                    "branch": branch,
                    "dest_dir": dest
                })
                st.json(res)
        elif mode == "Upload Files":
            files = st.file_uploader(
                "Upload files", accept_multiple_files=True)
            if not project_id:
                st.warning("Please select a project first.")
            elif st.button("Upload"):
                uploaded = []
                for f in files:
                    uploaded.append(("files", (f.name, f.getvalue())))
                res = requests.post(
                    f"{API_BASE}/projects/{project_id}/upsert-files",
                    files=uploaded).json()
                st.json(res)
        elif mode == "Re-Embed":
            strategy = st.selectbox("Strategy", ["replace", "append"])
            if not project_id:
                st.warning("Please select a project first.")
            elif st.button("Re-Embed Project"):
                res = api_post(
                    f"/projects/{project_id}/reembed",
                    json={
                        "strategy": strategy})
                st.json(res)

# ---------------- Browse tab ----------------
elif menu == "Browse":
    st.header("🗂 Browse project contents")
    projects = api_get("/projects")
    if not projects:
        st.warning("Create a project first.")
    else:
        project_options = {p["project_name"]: p["project_id"]
                           for p in projects}
        project_name = st.selectbox(
            "Select project", list(
                project_options.keys()))
        project_id = project_options[project_name]

        view = st.radio("View", ["Files", "Documents", "Chunks"])
        if view == "Files":
            pattern = st.text_input("Filename pattern (optional, e.g. *.py)")
            res = api_get(
                f"/projects/{project_id}/files",
                params={
                    "pattern": pattern})
            st.dataframe(res)
        elif view == "Documents":
            res = api_get(f"/projects/{project_id}/documents")
            st.dataframe(res)
        else:
            rel = st.text_input("Relative path (optional)")
            res = api_get(
                f"/projects/{project_id}/chunks",
                params={
                    "rel_path": rel})
            for c in res:
                st.markdown(
                    f"**{c['metadata'].get('rel_path')}** (chunk {c['metadata'].get('chunk_idx')})")
                st.code(c["text"], language="python")
