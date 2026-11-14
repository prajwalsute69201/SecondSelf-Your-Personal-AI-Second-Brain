"""SecondSelf Streamlit Application (Phase 4, Task 4.2.2).

Wired user-facing web application for SecondSelf:
- Header: Application title ('🧠 SecondSelf') with 'Refresh Graph' action (`build_graph.build_and_save_graph`).
- Ask Section: 'Ask your brain' input box, submit button, answer panel, and source citations (`ask.ask`).
- Graph Section: Interactive force-directed knowledge graph embedding vis-network (`components.html`).
- Sidebar: Quick note capture (`capture.capture_note`), pipeline processor (`pipeline.process`),
  and knowledge brain statistics (node/edge counts from `graph.json`).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

# Ensure secondself project root is in sys.path so modules and lib are importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

import ask
import build_graph
import capture
from lib import auth
from lib.constants import (
    CATEGORY_DEFINITIONS,
    DEFAULT_CATEGORY,
    VALID_CATEGORIES,
    VALID_PARA_CATEGORIES,
)
import lib.database as db
from lib.database import DEFAULT_USER_ID, get_all_system_stats
import lib.embeddings as emb
from lib.models import (
    AskResult,
    CaptureMeta,
    CaptureResult,
    User,
    UserSession,
    WikiNote,
)
import lib.storage as storage
import pipeline

logger = logging.getLogger(__name__)

# Fallback empty graph schema
DEFAULT_GRAPH_DATA: dict[str, Any] = {
    "nodes": [],
    "edges": [],
    "metadata": {
        "generated_at": None,
        "node_count": 0,
        "edge_count": 0,
    },
}

__all__ = [
    "sanitize_error_message",
    "load_embeddings",
    "load_embedding_model",
    "load_note_embeddings",
    "load_graph",
    "load_graph_data",
    "get_graph_html",
    "get_brain_stats",
    "capture_uploaded_file",
    "save_edited_note",
    "delete_wiki_note",
    "render_header",
    "render_ask_section",
    "render_graph_section",
    "render_sidebar",
    "render_auth_page",
    "render_admin_portal",
    "main",
]


def get_active_groq_model() -> str:
    """Retrieve configured Groq model identifier from st.secrets, env, or default."""
    try:
        if hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets:
            return str(st.secrets["GROQ_MODEL"]).strip()
    except Exception:
        pass
    return os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def sanitize_error_message(exc: Exception | str) -> str:
    """Format and sanitize error messages for safe user-facing display.

    Redacts API keys, credentials, local filesystem paths, and internal stack traces.

    Args:
        exc: Exception instance or error message string.

    Returns:
        str: Clean, actionable user-facing error message.
    """
    raw_msg = str(exc)

    # Redact Groq API keys (gsk_...)
    sanitized = re.sub(r"gsk_[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", raw_msg)

    # Redact absolute filesystem paths (Windows and Unix-like)
    sanitized = re.sub(r"[A-Za-z]:\\[\w\s\\/.\-()]+", "[local path]", sanitized)
    sanitized = re.sub(r"/(?:[\w.\-()]+/)+[\w.\-()]+", "[local path]", sanitized)

    # Strip tracebacks
    sanitized = re.sub(r"Traceback \(most recent call last\):.*", "", sanitized, flags=re.DOTALL).strip()

    lower_msg = sanitized.lower()

    if (
        "groq_api_key" in lower_msg
        or "authentication" in lower_msg
        or "invalid api key" in lower_msg
        or "[redacted_api_key]" in lower_msg
        or "api key" in lower_msg
    ):
        return (
            "Groq API authentication failed. Please verify that GROQ_API_KEY is correctly "
            "configured in your environment or .env file."
        )
    if "rate limit" in lower_msg or "429" in lower_msg:
        return "Groq API rate limit reached. Please wait a moment before trying again."
    if "connection" in lower_msg or "timeout" in lower_msg or "unreachable" in lower_msg:
        return "Unable to connect to the language model service. Please check your network connection."
    if "sentence-transformers" in lower_msg or "embedding" in lower_msg:
        return "Embedding generation failed. Please verify local embedding model dependencies."

    # Generic clean message without leaking internals
    return f"Unable to process request: {sanitized[:120]}"


@st.cache_resource(show_spinner=False)
def load_embeddings(model_name: str | None = None, root: str | None = None) -> Any:
    """Load and cache the sentence-transformers embedding model resource (Implementationplan 4.2.3).

    Uses Streamlit's @st.cache_resource to initialize the model once and reuse it across
    application reruns and user sessions, preventing expensive repeated model loading.

    Args:
        model_name: Optional model identifier override (e.g. 'all-MiniLM-L6-v2').
        root: Optional project root path override.

    Returns:
        SentenceTransformer: Initialized and cached sentence-transformers model instance.
    """
    proj_root = Path(root).resolve() if root else None
    return emb.load_model(model_name=model_name, root=proj_root)


# Alias for explicit model loading
load_embedding_model = load_embeddings


@st.cache_data(show_spinner=False)
def load_note_embeddings(root: str | None = None) -> dict[str, Any]:
    """Load and cache precomputed note embeddings dictionary from data/embeddings.pkl.

    Args:
        root: Optional project root path override.

    Returns:
        dict[str, np.ndarray]: Mapping of note ID to 1D embedding vector.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    return emb.load_embeddings(root=proj_root)


@st.cache_data(show_spinner=False)
def load_graph(root: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    """Load data/graph.json from disk or persistent SQLite with caching (Implementationplan 4.2.3).

    Uses Streamlit's @st.cache_data to cache parsed graph nodes, edges, and metadata,
    avoiding repeated filesystem reads and JSON deserialization across reruns.
    Automatically restores from SQLite or rebuilds graph if records exist but file is missing.

    Args:
        root: Optional project root path override.
        user_id: Optional user ID to isolate the knowledge graph.

    Returns:
        dict: Parsed graph data containing nodes, edges, and metadata.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    effective_user_id = user_id or DEFAULT_USER_ID
    graph_path = build_graph.get_graph_path(root=proj_root, user_id=effective_user_id)

    # 1. If graph.json exists and is valid, load it
    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "nodes" in data and "edges" in data:
                    if len(data.get("nodes", [])) > 0:
                        return data
                    # If nodes are empty, try auto-restoring from notes
                    try:
                        from lib import startup
                        if startup.ensure_graph_populated(root=proj_root, user_id=effective_user_id):
                            with open(graph_path, "r", encoding="utf-8") as f2:
                                return json.load(f2)
                    except Exception:
                        pass
                    return data
        except Exception as e:
            logger.warning(f"Failed to read graph.json: {e}")

    # 2. Try loading persisted graph records directly from SQLite database
    try:
        from lib import database as db
        db_graph = db.load_graph_records(root=proj_root, user_id=effective_user_id)
        if db_graph and db_graph.get("nodes"):
            return db_graph
    except Exception:
        pass

    # 3. Try automatic rebuilding from persistent notes
    try:
        from lib import startup
        if startup.ensure_graph_populated(root=proj_root, user_id=effective_user_id):
            if graph_path.exists():
                with open(graph_path, "r", encoding="utf-8") as f3:
                    return json.load(f3)
    except Exception:
        pass

    return dict(DEFAULT_GRAPH_DATA)


# Backward compatibility alias for Phase 4.2.2
load_graph_data = load_graph


def get_graph_html(graph_data: dict[str, Any], root: Path | str | None = None) -> str:
    """Load static/graph.html and inline graph data for Streamlit iframe rendering.

    Inlines the graph JSON data into the HTML document to avoid iframe cross-origin
    or relative path fetch errors.

    Args:
        graph_data: Graph dictionary containing nodes, edges, and metadata.
        root: Optional project root path override.

    Returns:
        str: Self-contained HTML string ready for components.html().
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    html_path = proj_root / "static" / "graph.html"

    if not html_path.exists():
        return (
            "<div style='color: #9ca3af; background: #111827; padding: 24px; "
            "border-radius: 8px; font-family: sans-serif; text-align: center;'>"
            "Knowledge graph template not found at <code>static/graph.html</code>.</div>"
        )

    raw_html = html_path.read_text(encoding="utf-8")
    graph_json = json.dumps(graph_data)

    # In static/graph.html, fetch("../data/graph.json") is used to load data.
    # We allow window.__INLINED_GRAPH_DATA__ to be used immediately if defined.
    fetch_target = (
        'const response = await fetch("../data/graph.json", { cache: "no-store" });\n'
        "          if (!response.ok) {\n"
        "            throw new Error(`HTTP ${response.status} (${response.statusText})`);\n"
        "          }\n\n"
        "          const graph = await response.json();"
    )
    fetch_replacement = (
        "const graph = (typeof window.__INLINED_GRAPH_DATA__ !== 'undefined' && window.__INLINED_GRAPH_DATA__ !== null)\n"
        "            ? window.__INLINED_GRAPH_DATA__\n"
        '            : await (await fetch("../data/graph.json", { cache: "no-store" })).json();'
    )

    if fetch_target in raw_html:
        modified_html = raw_html.replace(fetch_target, fetch_replacement, 1)
    else:
        pattern = r'const response = await fetch\("\.\./data/graph\.json".*?const graph = await response\.json\(\);'
        modified_html = re.sub(pattern, fetch_replacement, raw_html, count=1, flags=re.DOTALL)

    # Inject the graph JSON script before </head>
    injected_script = f"<script>window.__INLINED_GRAPH_DATA__ = {graph_json};</script>\n</head>"
    if "</head>" in modified_html:
        modified_html = modified_html.replace("</head>", injected_script, 1)
    else:
        modified_html = f"<script>window.__INLINED_GRAPH_DATA__ = {graph_json};</script>\n" + modified_html

    return modified_html


def get_brain_stats(root: Path | str | None = None, user_id: str | None = None) -> dict[str, Any]:
    """Compute knowledge base statistics for the sidebar display.

    Retrieves counts from persisted notes, raw captures store, and SQLite database.

    Args:
        root: Optional project root path override.
        user_id: Optional user ID to isolate statistics.

    Returns:
        dict: Counts of notes, raw captures, pending captures, graph nodes, edges, and PARA categories.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    effective_user_id = user_id or DEFAULT_USER_ID
    stats: dict[str, Any] = {
        "total_notes": 0,
        "raw_captures": 0,
        "pending_captures": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
        "last_graph_build": None,
        "para_breakdown": {cat: 0 for cat in VALID_CATEGORIES},
    }

    try:
        notes = storage.read_wiki_notes(root=proj_root, user_id=effective_user_id)
        stats["total_notes"] = len(notes)
        for note in notes:
            category = note.para if note.para in stats["para_breakdown"] else "Other"
            stats["para_breakdown"][category] = stats["para_breakdown"].get(category, 0) + 1
    except Exception as e:
        logger.debug(f"Error reading wiki notes for stats: {e}")

    try:
        all_raw = storage.read_all_raw_captures(root=proj_root, user_id=effective_user_id)
        pending_raw = storage.read_raw_captures(root=proj_root, user_id=effective_user_id)
        stats["raw_captures"] = len(all_raw)
        stats["pending_captures"] = len(pending_raw)
    except Exception as e:
        logger.debug(f"Error reading raw captures for stats: {e}")

    try:
        graph = load_graph(root=str(proj_root) if proj_root else None, user_id=effective_user_id)
        metadata = graph.get("metadata", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        stats["graph_nodes"] = metadata.get("node_count", len(nodes))
        stats["graph_edges"] = metadata.get("edge_count", len(edges))
        stats["last_graph_build"] = metadata.get("generated_at")
    except Exception as e:
        logger.debug(f"Error reading graph for stats: {e}")

    # Fallback to SQLite database if filesystem lists are empty or out of sync
    try:
        from lib import database as db
        db_stats = db.get_stats_summary(root=proj_root, user_id=effective_user_id)
        if stats["total_notes"] == 0 and db_stats["total_notes"] > 0:
            stats["total_notes"] = db_stats["total_notes"]
            stats["para_breakdown"] = db_stats["para_breakdown"]
        if stats["raw_captures"] == 0 and db_stats["raw_captures"] > 0:
            stats["raw_captures"] = db_stats["raw_captures"]
        if stats["graph_nodes"] == 0 and db_stats["graph_nodes"] > 0:
            stats["graph_nodes"] = db_stats["graph_nodes"]
            stats["graph_edges"] = db_stats["graph_edges"]
        if not stats["last_graph_build"] and db_stats["last_graph_build"]:
            stats["last_graph_build"] = db_stats["last_graph_build"]
    except Exception:
        pass

    return stats


def render_header(user: User) -> None:
    """Render top application header and Refresh Graph control."""
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.title("🧠 SecondSelf")
        st.caption(f"Personal AI Knowledge Brain — Workspace for **{user.full_name or user.email}**")
    with col_btn:
        st.write("")  # Vertical spacing to align with title
        if st.button("🔄 Refresh Graph", key="btn_refresh_graph", help="Rebuild and reload the knowledge graph"):
            try:
                with st.spinner("Rebuilding knowledge graph..."):
                    build_graph.build_and_save_graph(user_id=user.id)
                load_graph.clear()
                st.session_state["graph_refresh_counter"] = st.session_state.get("graph_refresh_counter", 0) + 1
                st.session_state["refresh_message"] = "Graph refreshed successfully!"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh graph: {sanitize_error_message(e)}")

    if st.session_state.get("refresh_message"):
        st.success(st.session_state["refresh_message"])
        st.session_state["refresh_message"] = None


def render_ask_section(user: User) -> None:
    """Render 'Ask your brain' input controls, answer panel, and source citations."""
    st.markdown("### Ask your brain")

    with st.form(key="ask_form", clear_on_submit=False):
        question_input = st.text_input(
            label="Ask your brain:",
            placeholder="Ask a question about your notes, projects, or saved resources...",
            key="ask_input_field",
            label_visibility="collapsed",
        )
        col_submit, col_hint = st.columns([1, 5])
        with col_submit:
            ask_submitted = st.form_submit_button("Ask", type="primary")
        with col_hint:
            st.caption("Press **Enter** or click **Ask** to query your notes.")

    if ask_submitted:
        clean_question = question_input.strip()
        if not clean_question:
            st.warning("Please enter a question to ask your brain.")
        else:
            with st.spinner("Searching knowledge base and synthesizing answer..."):
                try:
                    # Pass cached embedding model resource to avoid reloading per query (Task 4.2.3)
                    cached_model = load_embeddings()
                    groq_model = get_active_groq_model()
                    result: AskResult = ask.ask(
                        clean_question,
                        model=cached_model,
                        llm_model=groq_model,
                        user_id=user.id,
                    )
                    st.session_state["current_question"] = clean_question
                    st.session_state["current_result"] = result
                    st.session_state["current_error"] = None
                except Exception as exc:
                    logger.error(f"Error during ask execution: {exc}")
                    st.session_state["current_question"] = clean_question
                    st.session_state["current_result"] = None
                    st.session_state["current_error"] = sanitize_error_message(exc)

    # Display Answer and Sources from session state
    if st.session_state.get("current_error"):
        st.error(st.session_state["current_error"])

    elif st.session_state.get("current_result") is not None:
        result: AskResult = st.session_state["current_result"]
        question_asked: str = st.session_state.get("current_question", "")

        is_no_notes = (
            result.answer == ask.NO_NOTES_ANSWER
            or "I don't have enough notes about that" in result.answer
            or "I don't have notes about that" in result.answer
        )

        with st.container(border=True):
            if question_asked:
                st.markdown(f"**Query:** *{question_asked}*")
                st.divider()
            st.markdown("#### Answer")
            if is_no_notes:
                st.info(f"ℹ️ {result.answer}")
                st.caption("No matching notes in your personal knowledge base met the relevance threshold.")
            else:
                st.markdown(result.answer)

        # Source Citations
        if result.sources:
            with st.container(border=True):
                st.markdown(f"#### Sources ({len(result.sources)})")

                # Cache notes lookup for content inspection scoped to current user
                notes_by_id: dict[str, WikiNote] = {}
                try:
                    all_notes = storage.read_wiki_notes(user_id=user.id)
                    notes_by_id = {n.id: n for n in all_notes}
                except Exception:
                    pass

                for src in result.sources:
                    src_id = src.get("id", "unknown")
                    para = src.get("para", "Uncategorized")
                    summary = src.get("summary", "(No summary)")
                    score = src.get("relevance_score", 0.0)

                    st.markdown(
                        f"- **`[{src_id}]`** `({para})` {summary} &nbsp;*(relevance: `{score:.4f}`)*"
                    )

                    # Inspectable note expander if note exists in wiki
                    if src_id in notes_by_id:
                        note_obj = notes_by_id[src_id]
                        with st.expander(f"Inspect [{src_id}] note content", expanded=False):
                            if note_obj.tags:
                                clean_tags = [str(t) for t in note_obj.tags if str(t).strip()]
                                if clean_tags:
                                    st.caption(f"Tags: {', '.join(clean_tags)}")
                            st.markdown(note_obj.body or "(Empty content)")
                            if st.button(f"✏️ Edit [{src_id}] in Editor", key=f"btn_edit_src_{src_id}"):
                                st.session_state["editing_note_id"] = src_id
                                st.session_state["sidebar_note_select"] = src_id
                                st.rerun()
        elif not is_no_notes:
            st.caption("No sources were cited for this response.")


def render_graph_section(user: User) -> None:
    """Render the Interactive Knowledge Graph (vis-network) section."""
    st.divider()
    st.subheader("Interactive Knowledge Graph")
    st.caption("Explore relationships across your notes. Hover to preview content, drag to reposition, and scroll to zoom.")

    graph_data = load_graph(user_id=user.id)
    counter = st.session_state.get("graph_refresh_counter", 0)
    graph_html = get_graph_html(graph_data)
    # Append dynamic version tag to ensure iframe reloads on refresh
    versioned_html = f"<!-- refresh_version: {counter} -->\n{graph_html}"

    components.html(versioned_html, height=550, scrolling=False)


def save_edited_note(
    note_id: str,
    new_para: str,
    new_summary: str,
    new_tags: list[str],
    new_body: str,
    root: Path | str | None = None,
    user_id: str | None = None,
) -> WikiNote:
    """Save modifications to an existing wiki note and refresh embeddings & graph.

    Args:
        note_id: Unique note identifier (e.g. 'cffaf460').
        new_para: PARA category ('Projects', 'Areas', 'Resources', 'Archives').
        new_summary: Single-line human readable summary / title.
        new_tags: List of semantic tags.
        new_body: Markdown body of the note.
        root: Optional project root path override.
        user_id: Optional user ID to isolate note updates.

    Returns:
        WikiNote: Updated and persisted note instance.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    effective_user_id = user_id or DEFAULT_USER_ID
    notes = storage.read_wiki_notes(root=proj_root, user_id=effective_user_id)
    existing = next((n for n in notes if n.id == note_id), None)
    if not existing:
        raise ValueError(f"Note '{note_id}' not found in knowledge base.")

    old_para = existing.para

    updated_note = WikiNote(
        id=existing.id,
        raw_id=existing.raw_id,
        para=new_para,
        tags=new_tags,
        summary=new_summary.strip() or existing.summary,
        created=existing.created,
        links=existing.links,
        body=new_body,
        user_id=effective_user_id,
    )

    # If PARA category changed, remove the old file from the previous category directory
    if old_para != new_para:
        old_file = storage.get_wiki_dir(proj_root) / old_para / f"{note_id}.md"
        if old_file.exists():
            try:
                old_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove old file {old_file}: {e}")

    # Persist updated note with frontmatter
    storage.write_wiki_note(updated_note, root=proj_root)

    # Re-embed updated note
    try:
        content_to_embed = f"{updated_note.summary}\n\n{updated_note.body}".strip()
        if content_to_embed:
            vec = emb.embed_text(content_to_embed, root=proj_root)
            all_embeddings = emb.load_embeddings(root=proj_root)
            all_embeddings[updated_note.id] = vec
            emb.save_embeddings(all_embeddings, root=proj_root)
    except Exception as e:
        logger.warning(f"Failed to update embedding for note {note_id}: {e}")

    # Rebuild graph for user
    try:
        build_graph.build_and_save_graph(root=proj_root, user_id=effective_user_id)
    except Exception as e:
        logger.warning(f"Failed to rebuild graph after editing note {note_id}: {e}")

    # Clear application caches so changes reflect immediately
    load_graph.clear()
    load_note_embeddings.clear()
    load_embeddings.clear()

    return updated_note


def delete_wiki_note(
    note_id: str,
    root: Path | str | None = None,
    user_id: str | None = None,
) -> bool:
    """Delete a wiki note from disk, remove its embedding, and rebuild graph.

    Args:
        note_id: Identifier of the note to delete.
        root: Optional project root path override.
        user_id: Optional user ID to isolate note deletion.

    Returns:
        bool: True if note was deleted, False if note was not found.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    effective_user_id = user_id or DEFAULT_USER_ID
    deleted = storage.delete_wiki_note(note_id, user_id=effective_user_id, root=proj_root)
    if not deleted:
        return False

    # Remove from embeddings.pkl
    try:
        embeddings = emb.load_embeddings(root=proj_root)
        if note_id in embeddings:
            del embeddings[note_id]
            emb.save_embeddings(embeddings, root=proj_root)
    except Exception as e:
        logger.warning(f"Failed to remove embedding for note {note_id}: {e}")

    # Rebuild graph for user
    try:
        build_graph.build_and_save_graph(root=proj_root, user_id=effective_user_id)
    except Exception as e:
        logger.warning(f"Failed to rebuild graph after deleting note {note_id}: {e}")

    # Clear application caches
    load_graph.clear()
    load_note_embeddings.clear()
    load_embeddings.clear()

    return True


def capture_uploaded_file(
    uploaded_file: Any,
    root: Path | str | None = None,
    user_id: str | None = None,
) -> CaptureResult:
    """Capture an uploaded file from Streamlit into raw/ capture store.

    Preserves original file bytes, records metadata with content hash,
    and stores the file in raw/{capture_id}/content.{ext}.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        root: Optional project root path override.
        user_id: Optional user ID who owns the uploaded file.

    Returns:
        CaptureResult: Result dataclass describing the capture.

    Raises:
        ValueError: If uploaded_file is None or empty.
    """
    if uploaded_file is None:
        raise ValueError("No file provided for capture.")

    effective_user_id = user_id or DEFAULT_USER_ID
    file_name = getattr(uploaded_file, "name", "uploaded_file.bin")
    orig_name = Path(file_name).name
    content_bytes = (
        uploaded_file.getvalue()
        if hasattr(uploaded_file, "getvalue")
        else uploaded_file.read()
    )

    if not content_bytes:
        raise ValueError(f"Uploaded file '{orig_name}' is empty.")

    capture_id = storage.generate_capture_id()
    now_iso = datetime.now(timezone.utc).isoformat()
    c_hash = storage.content_hash(content_bytes)

    existing_id = capture.find_duplicate_capture(c_hash, user_id=effective_user_id)
    if existing_id:
        logger.warning(
            f"Content with hash {c_hash} was already captured at raw/{existing_id}"
        )

    meta = CaptureMeta(
        id=capture_id,
        timestamp=now_iso,
        type="file",
        source=f"upload:{orig_name}",
        original_filename=orig_name,
        content_hash=c_hash,
        user_id=effective_user_id,
    )

    try:
        saved_path = storage.write_raw_capture(meta, content_bytes, root=root)
    except Exception:
        raw_dir = storage.get_raw_dir(root)
        partial_dir = raw_dir / capture_id
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    return CaptureResult(
        id=capture_id,
        path=str(saved_path),
        type="file",
    )


def render_sidebar(user: User) -> None:
    """Render the Sidebar containing User Profile, Navigation, Capture, Process, and Stats."""
    with st.sidebar:
        # Custom CSS for modern tab appearance matching SecondSelf design
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] [data-baseweb="tab-list"] {
                gap: 12px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 12px;
            }
            [data-testid="stSidebar"] [data-baseweb="tab"] {
                padding: 6px 10px;
                font-size: 14px;
                font-weight: 500;
                color: #9ca3af;
                background-color: transparent;
                border: none;
            }
            [data-testid="stSidebar"] [data-baseweb="tab"][aria-selected="true"] {
                color: #a855f7 !important;
                font-weight: 600;
            }
            [data-testid="stSidebar"] [data-baseweb="tab-highlight"] {
                background-color: #a855f7 !important;
                height: 2px;
            }
            [data-testid="stSidebar"] [data-baseweb="tab-border"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 0.2rem; margin-bottom: 0.8rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-brain" style="flex-shrink: 0;">
                    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
                    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
                    <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
                    <path d="M12 18v4"/>
                    <path d="M8 22h8"/>
                </svg>
                <h2 id="brain-controls" style="margin: 0; padding: 0; font-size: 1.45rem; font-weight: 700; color: #f9fafb; letter-spacing: -0.01em; line-height: 1.2;">Brain Controls</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Section 1: Capture (Tabbed: Note, Link, File)
        st.subheader("📥 Capture")
        tab_note, tab_link, tab_file = st.tabs(["Note", "Link", "File"])

        with tab_note:
            with st.form(key="note_capture_form", clear_on_submit=True):
                note_text = st.text_area(
                    "Note Content",
                    placeholder="Type your note here...",
                    height=120,
                    label_visibility="collapsed",
                    help="Captures raw note text to the raw/ captures store.",
                )
                capture_note_btn = st.form_submit_button("Capture Note", use_container_width=True)

            if capture_note_btn:
                clean_note = note_text.strip()
                if not clean_note:
                    st.warning("Note text cannot be empty.")
                else:
                    try:
                        if user.id == DEFAULT_USER_ID:
                            cap_res: CaptureResult = capture.capture_note(clean_note)
                        else:
                            cap_res: CaptureResult = capture.capture_note(clean_note, user_id=user.id)
                        st.session_state["capture_message"] = f"Captured note to `raw/{cap_res.id}`"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Capture failed: {sanitize_error_message(e)}")

        with tab_link:
            with st.form(key="link_capture_form", clear_on_submit=True):
                link_url = st.text_input(
                    "URL",
                    placeholder="https://...",
                    label_visibility="collapsed",
                    help="Web link or bookmark to capture.",
                )
                link_context = st.text_input(
                    "Context",
                    placeholder="Brief context...",
                    label_visibility="collapsed",
                    help="Optional context or annotation for this link.",
                )
                capture_link_btn = st.form_submit_button("Capture Link", use_container_width=True)

            if capture_link_btn:
                clean_url = link_url.strip()
                clean_context = link_context.strip()
                if not clean_url:
                    st.warning("URL cannot be empty.")
                else:
                    try:
                        cap_res: CaptureResult = capture.capture_link(
                            clean_url, notes=clean_context, user_id=user.id
                        )
                        st.session_state["capture_message"] = f"Captured link to `raw/{cap_res.id}`"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Capture failed: {sanitize_error_message(e)}")

        with tab_file:
            with st.form(key="file_capture_form", clear_on_submit=True):
                uploaded_file = st.file_uploader(
                    "File Upload",
                    type=["pdf", "txt", "md", "csv", "json", "py", "doc", "docx"],
                    help="200MB per file",
                    label_visibility="collapsed",
                )
                capture_file_btn = st.form_submit_button("Capture File", use_container_width=True)

            if capture_file_btn:
                if uploaded_file is None:
                    st.warning("Please select a file to upload.")
                else:
                    try:
                        cap_res = capture_uploaded_file(uploaded_file, user_id=user.id)
                        st.session_state["capture_message"] = f"Captured file `{uploaded_file.name}` to `raw/{cap_res.id}`"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Capture failed: {sanitize_error_message(e)}")

        if st.session_state.get("capture_message"):
            st.success(st.session_state["capture_message"])
            st.session_state["capture_message"] = None

        st.divider()

        # Section 2: Process
        st.subheader("⚙️ Process")
        st.caption("Classify raw captures, link semantic notes, and update graph.")
        if st.button("🚀 Process New Captures", key="btn_run_pipeline"):
            try:
                with st.spinner("Running classification & linking pipeline..."):
                    groq_model = get_active_groq_model()
                    report = pipeline.process(llm_model=groq_model, user_id=user.id)
                load_graph.clear()
                load_embeddings.clear()
                load_note_embeddings.clear()
                st.session_state["graph_refresh_counter"] = st.session_state.get("graph_refresh_counter", 0) + 1
                st.session_state["process_message"] = (
                    f"Pipeline completed: {report.captures_classified} classified, "
                    f"{report.links_created} linked. Graph nodes: {report.graph_node_count}."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {sanitize_error_message(e)}")

        if st.session_state.get("process_message"):
            st.success(st.session_state["process_message"])
            st.session_state["process_message"] = None

        st.divider()

        # Section 3: Stats
        st.subheader("📊 Stats")
        stats = get_brain_stats(user_id=user.id)
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total Notes", stats["total_notes"])
            st.metric("Graph Nodes", stats["graph_nodes"])
        with col_stat2:
            st.metric("Raw Captures", stats["raw_captures"])
            st.metric("Connections", stats["graph_edges"])

        if stats.get("pending_captures", 0) > 0:
            st.caption(f"⏳ **{stats['pending_captures']}** capture(s) pending processing")

        if stats.get("last_graph_build"):
            ts_clean = str(stats["last_graph_build"])[:19].replace("T", " ")
            st.caption(f"Last Graph Build: {ts_clean}")

        if stats.get("para_breakdown"):
            with st.expander("Notes by Category"):
                for para_cat, count in stats["para_breakdown"].items():
                    icon = CATEGORY_DEFINITIONS.get(para_cat, {}).get("icon", "📌")
                    st.write(f"- {icon} **{para_cat}**: {count}")

        st.divider()

        # Section 4: Note Editor & Delete (down side of Notes by Category)
        st.subheader("✏️ Note Editor")
        all_notes = storage.read_wiki_notes(user_id=user.id)
        if not all_notes:
            st.caption("No notes in knowledge base yet.")
        else:
            all_notes.sort(key=lambda n: (n.para, n.summary))
            notes_by_id = {n.id: n for n in all_notes}
            note_options = list(notes_by_id.keys())

            current_edit_id = st.session_state.get("editing_note_id")
            default_idx = (
                note_options.index(current_edit_id)
                if (current_edit_id and current_edit_id in note_options)
                else None
            )

            selected_note_id = st.selectbox(
                "Select Note:",
                options=note_options,
                index=default_idx,
                placeholder="Choose a note to edit...",
                format_func=lambda nid: (
                    f"[{nid}] ({CATEGORY_DEFINITIONS.get(notes_by_id[nid].para, {}).get('icon', '📌')} "
                    f"{notes_by_id[nid].para}) {notes_by_id[nid].summary[:35]}"
                ),
                key="sidebar_note_select",
            )

            if not selected_note_id or selected_note_id not in notes_by_id:
                st.session_state["editing_note_id"] = None
                st.caption("Select a note from the dropdown above to view or edit its details.")
            else:
                note = notes_by_id[selected_note_id]
                st.session_state["editing_note_id"] = selected_note_id

                valid_paras = list(VALID_CATEGORIES)
                current_para_idx = valid_paras.index(note.para) if note.para in valid_paras else 0
                new_para = st.selectbox(
                    "Category",
                    valid_paras,
                    index=current_para_idx,
                    format_func=lambda c: f"{CATEGORY_DEFINITIONS.get(c, {}).get('icon', '📌')} {c}",
                    key=f"sb_para_{note.id}",
                )

                new_summary = st.text_input(
                    "Title / Summary",
                    value=note.summary,
                    key=f"sb_sum_{note.id}",
                )

                tags_str = ", ".join(note.tags) if note.tags else ""
                new_tags_str = st.text_input(
                    "Tags (comma-separated)",
                    value=tags_str,
                    key=f"sb_tags_{note.id}",
                )

                new_body = st.text_area(
                    "Content",
                    value=note.body,
                    height=140,
                    key=f"sb_body_{note.id}",
                )

                col_save, col_del = st.columns(2)
                with col_save:
                    if st.button("💾 Save", type="primary", use_container_width=True, key=f"btn_save_sb_{note.id}"):
                        clean_summary = new_summary.strip()
                        if not clean_summary:
                            st.warning("Title cannot be empty.")
                        else:
                            clean_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]
                            try:
                                save_edited_note(
                                    note_id=note.id,
                                    new_para=new_para,
                                    new_summary=clean_summary,
                                    new_tags=clean_tags,
                                    new_body=new_body,
                                    user_id=user.id,
                                )
                                st.session_state["graph_refresh_counter"] = st.session_state.get("graph_refresh_counter", 0) + 1
                                st.session_state["sidebar_note_msg"] = f"✅ Saved [{note.id}]"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Save failed: {sanitize_error_message(e)}")

                with col_del:
                    if st.button("🗑️ Delete", type="secondary", use_container_width=True, key=f"btn_del_sb_{note.id}"):
                        st.session_state[f"confirm_delete_{note.id}"] = True

                if st.session_state.get(f"confirm_delete_{note.id}"):
                    st.warning(f"Delete note `[{note.id}]` permanently?")
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        if st.button("Confirm", type="primary", use_container_width=True, key=f"btn_confirm_del_{note.id}"):
                            delete_wiki_note(note.id, user_id=user.id)
                            st.session_state[f"confirm_delete_{note.id}"] = False
                            st.session_state["editing_note_id"] = None
                            st.session_state["sidebar_note_select"] = None
                            st.session_state["graph_refresh_counter"] = st.session_state.get("graph_refresh_counter", 0) + 1
                            st.session_state["sidebar_note_msg"] = f"🗑️ Deleted [{note.id}]"
                            st.rerun()
                    with col_c2:
                        if st.button("Cancel", use_container_width=True, key=f"btn_cancel_del_{note.id}"):
                            st.session_state[f"confirm_delete_{note.id}"] = False
                            st.rerun()

        if st.session_state.get("sidebar_note_msg"):
            st.success(st.session_state["sidebar_note_msg"])
            st.session_state["sidebar_note_msg"] = None

        # User Authentication & Workspace Profile Footer
        st.divider()
        with st.container(border=True):
            user_display = user.full_name or user.email.split("@")[0]
            st.markdown(f"**👤 {user_display}**")
            role_label = "🛡️ Administrator" if user.is_admin else "👤 Member"
            st.caption(f"{user.email} &bull; `{role_label}`")

            col_u1, col_u2 = st.columns(2)
            with col_u1:
                if user.is_admin:
                    in_admin_view = st.session_state.get("view_mode") == "admin"
                    nav_label = "🧠 Second Brain" if in_admin_view else "🛡️ Admin Portal"
                    if st.button(nav_label, key="btn_nav_toggle", use_container_width=True):
                        st.session_state["view_mode"] = "user" if in_admin_view else "admin"
                        st.rerun()
            with col_u2:
                if st.button("🚪 Logout", key="btn_logout", use_container_width=True):
                    auth.logout_user(st.session_state.get("session_token"))
                    st.session_state["authenticated_user"] = None
                    st.session_state["session_token"] = None
                    st.session_state["view_mode"] = "user"
                    st.rerun()


def render_auth_page() -> None:
    """Render the SecondSelf authentication gateway (Login, Register, Forgot Password)."""
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 40px; margin-bottom: 24px;">
                <h1 style="font-size: 2.6rem; font-weight: 700; margin-bottom: 4px;">🧠 SecondSelf</h1>
                <p style="color: #c084fc; font-size: 1.1rem; font-weight: 500;">Your Personal AI Second Brain</p>
                <p style="color: #9ca3af; font-size: 0.9rem;">Secure Multi-User Personal Knowledge Management</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register, tab_reset = st.tabs(["🔐 Sign In", "✨ Create Account", "🔑 Forgot Password"])

        with tab_login:
            with st.form(key="form_user_login", clear_on_submit=False):
                email = st.text_input("Email", placeholder="you@domain.com", key="auth_login_email")
                password = st.text_input("Password", type="password", placeholder="••••••••", key="auth_login_password")
                btn_login = st.form_submit_button("Sign In to SecondSelf", type="primary", use_container_width=True)

            if btn_login:
                ok, msg, session, user = auth.login_user(email, password)
                if ok and session and user:
                    st.session_state["authenticated_user"] = user
                    st.session_state["session_token"] = session.token
                    st.session_state["view_mode"] = "user"
                    st.rerun()
                else:
                    st.error(msg)

        with tab_register:
            with st.form(key="form_user_register", clear_on_submit=False):
                full_name = st.text_input("Full Name", placeholder="Jane Doe", key="auth_reg_name")
                reg_email = st.text_input("Email Address", placeholder="you@domain.com", key="auth_reg_email")
                reg_pwd = st.text_input("Password", type="password", placeholder="At least 8 characters", key="auth_reg_pwd")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="auth_reg_confirm")
                btn_register = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if btn_register:
                if reg_pwd != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg, new_user = auth.register_user(reg_email, reg_pwd, full_name=full_name)
                    if ok and new_user:
                        st.success("Account registered successfully! Logging you in...")
                        log_ok, log_msg, session, user = auth.login_user(reg_email, reg_pwd)
                        if log_ok and session and user:
                            st.session_state["authenticated_user"] = user
                            st.session_state["session_token"] = session.token
                            st.session_state["view_mode"] = "user"
                            st.rerun()
                    else:
                        st.error(msg)

        with tab_reset:
            st.markdown("##### Request Password Reset Token")
            with st.form(key="form_user_req_reset", clear_on_submit=False):
                reset_email = st.text_input("Account Email", placeholder="you@domain.com", key="auth_reset_email")
                btn_req_token = st.form_submit_button("Generate Reset Token", use_container_width=True)

            if btn_req_token:
                ok, msg, token = auth.request_password_reset(reset_email)
                if ok and token:
                    st.success("Reset token generated successfully! Copy the token below:")
                    st.code(token, language="text")
                else:
                    st.info(msg)

            st.markdown("##### Reset Password with Token")
            with st.form(key="form_user_perform_reset", clear_on_submit=False):
                tok_input = st.text_input("Reset Token", key="auth_tok_input")
                new_pwd = st.text_input("New Password", type="password", placeholder="At least 8 characters", key="auth_new_pwd")
                new_confirm = st.text_input("Confirm New Password", type="password", placeholder="••••••••", key="auth_confirm_new_pwd")
                btn_reset_pwd = st.form_submit_button("Reset Password", type="primary", use_container_width=True)

            if btn_reset_pwd:
                if new_pwd != new_confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = auth.reset_password(tok_input, new_pwd)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


def render_admin_portal(current_user: User) -> None:
    """Render the Administrator Portal for platform metrics and account management."""
    if not auth.verify_admin(current_user):
        st.error("⛔ Access Denied: Administrator privileges required.")
        return

    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.title("🛡️ Administrator Portal")
        st.caption("Platform Administration, User Account Management & System Statistics")
    with col_h2:
        st.write("")
        if st.button("← Back to Second Brain", key="btn_back_to_brain", type="primary"):
            st.session_state["view_mode"] = "user"
            st.rerun()

    st.divider()

    # Section 1: Platform Metrics
    st.subheader("Platform Metrics")
    sys_stats = get_all_system_stats()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Total Users", sys_stats.get("total_users", 0))
    with m2:
        st.metric("Active Users", sys_stats.get("active_users", 0))
    with m3:
        st.metric("Verified Users", sys_stats.get("verified_users", 0))
    with m4:
        st.metric("Total Notes", sys_stats.get("total_notes", 0))
    with m5:
        st.metric("Total Captures", sys_stats.get("total_captures", 0))
    with m6:
        st.metric("Total Edges", sys_stats.get("total_edges", 0))

    st.divider()

    # Section 2: User Account Management
    st.subheader("Registered Users Management")
    users = auth.admin_list_users(current_user)

    for u in users:
        uid = u["id"]
        is_self = (uid == current_user.id)
        with st.container(border=True):
            col_info, col_actions = st.columns([3, 2])
            with col_info:
                badge = "🛡️ Admin" if u["role"] == "admin" else "👤 User"
                status_color = "🟢 Active" if u["is_active"] else "🔴 Deactivated"
                self_tag = " *(You)*" if is_self else ""
                name_display = u["full_name"] or "Unnamed User"
                st.markdown(f"**{name_display}** (`{u['email']}`){self_tag}")
                st.caption(
                    f"Role: **{badge}** &bull; Status: **{status_color}** &bull; "
                    f"Notes: **{u.get('note_count', 0)}** &bull; Captures: **{u.get('capture_count', 0)}**"
                )
                created_str = str(u.get("created_at", ""))[:19].replace("T", " ")
                last_login_str = str(u.get("last_login_at") or "Never")[:19].replace("T", " ")
                st.caption(f"Registered: `{created_str}` | Last Login: `{last_login_str}`")

            with col_actions:
                if not is_self:
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if u["is_active"]:
                            if st.button("Deactivate", key=f"btn_deact_{uid}", help="Disable user login"):
                                ok, msg = auth.admin_set_user_status(current_user, uid, False)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        else:
                            if st.button("Activate", key=f"btn_act_{uid}", help="Enable user login"):
                                ok, msg = auth.admin_set_user_status(current_user, uid, True)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                    with col_act2:
                        if u["role"] == "user":
                            if st.button("Make Admin", key=f"btn_mkadmin_{uid}", help="Promote to Admin"):
                                ok, msg = auth.admin_set_user_role(current_user, uid, "admin")
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        else:
                            if st.button("Demote", key=f"btn_demote_{uid}", help="Demote to normal User"):
                                ok, msg = auth.admin_set_user_role(current_user, uid, "user")
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                else:
                    st.caption("Active Session")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="SecondSelf",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Ensure durable startup recovery and persistent data restoration
    try:
        from lib import startup
        startup.ensure_data_restored()
    except Exception as e:
        logger.warning(f"Startup restoration encountered non-fatal error: {e}")

    # Initialize session state keys
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = ""
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None
    if "current_error" not in st.session_state:
        st.session_state["current_error"] = None
    if "graph_refresh_counter" not in st.session_state:
        st.session_state["graph_refresh_counter"] = 0
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "user"

    # Validate active session token against database
    current_user: User | None = st.session_state.get("authenticated_user")
    token = st.session_state.get("session_token")
    if token:
        db_user = auth.authenticate_session(token)
        if db_user:
            current_user = db_user
            st.session_state["authenticated_user"] = db_user
        else:
            current_user = None
            st.session_state["authenticated_user"] = None
            st.session_state["session_token"] = None

    # Gatekeeper: If unauthenticated, check test environment or display authentication gateway and halt
    if not current_user:
        if os.environ.get("SECONDSELF_AUTO_LOGIN_ADMIN") == "1":
            admin_user = db.get_user_by_id(db.DEFAULT_USER_ID)
            if admin_user:
                current_user = admin_user
                st.session_state["authenticated_user"] = admin_user
        if not current_user:
            render_auth_page()
            return

    # If Admin Portal mode requested and user is server-side verified admin
    if st.session_state.get("view_mode") == "admin" and auth.verify_admin(current_user):
        render_sidebar(current_user)
        render_admin_portal(current_user)
        return

    # Normal authenticated user dashboard
    render_header(current_user)
    render_ask_section(current_user)
    render_graph_section(current_user)
    render_sidebar(current_user)


if __name__ == "__main__":
    main()
