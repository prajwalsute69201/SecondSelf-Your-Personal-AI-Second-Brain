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
import lib.embeddings as emb
from lib.models import AskResult, CaptureMeta, CaptureResult, WikiNote
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
def load_graph(root: str | None = None) -> dict[str, Any]:
    """Load data/graph.json from disk with caching (Implementationplan 4.2.3).

    Uses Streamlit's @st.cache_data to cache parsed graph nodes, edges, and metadata,
    avoiding repeated filesystem reads and JSON deserialization across reruns.

    Args:
        root: Optional project root path override.

    Returns:
        dict: Parsed graph data containing nodes, edges, and metadata.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    graph_path = build_graph.get_graph_path(root=proj_root)

    if not graph_path.exists():
        return dict(DEFAULT_GRAPH_DATA)

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "nodes" in data and "edges" in data:
                return data
    except Exception as e:
        logger.warning(f"Failed to read graph.json: {e}")

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


def get_brain_stats(root: Path | str | None = None) -> dict[str, Any]:
    """Compute knowledge base statistics for the sidebar display.

    Args:
        root: Optional project root path override.

    Returns:
        dict: Counts of notes, raw captures, graph nodes, edges, and PARA categories.
    """
    stats: dict[str, Any] = {
        "total_notes": 0,
        "raw_captures": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
        "last_graph_build": None,
        "para_breakdown": {"Projects": 0, "Areas": 0, "Resources": 0, "Archives": 0},
    }

    try:
        notes = storage.read_wiki_notes(root=root)
        stats["total_notes"] = len(notes)
        for note in notes:
            category = note.para if note.para in stats["para_breakdown"] else "Other"
            stats["para_breakdown"][category] = stats["para_breakdown"].get(category, 0) + 1
    except Exception as e:
        logger.debug(f"Error reading wiki notes for stats: {e}")

    try:
        raw_items = storage.read_raw_captures(root=root)
        stats["raw_captures"] = len(raw_items)
    except Exception as e:
        logger.debug(f"Error reading raw captures for stats: {e}")

    try:
        graph = load_graph(root=str(root) if root else None)
        metadata = graph.get("metadata", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        stats["graph_nodes"] = metadata.get("node_count", len(nodes))
        stats["graph_edges"] = metadata.get("edge_count", len(edges))
        stats["last_graph_build"] = metadata.get("generated_at")
    except Exception as e:
        logger.debug(f"Error reading graph for stats: {e}")

    return stats


def render_header() -> None:
    """Render top application header and Refresh Graph control."""
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.title("🧠 SecondSelf")
        st.caption("Personal AI Knowledge Brain — Grounded RAG & Semantic Graph")
    with col_btn:
        st.write("")  # Vertical spacing to align with title
        if st.button("🔄 Refresh Graph", key="btn_refresh_graph", help="Rebuild and reload the knowledge graph"):
            try:
                with st.spinner("Rebuilding knowledge graph..."):
                    build_graph.build_and_save_graph()
                load_graph.clear()
                st.session_state["graph_refresh_counter"] = st.session_state.get("graph_refresh_counter", 0) + 1
                st.session_state["refresh_message"] = "Graph refreshed successfully!"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh graph: {sanitize_error_message(e)}")

    if st.session_state.get("refresh_message"):
        st.success(st.session_state["refresh_message"])
        st.session_state["refresh_message"] = None


def render_ask_section() -> None:
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
                    result: AskResult = ask.ask(clean_question, model=cached_model, llm_model=groq_model)
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

                # Cache notes lookup for content inspection
                notes_by_id: dict[str, WikiNote] = {}
                try:
                    all_notes = storage.read_wiki_notes()
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
                                st.rerun()
        elif not is_no_notes:
            st.caption("No sources were cited for this response.")


def render_graph_section() -> None:
    """Render the Interactive Knowledge Graph (vis-network) section."""
    st.divider()
    st.subheader("Interactive Knowledge Graph")
    st.caption("Explore relationships across your notes. Hover to preview content, drag to reposition, and scroll to zoom.")

    graph_data = load_graph()
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
) -> WikiNote:
    """Save modifications to an existing wiki note and refresh embeddings & graph.

    Args:
        note_id: Unique note identifier (e.g. 'cffaf460').
        new_para: PARA category ('Projects', 'Areas', 'Resources', 'Archives').
        new_summary: Single-line human readable summary / title.
        new_tags: List of semantic tags.
        new_body: Markdown body of the note.
        root: Optional project root path override.

    Returns:
        WikiNote: Updated and persisted note instance.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    notes = storage.read_wiki_notes(root=proj_root)
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

    # Rebuild graph
    try:
        build_graph.build_and_save_graph(root=proj_root)
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
) -> bool:
    """Delete a wiki note from disk, remove its embedding, and rebuild graph.

    Args:
        note_id: Identifier of the note to delete.
        root: Optional project root path override.

    Returns:
        bool: True if note was deleted, False if note was not found.
    """
    proj_root = Path(root).resolve() if root else storage.get_project_root()
    notes = storage.read_wiki_notes(root=proj_root)
    target = next((n for n in notes if n.id == note_id), None)
    if not target:
        return False

    # Remove markdown file from wiki/{para}/{id}.md
    note_file = storage.get_wiki_dir(proj_root) / target.para / f"{note_id}.md"
    if note_file.exists():
        try:
            note_file.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete note file {note_file}: {e}")

    # Remove from embeddings.pkl
    try:
        embeddings = emb.load_embeddings(root=proj_root)
        if note_id in embeddings:
            del embeddings[note_id]
            emb.save_embeddings(embeddings, root=proj_root)
    except Exception as e:
        logger.warning(f"Failed to remove embedding for note {note_id}: {e}")

    # Rebuild graph
    try:
        build_graph.build_and_save_graph(root=proj_root)
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
) -> CaptureResult:
    """Capture an uploaded file from Streamlit into raw/ capture store.

    Preserves original file bytes, records metadata with content hash,
    and stores the file in raw/{capture_id}/content.{ext}.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        root: Optional project root path override.

    Returns:
        CaptureResult: Result dataclass describing the capture.

    Raises:
        ValueError: If uploaded_file is None or empty.
    """
    if uploaded_file is None:
        raise ValueError("No file provided for capture.")

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

    existing_id = capture.find_duplicate_capture(c_hash)
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


def render_sidebar() -> None:
    """Render the Sidebar containing Capture, Process, and Stats sections."""
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

        st.header("Brain Controls")
        st.divider()

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
                        cap_res: CaptureResult = capture.capture_note(clean_note)
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
                        cap_res: CaptureResult = capture.capture_link(clean_url, notes=clean_context)
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
                        cap_res = capture_uploaded_file(uploaded_file)
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
                    report = pipeline.process(llm_model=groq_model)
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
        stats = get_brain_stats()
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total Notes", stats["total_notes"])
            st.metric("Graph Nodes", stats["graph_nodes"])
        with col_stat2:
            st.metric("Raw Captures", stats["raw_captures"])
            st.metric("Connections", stats["graph_edges"])

        if stats.get("last_graph_build"):
            ts_clean = str(stats["last_graph_build"])[:19].replace("T", " ")
            st.caption(f"Last Graph Build: {ts_clean}")

        if stats.get("para_breakdown"):
            with st.expander("Notes by PARA"):
                for para_cat, count in stats["para_breakdown"].items():
                    st.write(f"- **{para_cat}**: {count}")

        st.divider()

        # Section 4: Note Editor & Delete (down side of Notes by PARA)
        st.subheader("✏️ Note Editor")
        all_notes = storage.read_wiki_notes()
        if not all_notes:
            st.caption("No notes in knowledge base yet.")
        else:
            all_notes.sort(key=lambda n: (n.para, n.summary))
            notes_by_id = {n.id: n for n in all_notes}
            note_options = list(notes_by_id.keys())

            current_edit_id = st.session_state.get("editing_note_id")
            default_idx = note_options.index(current_edit_id) if current_edit_id in note_options else 0

            selected_note_id = st.selectbox(
                "Select Note:",
                options=note_options,
                index=default_idx,
                format_func=lambda nid: f"[{nid}] ({notes_by_id[nid].para}) {notes_by_id[nid].summary[:35]}",
                key="sidebar_note_select",
            )
            note = notes_by_id[selected_note_id]

            valid_paras = ["Projects", "Areas", "Resources", "Archives"]
            current_para_idx = valid_paras.index(note.para) if note.para in valid_paras else 0
            new_para = st.selectbox(
                "Category",
                valid_paras,
                index=current_para_idx,
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
                        delete_wiki_note(note.id)
                        st.session_state[f"confirm_delete_{note.id}"] = False
                        st.session_state["editing_note_id"] = None
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


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="SecondSelf",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state keys
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = ""
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None
    if "current_error" not in st.session_state:
        st.session_state["current_error"] = None
    if "graph_refresh_counter" not in st.session_state:
        st.session_state["graph_refresh_counter"] = 0

    # Render main UI sections according to task 4.2.1/4.2.2 layout
    render_header()
    render_ask_section()
    render_graph_section()
    render_sidebar()


if __name__ == "__main__":
    main()
