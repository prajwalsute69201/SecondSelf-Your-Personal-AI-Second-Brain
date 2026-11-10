# SecondSelf — Detailed System Architecture

Architecture for a **personal knowledge brain** that captures anything, self-organizes with AI, visualizes as a graph, and answers questions via RAG — aligned with the 4-week build in `PROBLEM_STATEMENT.md`.

---

## 1. System Overview

SecondSelf is a **local-first, pipeline-driven knowledge system** with a thin web UI for exploration and Q&A. There is no traditional database in v1 — the filesystem (`raw/` + `wiki/`) is the source of truth.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SecondSelf System                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────────┐  │
│  │ Capture  │───▶│ Classify +  │───▶│  Graph   │───▶│  Ask (RAG)  │  │
│  │ Pipeline │    │   Link      │    │  Builder │    │   Engine    │  │
│  └──────────┘    └─────────────┘    └──────────┘    └─────────────┘  │
│       │                 │                  │                 │         │
│       ▼                 ▼                  ▼                 ▼         │
│    raw/              wiki/            graph.json          answers      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              Streamlit App (Week 4 — public deployment)           │  │
│  │   [ Interactive Graph ]  +  [ Ask-Anything Search Bar ]          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Filesystem as database** | Simple, inspectable, git-friendly; no DB setup for v1 |
| **Idempotent pipelines** | Re-run classify/link without duplicating work |
| **Local embeddings, cloud LLM** | Free local vectors; cheap/free LLM for classification & synthesis |
| **Markdown everywhere** | Human-readable wiki notes; links are standard `[[note-id]]` wikilinks |
| **Progressive enhancement** | Each week adds a layer; prior weeks keep working standalone |

---

## 2. Repository Layout

```
secondself/
├── raw/                          # Immutable captures (append-only)
│   └── {YYYY-MM-DD}_{uuid}/
│       ├── meta.json             # id, timestamp, type, source
│       └── content.*             # note.md | url.txt | file.pdf
│
├── wiki/                         # Processed, classified, linked notes
│   └── {para_category}/
│       └── {note-id}.md          # Frontmatter + body + wikilinks
│
├── data/                         # Derived indices (not hand-edited)
│   ├── embeddings.pkl            # {note_id: vector}
│   ├── index.json                # Processing state / checksums
│   └── graph.json                # Nodes + edges for visualization
│
├── capture.py                    # Week 1
├── classify.py                   # Week 2.1
├── link.py                       # Week 2.2
├── build_graph.py                # Week 3.1
├── ask.py                        # Week 4.1
├── app.py                        # Week 4.2 — Streamlit UI
├── pipeline.py                   # Optional: orchestrate full flow
├── lib/                          # Shared utilities
│   ├── models.py                 # Dataclasses / schemas
│   ├── storage.py                # Read/write raw & wiki
│   ├── llm.py                    # Groq client wrapper
│   └── embeddings.py             # sentence-transformers wrapper
├── static/                       # JS graph assets (vis-network)
│   └── graph.html
├── requirements.txt
├── .env.example                  # GROQ_API_KEY
└── README.md
```

---

## 3. Data Models

### 3.1 Raw Capture (`raw/{id}/`)

Every capture gets a **folder** (not a flat file) so metadata and content stay together.

**`meta.json`**

```json
{
  "id": "a1b2c3d4",
  "timestamp": "2026-07-06T22:30:00Z",
  "type": "note | link | file",
  "source": "cli | stdin | path",
  "original_filename": "research.pdf",
  "content_hash": "sha256:..."
}
```

**Content files by type:**

| Type | Stored as | Notes |
|------|-----------|-------|
| `note` | `content.md` | Plain text or markdown |
| `link` | `content.txt` | URL on line 1; optional title/notes below |
| `file` | `content.{ext}` | Binary copy; text extracted later for classify |

**ID generation:** `uuid4().hex[:8]` prefixed with date: `2026-07-06_a1b2c3d4`

---

### 3.2 Wiki Note (`wiki/{category}/{id}.md`)

**Frontmatter (YAML):**

```yaml
---
id: a1b2c3d4
raw_id: 2026-07-06_a1b2c3d4
para: Projects          # Projects | Areas | Resources | Archives
tags: [ml, career]
summary: "One-line summary from LLM"
created: 2026-07-06T22:30:00Z
links: [b2c3d4e5, c3d4e5f6]   # auto-linked note IDs
---
```

**Body:** Cleaned content + inline `[[b2c3d4e5]]` wikilinks where relationships were detected.

**PARA mapping:**

| Category | Meaning | Example |
|----------|---------|---------|
| Projects | Active work with a deadline | "Build SecondSelf" |
| Areas | Ongoing responsibilities | "Health", "Finances" |
| Resources | Reference material | "ML papers", "Recipes" |
| Archives | Inactive / completed | Old project notes |

---

### 3.3 Graph (`data/graph.json`)

```json
{
  "nodes": [
    {
      "id": "a1b2c3d4",
      "label": "ML career pivot plan",
      "para": "Projects",
      "tags": ["ml", "career"],
      "summary": "...",
      "content_preview": "First 200 chars...",
      "group": "Projects"
    }
  ],
  "edges": [
    {
      "source": "a1b2c3d4",
      "target": "b2c3d4e5",
      "weight": 0.87,
      "type": "semantic"
    }
  ],
  "metadata": {
    "generated_at": "2026-07-06T22:30:00Z",
    "node_count": 15,
    "edge_count": 23
  }
}
```

`group` maps to vis-network color clusters (one color per PARA category).

---

### 3.4 Embeddings Index (`data/embeddings.pkl`)

```python
{
  "a1b2c3d4": np.ndarray,  # shape (384,) for all-MiniLM-L6-v2
  "b2c3d4e5": np.ndarray,
}
```

Keyed by wiki note `id`. Recomputed only when note content changes (hash mismatch in `index.json`).

---

## 4. Component Architecture

### 4.1 Capture (`capture.py`) — Week 1

**CLI interface:**

```bash
python capture.py note "Remember to review embeddings paper"
python capture.py link "https://arxiv.org/abs/..."
python capture.py file ./documents/resume.pdf
python capture.py                    # interactive stdin mode
```

**Internal flow:**

```
Input (note/link/file)
    │
    ├─▶ Detect type
    ├─▶ Generate id + timestamp
    ├─▶ Copy/store content
    ├─▶ Write meta.json
    └─▶ Print confirmation: "Captured → raw/2026-07-06_a1b2c3d4"
```

**Key functions:**

- `capture_note(text: str) -> CaptureResult`
- `capture_link(url: str, notes: str = "") -> CaptureResult`
- `capture_file(path: Path) -> CaptureResult`

**File extraction (for later weeks):** PDF via `pypdf`, plain text as-is. Defer heavy extraction to classify stage.

---

### 4.2 Classify (`classify.py`) — Week 2.1

**Purpose:** Transform unprocessed `raw/` items into structured `wiki/` notes.

**Flow:**

```
For each raw/ item where index.json says "unprocessed":
    │
    ├─▶ Extract text (markdown, URL fetch + strip HTML, PDF text)
    ├─▶ Call Groq LLM with structured prompt
    │       → para, tags[], summary
    ├─▶ Write wiki/{para}/{id}.md with frontmatter
    └─▶ Mark processed in index.json
```

**LLM prompt pattern (structured output):**

```
You are a personal knowledge librarian using the PARA method.
Given this capture, return JSON:
{ "para": "...", "tags": [...], "summary": "..." }

Content:
---
{extracted_text}
---
```

**Model:** `llama-3.1-8b-instant` on Groq (free tier, fast).

**`lib/llm.py` responsibilities:**

- API key from `GROQ_API_KEY` env var
- Retry with exponential backoff
- JSON parse + schema validation
- Token truncation for long captures (keep first ~4000 tokens)

---

### 4.3 Link (`link.py`) — Week 2.2

**Purpose:** Find semantically related notes and insert wikilinks.

**Flow:**

```
For each new wiki note:
    │
    ├─▶ Compute embedding (title + summary + body)
    ├─▶ Compare cosine similarity vs all existing embeddings
    ├─▶ For pairs where similarity ≥ THRESHOLD (e.g. 0.75):
    │       ├─▶ Add to frontmatter links[]
    │       └─▶ Append [[other-id]] in body (deduplicated)
    ├─▶ Update embeddings.pkl
    └─▶ Regenerate affected graph edges (or defer to build_graph)
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (local, ~80MB, 384-dim).

**Similarity:**

```python
cosine_sim(a, b) = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Threshold tuning:** Start at `0.75`; lower to `0.65` if graph feels sparse, raise to `0.80` if too noisy.

**Orchestration script (`pipeline.py`):**

```bash
python pipeline.py process    # classify + link all new raw items
python pipeline.py classify   # classify only
python pipeline.py link       # link only
```

---

### 4.4 Graph Builder (`build_graph.py`) — Week 3.1

**Purpose:** Materialize `graph.json` from wiki notes.

**Node extraction:** Parse every `wiki/**/*.md` → one node per note.

**Edge extraction (two sources):**

1. **Explicit links:** `[[note-id]]` in body + `links[]` in frontmatter
2. **Semantic edges:** From `link.py` similarity scores (optional second edge type)

**Deduplication:** Edge key = `(min(source,target), max(source,target))`.

---

### 4.5 Interactive Graph (Week 3.2)

**Recommended library:** **vis-network** (simpler Streamlit integration than Cytoscape).

**Architecture options:**

| Approach | Pros | Cons |
|----------|------|------|
| **A. `st.components.v1.html()`** | Native Streamlit, no separate server | iframe sizing quirks |
| **B. Standalone `static/graph.html` + JSON fetch** | Easier to debug graph in browser | Needs file serving in deploy |

**Recommended: Approach A** — embed vis-network HTML directly in Streamlit.

**Graph interactions:**

- **Hover:** Tooltip with `summary` + `content_preview`
- **Click:** Optional sidebar with full note content
- **Drag / zoom:** vis-network physics engine (Barnes-Hut)
- **Color:** Node `group` = PARA category
- **Pulse effect:** CSS animation on nodes with `borderWidth` oscillation

**Physics config (force-directed):**

```javascript
physics: {
  barnesHut: { gravitationalConstant: -8000, springLength: 150 },
  stabilization: { iterations: 200 }
}
```

---

### 4.6 Ask Engine (`ask.py`) — Week 4.1

**RAG pipeline:**

```
User question
    │
    ├─▶ Embed question (same model as notes)
    ├─▶ Top-K retrieval (K=5) by cosine similarity from embeddings.pkl
    ├─▶ Load full wiki content for retrieved note IDs
    ├─▶ Build context prompt:
    │       "Answer using ONLY these notes. Cite note IDs."
    ├─▶ Call Groq LLM → synthesized answer
    └─▶ Return { answer, sources: [{id, summary, score}] }
```

**`ask()` signature:**

```python
def ask(question: str, top_k: int = 5) -> AskResult:
    """
    AskResult:
      answer: str
      sources: list[{id, summary, relevance_score, para}]
    """
```

**Prompt template:**

```
You are SecondSelf, answering from the user's personal knowledge base.
Use ONLY the provided notes. If the answer isn't in the notes, say so.
Cite sources as [note-id].

Notes:
{retrieved_notes}

Question: {question}
```

**Guardrails:**

- Max context window: truncate notes to fit (~6000 tokens total)
- Temperature: `0.3` for factual synthesis
- Return source list so UI can show "based on these notes"

---

### 4.7 Streamlit App (`app.py`) — Week 4.2

**Layout:**

```
┌────────────────────────────────────────────────────┐
│  🧠 SecondSelf                    [Refresh Graph]  │
├────────────────────────────────────────────────────┤
│  Ask your brain: [________________________] [Ask]  │
│  ┌──────────────────────────────────────────────┐  │
│  │ Answer panel (markdown rendered)             │  │
│  │ Sources: [note-1] [note-2] ...               │  │
│  └──────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────┐  │
│  │     Interactive Knowledge Graph (vis-network) │  │
│  │              (hover / drag / zoom)             │  │
│  └──────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────┤
│  Sidebar: Capture form | Pipeline status | Stats   │
└────────────────────────────────────────────────────┘
```

**Session state:** Cache `graph.json` and embeddings in `@st.cache_resource`.

**Sidebar actions:**

- Quick capture (note text area → calls `capture.py` logic)
- "Process new captures" button → runs classify + link + rebuild graph
- Stats: note count, edge count, last processed timestamp

---

## 5. End-to-End Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Capture
    participant Raw as raw/
    participant Classify
    participant Link
    participant Wiki as wiki/
    participant Graph as build_graph
    participant App as Streamlit
    participant Ask as ask()

    User->>Capture: note / link / file
    Capture->>Raw: meta.json + content

    User->>App: "Process captures"
    App->>Classify: process unprocessed raw
    Classify->>Wiki: classified .md files
    Classify->>Link: new note IDs
    Link->>Wiki: insert [[wikilinks]]
    Link->>Link: update embeddings.pkl
    App->>Graph: rebuild
    Graph->>Graph: write graph.json

    User->>App: "What did I save about ML?"
    App->>Ask: question
    Ask->>Ask: embed → retrieve top-K
    Ask->>Wiki: load note bodies
    Ask->>Ask: LLM synthesize
    Ask->>App: answer + sources
    App->>User: render answer + highlight graph nodes
```

---

## 6. Processing State (`data/index.json`)

Tracks what's been processed to make pipelines **idempotent**:

```json
{
  "raw_processed": {
    "2026-07-06_a1b2c3d4": {
      "wiki_id": "a1b2c3d4",
      "classified_at": "2026-07-06T22:35:00Z",
      "content_hash": "sha256:..."
    }
  },
  "embeddings_version": "all-MiniLM-L6-v2",
  "last_graph_build": "2026-07-06T22:40:00Z"
}
```

Re-processing: if `content_hash` changes, re-classify and re-embed.

---

## 7. Technology Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11+ | Ecosystem for ML + Streamlit |
| LLM | Groq + Llama 3.1 8B | Free tier, fast, good enough for classify + RAG |
| Embeddings | sentence-transformers (local) | Free, offline, no API cost |
| UI | Streamlit | Fast to ship; embeds HTML components |
| Graph viz | vis-network | Force-directed, hover, drag, zoom out of box |
| Text extraction | pypdf, requests+beautifulsoup4 | PDF + URL content |
| Config | python-dotenv | `GROQ_API_KEY` management |
| Deploy | Streamlit Community Cloud | Free public URL, git-push deploy |

**`requirements.txt` (starter):**

```
streamlit>=1.32
groq>=0.4
sentence-transformers>=2.3
numpy>=1.24
pyyaml>=6.0
pypdf>=4.0
requests>=2.31
beautifulsoup4>=4.12
python-dotenv>=1.0
```

---

## 8. Deployment Architecture

```
GitHub Repo (secondself)
        │
        ▼
Streamlit Community Cloud
        │
        ├─▶ Secrets: GROQ_API_KEY
        ├─▶ Bundled: wiki/, data/graph.json, data/embeddings.pkl
        └─▶ Public URL: https://secondself-username.streamlit.app
```

### Deployment Considerations

| Concern | Strategy |
|---------|----------|
| **Private notes on public URL** | Ship with *your* demo data OR add auth later; document this tradeoff |
| **Embedding model cold start** | `@st.cache_resource` loads model once; first query slow (~10s) |
| **Repo size** | Don't commit `raw/` binaries; commit `wiki/` + `data/` for demo |
| **API key** | Streamlit Secrets manager; never commit `.env` |
| **Graph refresh** | Pre-build `graph.json` in CI or on "Process" button |

**Alternative:** Hugging Face Spaces with Streamlit SDK — same architecture, different host.

---

## 9. Module Dependency Graph

```
capture.py          (standalone)

classify.py ──────▶ lib/llm.py
              └──▶ lib/storage.py

link.py     ──────▶ lib/embeddings.py
              └──▶ lib/storage.py

build_graph.py ──▶ lib/storage.py

ask.py      ──────▶ lib/embeddings.py
              ├──▶ lib/llm.py
              └──▶ lib/storage.py

app.py      ──────▶ capture, classify, link, build_graph, ask
              └──▶ static/graph HTML component

pipeline.py ──────▶ classify + link + build_graph (orchestrator)
```

---

## 10. Week-by-Week Build Milestones

| Week | Components | Validates |
|------|------------|-----------|
| **1 — Archivist** | `raw/`, `wiki/`, `capture.py`, `lib/storage.py` | 10+ real captures with id + timestamp |
| **2 — Librarian** | `classify.py`, `link.py`, `lib/llm.py`, `lib/embeddings.py` | 15+ items in organized `wiki/` with links |
| **3 — Cartographer** | `build_graph.py`, vis-network component | Interactive graph from real notes |
| **4 — Oracle** | `ask.py`, `app.py`, Streamlit deploy | RAG answers + public URL |

---

## 11. Error Handling & Edge Cases

| Scenario | Handling |
|----------|----------|
| Duplicate capture (same URL) | Hash check in `index.json`; skip or warn |
| LLM returns invalid JSON | Retry once; fallback to `para: Resources` |
| URL fetch fails | Store URL only; classify from URL string |
| PDF unreadable | Store file; summary = filename only |
| No similar notes found | Note stands alone; graph is a single node |
| Question with no relevant notes | `ask()` returns "I don't have notes about that" |
| Very long note | Truncate for LLM; full text kept in wiki file |

---

## 12. Future Extensions (post-v1)

These are out of scope for the 4-week build but the architecture supports them:

- **SQLite/FTS** for keyword search alongside semantic retrieval
- **Webhook capture** (browser extension → POST to API)
- **Multi-user auth** (Streamlit-Authenticator or move to FastAPI + React)
- **Incremental graph updates** (WebSocket push on new captures)
- **Obsidian sync** (wiki/ is already compatible with Obsidian vault structure)
- **Hybrid retrieval** (BM25 + embeddings reranking)

---

## 13. Suggested Implementation Order

1. `lib/models.py` + `lib/storage.py` — shared schemas first
2. `capture.py` — get real data flowing into `raw/`
3. `classify.py` + `lib/llm.py` — PARA organization
4. `link.py` + `lib/embeddings.py` — semantic linking
5. `build_graph.py` — JSON export
6. Graph HTML component — visualize in browser, then embed in Streamlit
7. `ask.py` — RAG Q&A
8. `app.py` — unify UI
9. Deploy + README

---

This architecture maps directly onto the suggested repo structure in `PROBLEM_STATEMENT.md`, with added `lib/` for shared code, `data/` for derived indices, and explicit schemas so each weekly milestone stays testable on its own.
