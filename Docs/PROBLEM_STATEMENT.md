# Project: SecondSelf — Your Personal AI Second Brain

## The Big Problem

Every notes app fails the same way: you capture hundreds of notes, bookmarks, PDFs, and
ideas — and then you never find them again. Information goes in, but nothing comes back out.
Notes sit in folders nobody re-reads. Bookmarks pile up unread. Knowledge doesn't compound.

**Goal:** Build an end-to-end system where you can capture *anything* (a note, a link, a file),
have AI automatically classify and file it, auto-link it to related knowledge, render it as a
live interactive graph you can explore, and — most importantly — *ask it any question in plain
English and get an answer synthesized from your own accumulated knowledge*. Then deploy it to a
public **Vercel** URL anyone can open.

> Not a notes app. Not a chatbot. A brain that organizes itself and answers for you.

---

## Final System (what you're building over 4 weeks)

```
Capture any note/link/file                      (local, Python)
        ↓
AI classifies & files it (PARA method)          (local, Python + Groq)
        ↓
AI auto-links it to related notes (embeddings)  (local, Python + sentence-transformers)
        ↓
Build graph.json + embeddings index             (local build step)
        ↓
Static frontend: live, interactive, hoverable graph          (Vercel static hosting)
        ↓
Ask it anything in plain English → answer from YOUR notes    (Vercel Python serverless function)
        ↓
Deployed on a public https://<your-app>.vercel.app URL
```

### Architecture at a glance

| Layer | Where it runs | Tech |
|---|---|---|
| Capture, classify, link, build graph | Your machine (offline "build" pipeline) | Python, Groq, sentence-transformers |
| Interactive graph + search bar (UI) | Vercel static hosting | HTML/CSS/JS + vis-network (or Cytoscape.js) |
| `ask()` endpoint | Vercel Serverless Function | Python (`api/ask.py`), Groq LLM |
| Knowledge data (`graph.json`, `index.json`, wiki notes) | Bundled into the deployment | Plain JSON / Markdown |

> **Why not Streamlit?** Streamlit needs a long-running Python server with websockets, which
> Vercel's serverless model does not support. So the project is split into a **pre-built
> knowledge base + static frontend + serverless API** — the standard Vercel pattern.

### Vercel constraints you must design around

- **Serverless size limit:** Python functions have a bundle size cap (~250 MB unzipped).
  `torch` + `sentence-transformers` will NOT fit. → Compute note embeddings **locally**, ship
  them as a JSON/NumPy file, and keep the deployed function lightweight.
- **Read-only filesystem:** a deployed function can't permanently write to `raw/` or `wiki/`.
  → The deployed app is a **read + ask** product; new captures are made locally, then
  `git push` triggers a redeploy. (Stretch goal: persist captures with Vercel Blob / KV.)
- **Secrets:** API keys (e.g. `GROQ_API_KEY`) live in Vercel **Environment Variables**, never
  in the repo.
- **Timeouts:** functions have execution time limits, so `ask()` must stay fast — retrieve
  top-k notes, make one LLM call, return.

---

# Week-by-Week Problem Statements

Each week is a self-contained problem. Build it, test it on **real** data (your own notes —
not test data), and each week's output becomes the next week's input.

---

## Week 1 — The Archivist: "Capture Everything, Lose Nothing"

### Problem
You have no single place to put things. Ideas, links, and notes scatter across apps, browser
tabs, and your memory. Build the foundation: **one command that captures anything into one place.**

### Build
1. Set up the project structure from scratch:
   - `raw/` — where every raw capture lands
   - `wiki/` — (used later) organized, linked notes
2. Write a Python capture script that takes any note, link, or file and saves it into `raw/`
   with:
   - a **timestamp**
   - a **unique ID**
   - the raw content
3. Test it on **10+ real** pieces of your own scattered information.

### Deliverable ("Ship the Capture Pipeline")
- A working capture script — one command saves anything to `raw/` with timestamp + unique ID.
- Your `raw/` folder populated with **10+ real captured items** (not test data).
- 🏅 **Badge: The Archivist**

### Acceptance Criteria
- [ ] `raw/` and `wiki/` folder structure exists
- [ ] One command captures a note, a link, AND a file
- [ ] Every capture has a timestamp + unique ID
- [ ] 10+ real items captured

---

## Week 2 — The Librarian: "Teach AI to Organize For You"

### Problem
A pile of raw captures is still a mess. Manual tagging never happens. Make the AI do the filing —
and make it notice when two notes are about the same thing and link them automatically.

### Build
**2.1 — Auto-Classify (The Sorting Hat)**
- Write a function that sends any raw capture to a free LLM (Groq / Llama 3) and gets back:
  - a **category** (using the **PARA** framework: Projects, Areas, Resources, Archives)
  - **tags**
  - a **one-line summary**
- Run it across last week's real captures and watch them organize themselves.

**2.2 — Auto-Link Related Notes (Connect the Dots)**
- Compute **embeddings** for each note (sentence-transformers, local + free).
- Compare each new capture against existing notes in `wiki/`.
- When content is related (similarity above a threshold), **auto-insert a link** between them.
- Save every note's embedding vector to a portable file (e.g. `index.json`) — Week 4's
  serverless function will reuse it, so it must be readable without `torch`.
- No manual tagging — the system notices relationships on its own.

### Deliverable ("Ship the Self-Organizing Wiki")
- A pipeline that auto-classifies raw captures with PARA and auto-links related notes.
- Run on **15+ real items** → an organized `wiki/` folder with linked notes.
- 🏅 **Badge: The Librarian**

### Acceptance Criteria
- [ ] Any raw capture → category + tags + summary automatically
- [ ] PARA categorization working
- [ ] Embeddings computed per note and saved to a portable index file
- [ ] Related notes auto-linked (no manual tagging)
- [ ] Runs on 15+ real items → organized `wiki/`

---

## Week 3 — The Cartographer: "Visualize the Brain"

### Problem
Your knowledge is now organized and linked — but you can't *see* it. Turn the wiki into
something you can actually look at, explore, and watch think — as a **static web page that
Vercel can host.**

### Build
**3.1 — Graph Data Model (Give It a Shape)**
- Write a script that reads every note and its links.
- Build a **nodes-and-edges** representation in memory:
  - every note → a **node**
  - every relationship/link → an **edge**
- Export it as clean **JSON** into the `public/` folder (`public/graph.json`).

**3.2 — Interactive Graph (The Brain Comes Alive)**
- Build a static frontend (`public/index.html` + JS/CSS) that loads `graph.json` and uses a JS
  graph library (**vis-network** or **Cytoscape.js**) to render:
  - notes as nodes (that pulse / are visually alive)
  - links as edges
  - **hover popups** that reveal each note's content
  - **drag-to-explore** and **zoom**
- A **force-directed** graph of your own knowledge.
- Preview it locally (`python -m http.server` inside `public/` or `vercel dev`).

### Deliverable ("Ship the Living Brain")
- Your wiki converted to a graph and rendered as an interactive visual brain (hover, drag, zoom),
  built from your real notes, running as a static page.
- 🏅 **Badge: The Cartographer**

### Acceptance Criteria
- [ ] Script builds nodes + edges from notes and exports clean JSON to `public/graph.json`
- [ ] Interactive force-directed graph renders from that JSON in a static HTML page
- [ ] Hover reveals note content
- [ ] Drag + zoom work
- [ ] Built from your real notes, not dummy data

---

## Week 4 — The Oracle: "Ask It Anything, Ship It on Vercel"

### Problem
A visual brain is beautiful, but the real payoff is **answers**. Wire up natural-language
search over everything you know — then package the whole thing into one product deployed on
**Vercel**.

### Build
**4.1 — Ask Your Brain (Natural Language Search)**
- Build a single `ask()` function (`ask.py`) that combines:
  - the **embeddings** (find relevant notes to a question)
  - the **wiki** (the source content)
  - an **LLM** (synthesize an answer from retrieved notes)
- This is retrieval-augmented Q&A over your own knowledge.
- Test it locally against real questions about your own captured notes.
- **Serverless-friendly retrieval:** load the precomputed `index.json`, embed the question with
  a lightweight method (e.g. `fastembed` / ONNX, or a hosted embeddings API using the *same*
  model as your notes), and rank notes with plain NumPy cosine similarity.

**4.2 — UI, Deployment, Public URL (Give It a Face)**
- Wrap `ask()` in a Vercel Python Serverless Function at `api/ask.py`
  (`POST /api/ask` with `{ "question": "..." }` → `{ "answer": "...", "sources": [...] }`).
- Extend the static frontend so **one page** contains:
  - the interactive brain graph
  - the ask-anything search bar (calls `/api/ask`)
  - answer panel with source notes that **highlight on the graph**
- Add `vercel.json` / `requirements.txt` and set `GROQ_API_KEY` in Vercel Environment Variables.
- Deploy to **Vercel** (import the GitHub repo in the dashboard, or run `vercel --prod`).
- Get a **public `*.vercel.app` URL** anyone can open.

### Deliverable ("Ship SecondSelf" — the final product)
Deploy the complete system — capture → auto-classify → auto-link → live interactive graph →
ask-anything search — with the build pipeline run locally and the result live on Vercel at a
public URL.
- 🏅 **Badge: The Oracle**

### Acceptance Criteria
- [ ] `ask()` returns answers synthesized from your own notes (retrieval + LLM)
- [ ] `/api/ask` serverless function works locally (`vercel dev`) and in production
- [ ] One page contains both the graph and the search bar
- [ ] Deployed live on Vercel with a public URL
- [ ] API keys stored as Vercel Environment Variables (none committed to the repo)
- [ ] Full pipeline works end to end: capture locally → build → `git push` → redeploy → graph + ask work live

---

## Final Deliverables (whole project)

- [ ] Public **GitHub repo** with a clean README + setup instructions (local build + Vercel deploy)
- [ ] **Live Vercel URL** — interactive graph + ask-your-brain search, both working
- [ ] End-to-end flow verified: capture → classify → link → graph → ask
- [ ] All 4 weekly milestones complete (Capture Pipeline, Self-Organizing Wiki, Living Brain,
      SecondSelf deployment on Vercel)

---

## Suggested Repo Structure

```
secondself/
├── raw/                  # Week 1: raw captures (timestamp + unique ID)
├── wiki/                 # Week 2: classified + auto-linked notes
├── capture.py            # Week 1: one-command capture
├── classify.py           # Week 2: PARA classification via LLM
├── link.py               # Week 2: embeddings + auto-linking (writes index.json)
├── build_graph.py        # Week 3: nodes/edges → public/graph.json
├── ask.py                # Week 4: retrieval + LLM answer (shared logic)
├── api/
│   └── ask.py            # Week 4: Vercel serverless endpoint  POST /api/ask
├── public/               # Static frontend served by Vercel
│   ├── index.html        # Graph + search bar
│   ├── app.js            # vis-network graph + fetch('/api/ask')
│   ├── style.css
│   ├── graph.json        # Week 3: exported graph data
│   └── index.json        # Week 2/4: precomputed embeddings + note text
├── vercel.json           # Vercel routing / function config
├── requirements.txt      # Lightweight deps for the serverless function only
├── requirements-dev.txt  # Local build deps (sentence-transformers, etc.)
├── .env.example          # GROQ_API_KEY=your_key_here
└── README.md
```

## Suggested Build Order in Cursor

1. Scaffold repo structure + `requirements.txt` / `requirements-dev.txt`
2. `capture.py` → test on real items (Week 1)
3. `classify.py` → PARA categories/tags/summary (Week 2.1)
4. `link.py` → embeddings + similarity auto-linking, export `index.json` (Week 2.2)
5. `build_graph.py` → `public/graph.json` nodes/edges (Week 3.1)
6. `public/index.html` + `app.js` → vis-network / Cytoscape graph (Week 3.2)
7. `ask.py` → retrieval-augmented Q&A, tested locally (Week 4.1)
8. `api/ask.py` → serverless wrapper + search bar in the frontend (Week 4.2)
9. Test with `vercel dev`, then deploy with `vercel --prod` (or GitHub import) → public URL
10. Write README, push to GitHub
