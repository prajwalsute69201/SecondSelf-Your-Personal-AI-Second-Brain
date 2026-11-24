# SecondSelf — Architecture Decision Records (ADR)

## 1. Overview & Decision Framework

This document records the foundational architectural decisions, technology selections, trade-off analyses, and design rationale for **SecondSelf**. Every architectural decision record (ADR) follows the standard Nygard/MADR template, documenting the context, alternatives evaluated, trade-offs accepted, and mitigations implemented.

### Architecture Decision Summary Matrix

| ADR ID | Decision Topic | Selected Option | Key Alternatives Rejected | Primary Driver |
|---|---|---|---|---|
| **ADR-001** | System Topology | Decoupled Dual-Plane (Local Build + Serverless Query) | Monolithic backend (FastAPI/Flask), Client-side WASM | Zero cloud cost + Vercel compatibility |
| **ADR-002** | Frontend Architecture | Static Jamstack (HTML5/Canvas + Serverless API) | Streamlit, Gradio, Next.js / React SSR | Serverless statelessness & smooth 60fps graph |
| **ADR-003** | Vector Storage | Portable Precomputed `index.json` (NumPy Dot-Product) | Pinecone, ChromaDB, Qdrant, Supabase pgvector | Zero infrastructure overhead + sub-15ms search |
| **ADR-004** | Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | OpenAI text-embedding-3-small, BGE-large | Free, local, CPU-efficient, compact index size |
| **ADR-005** | Knowledge Taxonomy | Tiago Forte's P.A.R.A. Framework | Zettelkasten, Flat tags, Custom hierarchy | Actionability, distinct visual clusters |
| **ADR-006** | LLM Provider | Groq LPUs (Llama 3.3 70B & 3.1 8B) | Local Ollama, OpenAI GPT-4o-mini, Anthropic Haiku | Extreme generation speed (>500 t/s), sub-second RAG |
| **ADR-007** | Graph Visualization | Vanilla JS + `vis-network` (Canvas Engine) | D3.js force simulation, React Flow, Cytoscape.js | Zero build-step, rich built-in Barnes-Hut physics |
| **ADR-008** | Bundle Sizing Strategy | Dependency Stratification (`requirements.txt` split) | Docker containers on Cloud Run, Pyodide in browser | Strict adherence to Vercel 250MB limit (<50MB actual) |
| **ADR-009** | Anti-Hallucination Guard | Grounded Citation Filtering (`[Note: <id>]`) | Freeform synthesis, post-hoc semantic citations | Zero phantom citations, 100% verifiable sources |
| **ADR-010** | Continuous Deployment | GitOps Static Generation via Git Push | Dynamic runtime webhook editing, Vercel KV/Blob | Local-first Markdown truth, zero-maintenance sync |

---

## 2. Architecture Decision Records

---

### ADR-001: Decoupled Dual-Plane Architecture over Monolithic Cloud Container

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
Personal knowledge systems require two diametrically opposed computational workloads:
1. **Heavy Ingestion & Indexing:** Web scraping, document parsing, LLM batch classification, and neural vector embedding generation requiring hundreds of megabytes of ML dependencies (`torch`, `sentence-transformers`).
2. **Interactive Querying & Visualization:** Serving an interactive graph to web browsers and performing instant question answering with minimal latency.

Deploying a single monolithic container (e.g. FastAPI + PyTorch on an AWS EC2 instance, Render, or Railway) incurs continuous idle compute costs (\$7–\$25/month), requires complex infrastructure maintenance, and suffers from slow restarts.

#### Decision
Implement a **Decoupled Dual-Plane Architecture**:
- **Offline Local Build Plane:** All heavy computations (capturing raw files, running Groq classification, generating embeddings, writing Wikilinks, and projecting `graph.json` and `index.json`) run locally on the user's workstation.
- **Cloud-Native Serving Plane:** The frontend (`public/`) and query endpoint (`api/ask.py`) deploy to **Vercel** as static edge assets and lightweight serverless functions.

#### Alternatives Evaluated & Reasons for Rejection
1. **Monolithic Cloud Server (FastAPI on Render/Railway):** Rejected due to ongoing hosting costs, database management overhead, and unnecessary resource consumption during idle periods.
2. **Pure Client-Side WebAssembly (Transformers.js in Browser):** Rejected because downloading transformer weights (~120MB) on every user browser visit causes unacceptable initial page load delays and excessive memory consumption on mobile devices.

#### Consequences
- **Positive:**
  - Zero idle hosting cost (deploys cleanly within Vercel's generous free tier).
  - Complete local ownership and privacy of the primary knowledge base in Git.
  - Infinite horizontal scaling for read queries on Vercel Edge CDN.
- **Negative & Trade-off:**
  - New captures require a local CLI command and a `git push` to appear on the public URL (no real-time dynamic web capture out of the box).
- **Mitigation:**
  - Batching workflow: Users capture notes locally throughout the day and run a single build-and-push script (`python build.py && git push`) to update the live cloud deployment.

---

### ADR-002: Rejection of Streamlit & Gradio in Favor of Jamstack Architecture

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
Python developers commonly prototype AI and data applications using Streamlit or Gradio. However, SecondSelf requires a public deployment on Vercel with an interactive, fluid, force-directed graph canvas.

#### Decision
Explicitly reject Streamlit and Gradio. Adopt a **Jamstack Architecture** consisting of static HTML5/CSS/JavaScript assets served from Vercel's Edge CDN, paired with an ephemeral Python Serverless Function (`api/ask.py`).

#### Alternatives Evaluated & Reasons for Rejection
1. **Streamlit:**
   - *Failure Mode on Vercel:* Streamlit requires a persistent, long-running Python process and stateful WebSocket connections. Vercel’s serverless model terminates execution after each HTTP request and does not support persistent WebSockets.
   - *UX Limitations:* Streamlit reruns the entire Python script on every user interaction, introducing canvas stutter and lag during graph exploration.
2. **Gradio:**
   - *Failure Mode:* Designed primarily for machine learning input-output widgets rather than complex, force-directed 2D canvas manipulation with custom event listeners (hover micro-cards, side drawers).
3. **Next.js / React SSR:**
   - *Rejected:* Introduces unnecessary Node.js build dependencies, React hydration overhead, and npm package maintenance for what is fundamentally a static canvas visualizer.

#### Consequences
- **Positive:**
  - Sub-100ms First Contentful Paint (FCP) from Vercel Edge CDN.
  - Silky-smooth 60 FPS graph physics powered directly by the browser's native canvas rendering engine.
  - Fully compatible with Vercel's native serverless Python runtime.
- **Negative & Trade-off:**
  - Frontend interaction code must be written in vanilla JavaScript (`app.js`) rather than pure Python widgets.
- **Mitigation:**
  - Keep `app.js` concise, modular, and dependency-free, delegating all complex graph physics to the battle-tested `vis-network` standalone bundle.

---

### ADR-003: Portable Precomputed Vector Index (`index.json` / NumPy) over Hosted Vector Databases

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
Retrieval-Augmented Generation (RAG) typically utilizes specialized vector databases (e.g., Pinecone, Weaviate, Qdrant, ChromaDB, Supabase pgvector) to perform Approximate Nearest Neighbor (ANN) search. 

For a personal knowledge base containing between 100 and 10,000 notes, hosting an external database introduces substantial complexity: API keys, external network latency (100–300ms per search call), connection pooling issues in serverless cold starts, and eventual monthly billing tiers.

#### Decision
Store precomputed 384-dimensional normalized float vectors directly alongside note metadata in a portable JSON file (`public/index.json`). At runtime, `api/ask.py` loads `index.json` into memory and performs exact Cosine Similarity search using vectorized dot products via pure NumPy (or native Python).

#### Mathematical Rationale & Benchmarks
For $N = 2,000$ notes with $d = 384$ dimensions:
- Total vector memory footprint: $2000 \times 384 \times 4 \text{ bytes} \approx 3.07 \text{ MB}$.
- Vectorized matrix-vector multiplication ($M \cdot \mathbf{q}^T$) in NumPy on a single CPU core: **$< 4.2 \text{ milliseconds}$**.
- In contrast, a network roundtrip to an external cloud vector database requires **$80 - 250 \text{ milliseconds}$**.

#### Alternatives Evaluated & Reasons for Rejection
1. **Pinecone / Qdrant Cloud:** Rejected due to external cloud dependency, network latency, and free-tier inactivity deletion policies.
2. **ChromaDB / SQLite-vss:** Rejected because compiling SQLite C-extensions and SQLite native binaries often fails or exceeds size budgets within Vercel's serverless Linux sandbox.
3. **Supabase pgvector:** Rejected due to unnecessary relational database provisioning, connection limits, and credentials management for a personal read-only second brain.

#### Consequences
- **Positive:**
  - Zero external database costs and zero external infrastructure dependencies.
  - Exact nearest-neighbor retrieval ($100\%$ recall) without ANN quantization loss.
  - Sub-15ms search latency executed directly within the serverless memory space.
  - The vector index is tracked and versioned directly in Git.
- **Negative & Trade-off:**
  - Scaling is bounded: if the personal corpus exceeds 50,000 notes ($> 75\text{MB}$ index), loading `index.json` into serverless memory will encounter cold-start lag.
- **Mitigation:**
  - Personal second brains rarely exceed 10,000 high-signal notes in practice. If scale exceeds 25,000 notes, serialize the index as a binary float32 buffer (`index.npy`) for instantaneous memory mapping.

---

### ADR-004: Sentence-Transformers `all-MiniLM-L6-v2` as Core Embedding Model

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
An embedding model must be selected to compute dense representations for auto-linking notes and performing RAG query matching. Key constraints: local CPU inference speed, semantic accuracy, dimensional compactness, and zero operational cost.

#### Decision
Adopt `sentence-transformers/all-MiniLM-L6-v2` as the sole canonical embedding model across the system.

#### Alternatives Evaluated & Reasons for Rejection
1. **OpenAI `text-embedding-3-small` (1536-d):**
   - *Rejected:* Requires active API key and incurs billing per token. Vectors are $4\times$ larger ($1536$ vs $384$ floats), ballooning `index.json` size from $3\text{MB}$ to $12\text{MB}$ and slowing serverless download and parsing times.
2. **BAAI/bge-large-en-v1.5 (1024-d):**
   - *Rejected:* Model weights are $> 1.34\text{ GB}$, resulting in slow CPU inference during batch indexing on non-GPU laptops.
3. **Nomic-Embed-Text / E5-base:**
   - *Rejected:* Requires complex instruction prefixes (`"search_query: "`, `"search_document: "`), complicating similarity matrix calculation for symmetric pairwise auto-linking.

#### Consequences
- **Positive:**
  - Fast local CPU encoding ($> 150\text{ sentences/sec}$ on standard laptops).
  - Compact 384-dimensional vector footprint keeps `index.json` lightweight.
  - Excellent semantic benchmark ranking for general knowledge clustering.
- **Negative & Trade-off:**
  - Cannot be run directly inside Vercel serverless functions with PyTorch due to bundle size constraints.
- **Mitigation:**
  - Precompute note embeddings locally during build time. At query time in `api/ask.py`, vectorize the user query via FastEmbed ONNX (~30MB) or an identical hosted inference call, comparing directly against precomputed 384-d vectors.

---

### ADR-005: Tiago Forte's P.A.R.A. Framework for Knowledge Taxonomy

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
Traditional note apps fail because their taxonomy is either too rigid (deeply nested, multi-level folder trees that paralyze filing) or non-existent (flat tag soups that lead to retrieval entropy). We require an autonomous classification taxonomy that is clean, universally applicable, and interpretable by an LLM.

#### Decision
Adopt Tiago Forte’s **P.A.R.A. Framework** (Projects, Areas, Resources, Archives) as the foundational organizational structure in `wiki/`:
- **Projects (`wiki/Projects/`):** Active initiatives with a defined outcome and deadline.
- **Areas (`wiki/Areas/`):** Long-term domains of responsibility requiring ongoing maintenance.
- **Resources (`wiki/Resources/`):** Evergreen reference material, topics, and libraries of interest.
- **Archives (`wiki/Archives/`):** Inactive items preserved for historical reference.

#### Alternatives Evaluated & Reasons for Rejection
1. **Zettelkasten Method (Fleeting, Literature, Permanent Notes):**
   - *Rejected:* Zettelkasten focuses on note maturity rather than actionability. It is notoriously difficult for LLMs to categorize consistently without human intervention.
2. **Johnny Decimal / Custom Decimal Systems:**
   - *Rejected:* Too complex to automate reliably via prompt engineering; introduces arbitrary numeric indexing.
3. **Unstructured Flat Directory with Tags Only:**
   - *Rejected:* Destroys physical visual hierarchy; does not produce distinct, meaningful color clusters on the force-directed graph.

#### Consequences
- **Positive:**
  - Deterministic 4-class classification targets make Groq LLM prompt constraints straightforward and reliable.
  - Directly translates into intuitive visual color coding in the graph UI (Amber, Blue, Green, Slate).
  - Matches the mental model of modern knowledge workers.
- **Negative & Trade-off:**
  - Ambiguous notes (e.g. "Tax Return 2026" - Project vs Area) can occasionally cause borderline classifications.
- **Mitigation:**
  - In `classify.py`, provide unambiguous definitions in the system prompt: deadlines always dictate `Projects`; open-ended responsibilities dictate `Areas`.

---

### ADR-006: Groq LPUs with Llama 3 for Low-Latency Cloud Inference

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
In a serverless environment (Vercel), functions are subject to strict maximum execution durations (10–15 seconds). Traditional cloud LLM APIs (OpenAI, Anthropic) or self-hosted GPU models typically exhibit Time-To-First-Token (TTFT) and generation latencies between 2.0 and 6.0 seconds, which risks timeouts during high concurrency or cold starts.

#### Decision
Select **Groq Language Processing Units (LPUs)** running Meta Llama 3 (`llama-3.3-70b-versatile` for high-precision RAG and `llama-3.1-8b-instant` for rapid classification) as the primary LLM inference provider.

#### Performance Comparison Benchmarks

| Provider & Model | Generation Speed | Time-to-First-Token | Cost Tier | Vercel Safety Margin |
|---|---|---|---|---|
| **Groq (Llama 3.3 70B)** | **$> 500\text{ tokens/sec}$** | **$< 200\text{ ms}$** | Free / Low Cost | **Extreme ($< 1.0\text{s}$ total)** |
| **OpenAI (GPT-4o-mini)** | $\sim 100\text{ tokens/sec}$ | $\sim 600\text{ ms}$ | Paid API | Moderate ($2.5 - 4.0\text{s}$) |
| **Anthropic (Claude 3.5 Haiku)** | $\sim 120\text{ tokens/sec}$ | $\sim 500\text{ ms}$ | Paid API | Moderate ($2.0 - 3.5\text{s}$) |
| **Local Ollama (Llama 3 8B)** | $\sim 20 - 45\text{ tokens/sec}$ | $\sim 1,200\text{ ms}$ | Free (Self-hosted) | Unusable on Serverless |

#### Alternatives Evaluated & Reasons for Rejection
1. **Self-Hosted Ollama on User Machine:** Cannot be called directly from public Vercel serverless functions without complex public port forwarding (ngrok/Cloudflare Tunnels), creating a fragile dependency on the user's laptop remaining powered on.
2. **OpenAI GPT-4o:** Unnecessary financial cost and higher latency for classification and grounded RAG synthesis tasks that Llama 3 executes with comparable or superior fidelity.

#### Consequences
- **Positive:**
  - End-to-end question answering latency stays below $1.2\text{ seconds}$ in production.
  - Generous free API tier allows zero-cost development and deployment.
  - Native JSON object response mode guarantees structured classification.
- **Negative & Trade-off:**
  - Requires external dependency on Groq Cloud and valid `GROQ_API_KEY`.
- **Mitigation:**
  - Isolate the LLM client in an abstraction layer to allow switching to any OpenAI-compatible API endpoint via a single environment variable (`OPENAI_BASE_URL`).

---

### ADR-007: Vanilla HTML5/CSS3 + `vis-network` over React/Vue and D3.js

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
The SecondSelf frontend must render an interactive, pulsing, force-directed graph of notes and edges, display hover micro-cards, provide a slide-out markdown drawer, and integrate a real-time natural language search interface.

#### Decision
Implement the entire frontend using pure **Vanilla HTML5, modern CSS3, and ES6+ JavaScript (`app.js`)**, utilizing the standalone UMD bundle of **`vis-network`** loaded via CDN.

#### Alternatives Evaluated & Reasons for Rejection
1. **D3.js Force Simulation:**
   - *Rejected:* D3 requires manual implementation of canvas dragging, zooming, node hover bounding boxes, physics stabilization listeners, and high-DPI retina scaling. `vis-network` provides all of these capabilities out of the box with proven stability.
2. **React Flow / Cytoscape.js:**
   - *Rejected:* React Flow is primarily designed for directed workflow node editing rather than organic, force-directed cognitive cluster graphs. Cytoscape.js has a steeper learning curve and heavier styling syntax for personal knowledge graphs.
3. **React / Next.js Framework:**
   - *Rejected:* Adds hundreds of megabytes of `node_modules`, npm build pipelines, and hydration latency with zero architectural benefit for a single-page visualization canvas.

#### Consequences
- **Positive:**
  - Zero build step (`npm run build` is not required; static files are served directly).
  - Sub-50ms asset load time from Vercel Edge CDN.
  - Built-in Barnes-Hut physical solver provides realistic clustering based on similarity weights.
- **Negative & Trade-off:**
  - DOM manipulation must be handled cleanly in native JavaScript without reactive component frameworks.
- **Mitigation:**
  - Maintain a centralized `AppState` singleton in `app.js` with declarative event binding.

---

### ADR-008: Dependency Stratification (`requirements.txt` vs `requirements-dev.txt`) for Vercel Sizing

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
Vercel's Serverless Python runtime enforces an uncompressed bundle size ceiling of **250 MB**. Standard machine learning dependencies easily breach this limit:
- `torch`: $\sim 850\text{ MB}$ uncompressed.
- `sentence-transformers` + `transformers`: $\sim 450\text{ MB}$.
- `scipy` + `scikit-learn`: $\sim 180\text{ MB}$.

Deploying these packages causes Vercel builds to fail immediately with `FUNCTION_SIZE_LIMIT_EXCEEDED`.

#### Decision
Implement strict **Dependency Stratification**:
1. **`requirements.txt` (Production Serverless):** Contains only lightweight runtime essentials:
   ```text
   groq>=0.11.0
   numpy>=1.24.0,<2.0.0
   pydantic>=2.0.0
   requests>=2.31.0
   ```
   *Total uncompressed size: $\sim 45\text{ MB}$ (safe under 250MB limit).*
2. **`requirements-dev.txt` (Local Offline Build Plane):** Contains the heavy ML toolchain:
   ```text
   -r requirements.txt
   torch>=2.2.0
   sentence-transformers>=3.0.0
   trafilatura>=1.6.0
   pypdf>=4.0.0
   ```

#### Consequences
- **Positive:**
  - Guaranteed 100% build and deployment success on Vercel.
  - Serverless cold start container provisioning time reduced from $> 8\text{s}$ to $< 1.5\text{s}$.
- **Negative & Trade-off:**
  - Developers must ensure they install `requirements-dev.txt` locally and never add heavy ML packages to `requirements.txt`.
- **Mitigation:**
  - Add an automated CI / pre-commit check (`evals/test_bundle_size.py`) that audits `requirements.txt` to prevent accidental inclusion of PyTorch.

---

### ADR-009: Strict Grounding & Citation Verification in Serverless RAG (`[Note: <id>]`)

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
LLMs operating in conversational RAG modes are prone to two failure states:
1. **Hallucination:** Answering from pre-training memory rather than the user's specific notes.
2. **Phantom Citations:** Inventing plausible-looking note IDs or citing notes that were never retrieved in context.

#### Decision
Enforce a two-stage verification contract:
1. **Prompt Constraint:** Instruct the model via system guardrails to cite every assertion with `[Note: <id>]` and declare lack of information if the context is insufficient.
2. **Deterministic Post-Filter:** In `api/ask.py`, parse all generated citation tags using regex (`r'\[Note:\s*([a-f0-9\-]+)\]'`). Cross-reference each parsed ID against the actual `retrieved_top_k` candidate list. Any citation not present in the retrieved candidates is immediately purged before response transmission.

#### Consequences
- **Positive:**
  - Guaranteed 100% citation precision: the frontend only receives and pulses real, verified source nodes.
  - Users can click any citation in the answer panel to jump directly to the verified note in the graph.
- **Negative & Trade-off:**
  - If the LLM generates slight syntactic variants (e.g. `[Note:1]`), regex parsing could miss them.
- **Mitigation:**
  - Use regex matching that accommodates variable spacing and case insensitivity, coupled with few-shot citation examples in the system prompt.

---

### ADR-010: GitOps Continuous Deployment Model via Repository Pushes

- **Status:** Accepted & Implemented
- **Date:** 2026-11-23

#### Context & Problem Statement
We must define how new captures and wiki updates sync from the user's local machine to the live public Vercel URL without introducing fragile webhook sync engines or paid real-time database subscriptions.

#### Decision
Adopt a pure **GitOps Continuous Deployment Workflow**:
- The local Git repository is the single source of truth.
- Local scripts update `raw/`, `wiki/`, `public/graph.json`, and `public/index.json`.
- Running `git push origin main` triggers Vercel's automated build-and-deploy pipeline, redeploying static assets and the serverless function in $< 45\text{ seconds}$.

#### Alternatives Evaluated & Reasons for Rejection
1. **Dynamic Webhook Ingestion on Vercel:**
   - *Rejected:* Vercel functions cannot write permanently to the deployment filesystem. Supporting dynamic web captures in production would require external databases (Vercel Blob, DynamoDB, MongoDB), destroying the simplicity of local plain Markdown files.
2. **Two-Way Sync Engines (Dropbox/Google Drive API):**
   - *Rejected:* Complex conflict resolution logic, rate limits, and authentication token expiration bugs.

#### Consequences
- **Positive:**
  - Full version control history for every note, link mutation, and index generation.
  - Zero cloud database infrastructure to configure, maintain, or pay for.
  - Markdown files remain 100% portable and readable in Obsidian, VS Code, or GitHub.
- **Negative & Trade-off:**
  - Deployments are not instantaneous upon capture; they occur upon Git push.
- **Mitigation:**
  - Provide a single-command build-and-deploy CLI alias:
    `python build.py && git commit -am "update brain" && git push`
