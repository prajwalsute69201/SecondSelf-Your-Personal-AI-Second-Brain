# SecondSelf — Vercel Serverless Function (`api/ask.py`)

## 1. Overview & Architecture

`api/ask.py` is a production-ready, stateless serverless HTTP endpoint deploying to **Vercel's Python runtime**. It exposes SecondSelf's grounded question-answering and synthesis engine (`ask.py`) to the static web frontend (`public/`) and third-party HTTP clients without reimplementing retrieval or synthesis logic.

```
+-----------------------------------------------------------------------------------------+
|                                    CLIENT LAYER                                         |
|   Static Web UI (public/)          CLI / curl                 Third-Party Apps          |
+-----------------------------------------------------------------------------------------+
                                           |
                                           | HTTP POST /api/ask  (JSON, max 8 KB)
                                           v
+-----------------------------------------------------------------------------------------+
|                              VERCEL SERVERLESS LAYER                                    |
|   api/ask.py (class handler : BaseHTTPRequestHandler)                                  |
|   - CORS Preflight (OPTIONS -> 204 No Content)                                          |
|   - Payload bounds validation (8 KB ceiling, question <= 1000 chars)                    |
|   - Defensive security headers (nosniff, DENY, strict-origin)                           |
|   - Lazy module imports for sub-second cold starts                                      |
+-----------------------------------------------------------------------------------------+
                                           |
                                           | Delegates to ask()
                                           v
+-----------------------------------------------------------------------------------------+
|                              SYNTHESIS & RETRIEVAL ENGINE                               |
|   ask.cli.ask()                                                                         |
|   ├── HybridRetriever                                                                   |
|   │     ├── Dense Cosine Vector Search  <── public/index.json (Precomputed 384d vectors)|
|   │     └── Graph 1st-Degree Expansion   <── public/graph.json (Force-directed model)   |
|   ├── ContextBuilder (Strict character budget truncation)                               |
|   └── Synthesizer                                                                       |
|         ├── Groq LPU API (llama-3.3-70b-versatile, temperature: 0.0)                    |
|         └── Fallback: Deterministic Retrieval-Only Synthesizer (Zero LLM / Offline)     |
+-----------------------------------------------------------------------------------------+
```

---

## 2. API Contract Specification

### `POST /api/ask`
Executes semantic retrieval and grounded answer synthesis.

#### Headers
| Header | Value / Requirement |
| :--- | :--- |
| `Content-Type` | `application/json` (Required) |
| `Content-Length` | Required (Must be $\le 8192$ bytes / 8 KB) |
| `Origin` | Optional (Used for dynamic CORS origin matching) |

#### Request Body
```json
{
  "question": "What is vector linking and how does it work?",
  "top_k": 5,
  "min_score": 0.35,
  "no_llm": false,
  "expand_graph": false,
  "max_context_chars": 8000
}
```

| Field | Type | Required | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `question` | `string` | **Yes** | — | Non-empty, $\le 1000$ chars | The natural language question to ask over the corpus. |
| `top_k` | `integer` | No | `5` | $1 \le k \le 20$ | Maximum number of top notes to retrieve for synthesis context. |
| `min_score` | `float` | No | `0.35` | $0.0 \le s \le 1.0$ | Minimum cosine similarity threshold for retrieved items. |
| `no_llm` | `boolean` | No | `false` | `true` \| `false` | When `true`, forces explainable retrieval-only mode without calling Groq. |
| `expand_graph` | `boolean` | No | `false` | `true` \| `false` | When `true`, expands 1st-degree neighbors from `public/graph.json`. |
| `max_context_chars` | `integer` | No | `8000` | $500 \le c \le 32000$ | Maximum character budget for assembled context. |

---

### Response Schemas

#### 1. Success (`200 OK`)
```json
{
  "ok": true,
  "answer": "SecondSelf uses dense 384-dimensional vector embeddings generated with all-MiniLM-L6-v2 to compute cosine similarities between notes. When similarity exceeds the configured threshold, reciprocal links are automatically injected [note-2].",
  "citations": [
    {
      "item_id": "item:note-2",
      "title": "Vector Embeddings & Semantic Search",
      "snippet": "Dense vector representations for nearest-neighbor similarity querying.",
      "score": 0.88,
      "source": "raw/20261123_050700_9f5ff45f.json"
    }
  ],
  "retrieved": [
    {
      "id": "item:note-2",
      "title": "Vector Embeddings & Semantic Search",
      "snippet": "Dense vector representations for nearest-neighbor similarity querying.",
      "content": "...",
      "score": 0.88,
      "category": "Resources",
      "source_type": "markdown",
      "retrieval_method": "vector",
      "tags": ["vector", "embeddings", "search"],
      "file_path": "wiki/Resources/Vector Embeddings & Semantic Search.md"
    }
  ],
  "mode": "llm_synthesis",
  "model": "llama-3.3-70b-versatile",
  "prompt_version": "1.0.0",
  "latency_ms": 342.15,
  "is_grounded": true,
  "uncited_claims_flag": false
}
```

#### 2. Client Error (`400 Bad Request`)
Returned when `question` is missing, parameters are out of range, or JSON is malformed:
```json
{
  "ok": false,
  "error": "Field 'question' must be a non-empty string"
}
```

#### 3. Payload Too Large (`413 Payload Too Large`)
Returned when incoming request body exceeds 8192 bytes:
```json
{
  "ok": false,
  "error": "Payload size (9048 bytes) exceeds maximum limit of 8192 bytes"
}
```

#### 4. Method Not Allowed (`405 Method Not Allowed`)
Returned on GET, PUT, DELETE, etc. Includes `Allow: POST, OPTIONS` header:
```json
{
  "ok": false,
  "error": "Method Not Allowed. Use POST."
}
```

#### 5. Server Error (`500 Internal Server Error`)
Returned on unexpected server failures. Zero secret keys or internal stack traces leak to the client:
```json
{
  "ok": false,
  "error": "Internal server error processing query"
}
```

---

### `OPTIONS /api/ask` (CORS Preflight)
Responds with `204 No Content` and standard CORS headers:
- `Access-Control-Allow-Origin: *` (or matched from `CORS_ORIGINS`)
- `Access-Control-Allow-Methods: POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With`
- `Access-Control-Max-Age: 86400`

---

## 3. Local Testing & Verification

### Running the API Contract Verifier Script
SecondSelf provides a zero-dependency local test script (`scripts/test_api.py`) that binds an ephemeral in-process test server and exercises all 12 contract assertions:

```bash
python scripts/test_api.py
```

### Running with Pytest
Run the comprehensive unit test suite:
```bash
pytest tests/test_ask_api.py -v
```

### Direct `curl` Examples

#### 1. Valid Query (Retrieval-Only Mode)
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does PARA classification work?", "no_llm": true, "top_k": 3}'
```

#### 2. Valid Query with Groq LLM Synthesis
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key architectural decisions in SecondSelf", "expand_graph": true}'
```

#### 3. Testing CORS Preflight
```bash
curl -i -X OPTIONS "http://localhost:8000/api/ask" \
  -H "Origin: https://myapp.vercel.app" \
  -H "Access-Control-Request-Method: POST"
```

#### 4. Testing Payload Size Limit (413)
```bash
python -c "import urllib.request, json; data = json.dumps({'question': 'X', 'pad': 'A'*10000}).encode(); req = urllib.request.Request('http://127.0.0.1:8000/api/ask', data=data, headers={'Content-Type': 'application/json'}); urllib.request.urlopen(req)"
```

---

## 4. Vercel Deployment & Configuration

### `vercel.json` Setup
```json
{
  "version": 2,
  "cleanUrls": true,
  "rewrites": [
    { "source": "/api/ask", "destination": "/api/ask.py" },
    { "source": "/(.*)", "destination": "/public/$1" }
  ],
  "functions": {
    "api/ask.py": {
      "maxDuration": 15,
      "memory": 1024,
      "excludeFiles": "{tests/**,raw/**,wiki/**,verify-artifacts/**,reports/**,evals/**,Docs/**,scripts/**}"
    }
  }
}
```

### Environment Variables
Configure these in the Vercel Project Settings (`Settings -> Environment Variables`):

| Variable | Required | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | **Yes** (in production) | Groq Cloud API key (must start with `gsk_`). |
| `GROQ_SYNTHESIS_MODEL` | No | Groq LLM model (default: `llama-3.3-70b-versatile`). |
| `CORS_ORIGINS` | No | Allowed CORS origins (e.g. `https://secondself.vercel.app`, default: `*`). |
| `ENVIRONMENT` | No | Set to `production` in production deployment. |

### Deploying via Vercel CLI
```bash
# Preview deployment
vercel

# Production deployment
vercel --prod
```

---

## 5. Cold Starts & Serverless Performance

1. **Precomputed Artifacts**: Vector embeddings (`public/index.json`) and the graph projection (`public/graph.json`) are pre-generated during the local build phase. The serverless function never runs heavy PyTorch models or embedding neural networks during cold starts.
2. **Lazy Module Loading**: Heavy synthesis and retrieval packages are imported only inside `do_POST()` to minimize module initialization overhead.
3. **Execution Limits**: Memory is allocated at 1024 MB with a 15-second execution timeout ceiling, comfortably accommodating Groq API network latency (< 1.5 seconds typical).
4. **Bundle Exclusion**: `vercel.json` excludes non-runtime directories (`tests/`, `raw/`, `reports/`, `evals/`, `Docs/`) reducing bundle upload time and disk footprint.
5. **Observability**: Real-time serverless executions can be monitored via `vercel logs <deployment-url>`.
