# SecondSelf — Vercel Production Deployment Runbook

This runbook documents the complete, reproducible process for deploying **SecondSelf** to **Vercel** as a unified static frontend and Python serverless microservice.

---

## 1. Architecture & Deployment Blueprint

```
+-----------------------------------------------------------------------------------------+
|                                    VERCEL EDGE PROXY                                    |
|   Custom Domain (e.g., secondself.vercel.app)                                            |
|   - Automatic SSL / TLS termination                                                     |
|   - Global CDN edge caching for static assets                                           |
|   - Route Rewrites & Security Headers                                                   |
+-----------------------------------------------------------------------------------------+
                    |                                                   |
    /api/*          |                                                   |  /*
    (Serverless)    v                                                   v  (Static CDN)
+-----------------------------------------+         +-------------------------------------+
|        VERCEL PYTHON SERVERLESS         |         |         STATIC WEB FRONTEND         |
|   api/ask.py (Python 3.12 Runtime)      |         |   public/                               |
|   - Memory: 1024 MB                     |         |   ├── index.html                    |
|   - Max Duration: 15 seconds            |         |   ├── css/main.css                  |
|   - Cold Start: ~2-3 seconds            |         |   ├── js/ (app, search, citations)  |
|   - Warm Latency: < 350 ms              |         |   ├── graph.json (Model Artifact)   |
|   - Read-only precomputed embeddings    |         |   └── index.json (Vector Artifact)  |
+-----------------------------------------+         +-------------------------------------+
                    |
                    v (HTTPS REST API)
+-----------------------------------------+
|             GROQ CLOUD LPU              |
|   llama-3.3-70b-versatile               |
|   - Temperature: 0.0                    |
|   - Grounded context synthesis          |
+-----------------------------------------+
```

---

## 2. Prerequisites & Tooling

Before initiating deployment, ensure you have:
1. **Vercel CLI installed**:
   ```bash
   npm install -g vercel
   vercel --version
   ```
2. **Authenticated Vercel session**:
   ```bash
   vercel login
   ```
3. **Groq Cloud API Key**: Obtain a key starting with `gsk_` from the [Groq Console](https://console.groq.com/keys).
4. **Local Python runtime**: Python 3.10+ (matching the `3.12` pinned in `.python-version`).

---

## 3. Pre-Flight Readiness Verification

Run the automated local verification suite to ensure all artifacts and unit tests pass before shipping:

```bash
# 1. Run full test suite (177 tests)
python -m pytest

# 2. Run API contract verifier (12 checks)
python scripts/test_api.py

# 3. Run frontend quality gate (5 checks)
python scripts/verify_frontend.py

# 4. Run deployment readiness audit
python scripts/verify_deployment_readiness.py
```

---

## 4. End-to-End Deployment Workflow

### Step 1: Link Vercel Project
In the repository root, link your workspace to a Vercel project:
```bash
vercel link
```
*Follow the interactive prompts:*
- Set up and link to project: `Y`
- Scope: Select your Vercel account or team.
- Link to existing project: `N` (or `Y` if already created).
- Project name: `secondself`
- Directory located: `./`

---

### Step 2: Configure Environment Variables
Configure the required environment variables across **Production**, **Preview**, and **Development** scopes:

```bash
# 1. Groq API Key (Sensitive / Secret)
vercel env add GROQ_API_KEY production
# Enter value: gsk_...
vercel env add GROQ_API_KEY preview
# Enter value: gsk_...

# 2. Target Environment
vercel env add ENVIRONMENT production
# Enter value: production
vercel env add ENVIRONMENT preview
# Enter value: preview

# 3. Synthesis Model
vercel env add GROQ_SYNTHESIS_MODEL production
# Enter value: llama-3.3-70b-versatile
vercel env add GROQ_SYNTHESIS_MODEL preview
# Enter value: llama-3.3-70b-versatile

# 4. Permitted CORS Origins
vercel env add CORS_ORIGINS production
# Enter value: * (or your exact production domain)
vercel env add CORS_ORIGINS preview
# Enter value: *
```

*Alternative via Web UI:* Navigate to **Vercel Dashboard → Your Project → Settings → Environment Variables** and enter the variables.

---

### Step 3: Pull Environment Variables Locally
Pull the configured cloud variables to verify local runtime alignment:
```bash
vercel env pull .env.local
```

---

### Step 4: Deploy to Preview Environment
Deploy an isolated preview deployment to test build steps and function bundle sizing:
```bash
vercel deploy
```
*Note the returned Preview URL:* `https://secondself-<hash>-<team>.vercel.app`.

---

### Step 5: Verify Preview Deployment

#### 5.1 Test Frontend Route
```bash
curl -I -s "https://your-preview-url.vercel.app/"
# Verify HTTP 200 OK and presence of static assets
```

#### 5.2 Test Serverless Q&A API
```bash
curl -X POST "https://your-preview-url.vercel.app/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is vector linking?", "no_llm": true, "top_k": 3}'
```
*Expected Response:*
```json
{
  "ok": true,
  "answer": "...",
  "citations": [ ... ],
  "mode": "retrieval_only",
  "latency_ms": 142.5
}
```

#### 5.3 Inspect Preview Build & Runtime Logs
```bash
# Stream live logs
vercel logs <your-preview-url>

# Check for error logs
vercel logs <your-preview-url> --level error
```

---

### Step 6: Production Release
Once preview verification passes, promote the build to production:
```bash
vercel deploy --prod
```
*Production URL is assigned:* `https://secondself.vercel.app`.

---

### Step 7: Live Production Health Verification
```bash
# 1. Probe root route
curl -I "https://secondself.vercel.app/"

# 2. Probe API with LLM synthesis
curl -X POST "https://secondself.vercel.app/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize SecondSelf architecture", "expand_graph": true}'

# 3. Check for any 5xx errors in the last 5 minutes
vercel logs --environment production --status-code 5xx --since 5m
```

---

## 5. Instant Rollback Procedure

If a regression or critical error is detected post-deployment, execute an instant rollback:

### Rollback via CLI
```bash
# 1. List recent deployments to find the previous stable deployment ID/URL
vercel ls

# 2. Execute instant rollback
vercel rollback <previous-deployment-id-or-url>

# 3. Verify rollback status
vercel rollback status
```

### Rollback via Dashboard
1. Open **Vercel Dashboard → Projects → secondself → Deployments**.
2. Locate the last known good deployment.
3. Click the three dots menu (`...`) and select **Promote to Production** (or **Instant Rollback**).
4. Verify traffic shifts in $< 5$ seconds without rebuilding.

---

## 6. Post-Deployment Monitoring & Observability

### Streaming Production Logs
Stream real-time access logs and function executions:
```bash
vercel logs --environment production --follow
```

### Error Tracking
Search for function crashes, timeouts, or unhandled exceptions:
```bash
vercel logs --environment production --level error --since 1h
```

### Key Performance Indicators (KPIs)
- **Status Code 200 Ratio**: Target $\ge 99.5\%$.
- **Serverless Duration**: Target $\le 1200\text{ ms}$ for warm LLM synthesis; $\le 300\text{ ms}$ for retrieval-only.
- **Memory Consumption**: Target $\le 450\text{ MB}$ (well within the $1024\text{ MB}$ allocation).

---

## 7. Cold-Start Characterization & Mitigation

1. **Cold Start Latency**: On Vercel, Python serverless containers go idle after ~15 minutes of inactivity. The first invocation initializes Python and imports dependencies, requiring **2.0 to 3.5 seconds**. Subsequent invocations complete in **150–350 ms**.
2. **Zero In-Memory State Assumption**: The serverless microservice is strictly stateless. Precomputed vector embeddings and graph structures are bundled in `public/` and read into memory per worker.
3. **Optional Warm-up Strategy**:
   - To eliminate cold starts for demos or critical presentations, set up a free scheduled ping via **Vercel Cron** (`vercel.json`) or an external monitor (e.g. UptimeRobot) hitting `POST /api/ask` with `no_llm: true` every 10 minutes:
     ```json
     "crons": [
       {
         "path": "/api/ask",
         "schedule": "*/10 * * * *"
       }
     ]
     ```
   - For standard personal usage, a 2.5-second initial cold start is typical and transparently communicated to the user via the frontend loading spinner.

---

## 8. Common Failure Modes & Quick Fix Matrix

| Symptom / Error | Root Cause | Exact Resolution |
| :--- | :--- | :--- |
| **HTTP 500: Internal Server Error** | Missing or invalid `GROQ_API_KEY` on Vercel. | Run `vercel env add GROQ_API_KEY production` and supply a valid `gsk_...` key. Redeploy. |
| **HTTP 413: Payload Too Large** | Request body exceeded the 8 KB limit. | Shorten query or payload. Max question length is 1000 characters. |
| **HTTP 405: Method Not Allowed** | Client sent a `GET` request to `/api/ask`. | `/api/ask` requires `POST` with a JSON payload. Send `POST` with `Content-Type: application/json`. |
| **Deployment Error: Function size exceeds 50 MB** | Heavy packages (torch, transformers) or test artifacts included in lambda zip. | Verify `requirements.txt` only contains lightweight runtime libraries (`groq`, `numpy`, `pydantic`). Ensure `excludeFiles` in `vercel.json` excludes `tests/**`, `raw/**`, and `reports/**`. |
| **CORS Error in Browser** | Client domain blocked by CORS origin policy. | Set `CORS_ORIGINS=*` or include client origin in Vercel environment variables. |
| **Serverless Function Timeout (504)** | LLM API latency exceeded `maxDuration` (15s). | Ensure Groq client timeout is 15s. If necessary, increase `maxDuration` in `vercel.json` (Pro plan supports up to 60s). |
| **Missing Graph / Index (404 on assets)** | Local build step omitted before deploying. | Run `python build_graph.py` and `python -m link` to regenerate `public/graph.json` and `public/index.json` before running `vercel deploy`. |
| **Wrong Python Version in Logs** | Vercel defaulted to Python 3.9. | Ensure `.python-version` exists in repository root with content `3.12`. |

---

## 9. Post-Deploy Verification Checklist

- [ ] **Frontend CDN Delivery**: `https://secondself.vercel.app/` loads with HTTP 200 OK.
- [ ] **Vis.js Graph Render**: Network graph renders 32 nodes and 72 edges with interactive force simulation.
- [ ] **Q&A Search Submission**: Submitting "What is vector linking?" shows loading state and synthesizes an answer.
- [ ] **Inline Citations**: Synthesized text displays clickable `[1]`, `[2]` badges.
- [ ] **Graph Highlighting**: Cited nodes pulse and scale in size; non-cited nodes dim to 20% opacity.
- [ ] **Reciprocal Focus**: Clicking a citation card centers the graph node and opens the Detail Drawer.
- [ ] **Retrieval-Only Mode**: Checking "Retrieval only" bypasses Groq and answers in $< 200\text{ ms}$.
- [ ] **Defensive Error Handling**: Network disconnect or invalid input displays user-friendly error banner without breaking graph.
- [ ] **Zero 5xx Errors**: `vercel logs --environment production --status-code 5xx --since 5m` reports 0 server errors.
