# SecondSelf — Environment & Configuration Architecture

SecondSelf adheres to the **Twelve-Factor App Methodology (Factor III: Config in the Environment)**. Configuration is strictly externalized, validated at startup, and loaded dynamically without hardcoded values.

---

## 1. Configuration Precedence Order

When SecondSelf initializes (via `from config import get_settings`), settings cascade in the following strict order of precedence:

```
[Highest Priority]
       │
       ▼
1. OS / Runtime Environment Variables (`export GROQ_API_KEY=...` or Vercel dashboard)
       │
       ▼
2. Local Targeted Environment File (`.env.local` or `.env.test`)
       │
       ▼
3. General Local Environment File (`.env`)
       │
       ▼
4. Default Values defined in Pydantic Schemas (`config/base.py`)
       │
       ▼
[Lowest Priority]
```

---

## 2. Secrets Management Strategy by Environment

| Environment | Storage Mechanism | Injection Method | Access Boundary |
|:---|:---|:---|:---|
| **Local Development** | `.env.local` / `.env` (gitignored) | Automatically loaded by `config.get_settings()` via `pydantic-settings` | Local developer machine only. |
| **Testing / CI** | `.env.test` + GitHub Repository Secrets | `.env.test` committed for offline tests; GitHub Actions injects `GROQ_API_KEY` for integration tests | CI runner container only. |
| **Staging / Preview** | Vercel Project Environment Variables | Injected into Vercel ephemeral serverless containers (Scoped to `Preview`) | Vercel preview URLs. |
| **Production** | Vercel Encrypted Secrets / Cloud KMS | Injected into production Lambda runtime (Scoped to `Production`) | Production endpoint (`api/ask.py`). |

---

## 3. How to Switch Environments

You can switch runtime environments using the `ENVIRONMENT` environment variable:

```bash
# 1. Run in Test mode (loads .env.test, mocks enabled)
ENVIRONMENT=test pytest

# 2. Run in Local mode (loads .env.local, debug on)
ENVIRONMENT=local python capture.py -t "New idea"

# 3. Simulate Production mode locally (verifies production constraints)
ENVIRONMENT=production python -c "from config import get_settings; print(get_settings().app.environment)"
```

---

## 4. How to Add a New Configuration Variable

Follow this 3-step checklist to introduce a new environment variable:

1. **Declare in `config/base.py`:** Add the field to the appropriate domain model (`AppConfig`, `GroqConfig`, etc.) with types, defaults, and boundary constraints:
   ```python
   class GroqConfig(BaseModel):
       ...
       max_tokens: int = Field(default=1024, ge=128, le=8192, description="Max response tokens")
   ```

2. **Add to `config/settings.py`:** Expose the environment variable key on the parent `Settings` class:
   ```python
   class Settings(BaseSettings):
       ...
       groq_max_tokens: int = Field(default=1024)
   ```

3. **Document in `.env.example`:** Add the variable with type, description, and placeholder value:
   ```bash
   # Maximum response token length for Groq LLM synthesis
   # Type: integer
   # Default: 1024
   GROQ_MAX_TOKENS=1024
   ```
