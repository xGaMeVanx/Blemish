# AGENTS.md — humanizar (Blemish)

**Blemish** is the public/display name of this product; the repo itself is
`humanizar`. Blemish is an LLM that **humanizes text**: the user pastes a
text, picks an output tone (Formal, Científico, Tarea, Casual) and gets a
"humanized" rewrite, drawing on a **reference base** of texts whose style the
model imitates.

The codebase was copied from LLMario (commit `066dc28`) and adapted into
Blemish v1: the business tools (`calcular_precio`, `consultar_stock`),
`negocio.json`, chat history (`conversaciones.sqlite`), and the tool-calling
agent loop are **gone**; the Mario persona was reworked into a fun, sarcastic
voice (`personalidad.md`) and the chat page was replaced by a paste-and-go
UI. The only remaining LLMario trace is `ingesta.py`'s module docstring.

Single Python service: FastAPI app + one LLM call per request + inline HTML
page, all in `main.py`; the reference corpus is built by `ingesta.py` into
`indice.sqlite` (SQLite + sqlite-vec).

## Run / Build / Verify

There is **no test suite, linter, or formatter**. Verification is manual.

```powershell
# Install (PowerShell; README uses bash `export`, which won't work here)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY = "..."; $env:GROQ_API_KEY = "..."

# 1. Build/rebuild the reference index from referencia/ (and docs/)
python ingesta.py                 # needs GEMINI_API_KEY; deletes + rebuilds indice.sqlite
python ingesta.py --solo-probar   # dry run: shows chunking, no API calls

# 2. Run the server
uvicorn main:app --reload         # http://localhost:8000
```

Smoke test: `curl http://localhost:8000/health` → `{"status":"ok"}`; POST to
`/humanizar` with `{"texto": "...", "tono": "casual"}` →
`{"texto_humanizado": "...", "tono": "casual", "referencias_usadas": N, "session_id": "..."}`.

`referencia/` holds three cleaned style texts (`.txt`); run `ingesta.py`
with `GEMINI_API_KEY` to build `indice.sqlite` (absent from this checkout).
The source PDFs were moved to `_originales_referencia/` (gitignored backup).
Without `indice.sqlite`, `/humanizar` works but without style references.

## Architecture

```
personalidad.md ─┐
indice.sqlite  ──┼──► main.py (FastAPI + humanizador) ──► Groq (generación) / Gemini (embeddings)
                 ┘
referencia/ + docs/ ──► ingesta.py ──► indice.sqlite (fragmentos + vec_fragmentos)
```

- **main.py** — the whole server: config constants (including the four tones),
  content loaders, the session gate (max `MAX_SESIONES` concurrent, idle
  timeout), `buscar_referencias` (vector search into the reference base),
  `humanizar` (one Groq call), FastAPI routes, and the inline HTML page
  (`PAGINA`). Loads `personalidad.md` and `indice.sqlite` at import time.
- **ingesta.py** — unchanged from the LLMario base: reads `referencia/` (the
  user's style texts) plus `docs/`; extraction pipeline
  (PDF/EPUB/DOCX/DOC/TXT/MD), chunking (~1200 chars / 200 overlap), Gemini
  embeddings (`gemini-embedding-001`, 768-dim, `RETRIEVAL_DOCUMENT`), writes
  `indice.sqlite`. Unlinks and rebuilds the DB on every run; sleeps 0.25 s
  between embeddings (free-tier rate limit).
- **personalidad.md** — the voice: fun and sarcastic, injected into the system
  prompt. The tone instructions and the humanizing rules live in `main.py`
  (`TONOS`, `REGLAS`).

Data flow: POST `/humanizar` → validate text + tone → session gate (register
or refresh `session_id`; reject with 429 when `MAX_SESIONES` slots are taken)
→ `humanizar` builds the system prompt (persona + rules + tone instructions +
up to `MAX_REFERENCIAS` style fragments fetched by `buscar_referencias`) → one
Groq call (`temperature=TEMPERATURA`, no tools) → `texto_humanizado` returned
with `referencias_usadas` and `session_id`. Sessions expire after
`TIEMPO_INACTIVIDAD_SESION` (7 min) of inactivity. No `chat_id`.

## Tones

The four fixed output tones (`TONOS` in `main.py`); keys are ASCII:

- `formal` — professional, neutral, structured; no colloquialisms.
- `cientifico` — technical, precise, academic vocabulary; impersonal.
- `tarea` — clear and simple, like a well-done homework assignment.
- `casual` — relaxed, colloquial, direct.

Tone input is normalized (lowercase, accents stripped) so "Científico" works.

## Key Files & Directories

- `main.py` — server, humanizer, API, inline UI (single file).
- `ingesta.py` — ingestion pipeline for the reference base (unchanged).
- `personalidad.md` — the sarcastic persona text for the system prompt.
- `requirements.txt` — the single dependency source, all `==`-pinned.
- `render.yaml` — Render blueprint; service name `blemish`.
- `.gitignore` — `.env`/`*.env`, `biblioteca/` and `conversaciones.sqlite`
  (vestigial LLMario lines), and `.codewhale/` (except the versioned
  `.codewhale/constitution.json`). `indice.sqlite` is **not** ignored — the
  README's Render flow requires generating and committing it before deploying.
  `referencia/` is **not** ignored either: the user's texts travel with the
  repo. `_originales_referencia/` (backup of the source PDFs) **is** ignored.
- `referencia/` — the user's style texts: the reference base Blemish imitates
  when humanizing. Holds three cleaned `.txt` files. `docs/` — loose Markdown
  notes, also indexed. `_originales_referencia/` — gitignored backup of the
  original PDFs. `negocio.json` was removed.

## Coding Conventions

- Code, docstrings, comments, commit messages, and UI text are all **Spanish**;
  technical identifiers stay as-is (`buscar_referencias`, `texto_humanizado`).
  Match this.
- Single-file style; sections delimited by wide `# ===` comment banners.
- Missing optional inputs degrade loudly: `indice.sqlite` and both API keys
  print an `ADVERTENCIA:` at import and the app still boots. Missing
  `GROQ_API_KEY` → `/humanizar` returns HTTP 503; missing `GEMINI_API_KEY` or
  an empty corpus → humanizes without references. A missing `personalidad.md`
  falls back silently to a neutral default persona.
- Tunables are constants at the top of `main.py`: `MODELO`,
  `MODELO_EMBEDDINGS`, `DIMENSIONES`, `UMBRAL_SIMILITUD`, `MAX_REFERENCIAS`,
  `TEMPERATURA`, `MAX_SESIONES`, `TIEMPO_INACTIVIDAD_SESION`, `TONOS`.
  `ingesta.py` keeps `MAX_CHAR_FRAGMENTO`, `SOLAPE`.
- No pyproject/lockfile; `requirements.txt` with `==` pins is the single
  dependency source.

## Git Workflow

- Single `main` branch; commits land directly on it. No PRs or feature branches.
- Commit style: `<area>: <description>` in lowercase Spanish, present tense —
  `web: ...`, `persona: ...`, `chore: ...`, `README: ...`. No ticket refs.
- History currently contains only `066dc28 chore: base inicial de humanizar
  copiada de llm`.

## CI/CD

No CI config in the repo. Deployment is Render (`render.yaml`): builds with
`pip install -r requirements.txt`, starts `uvicorn main:app --host 0.0.0.0
--port $PORT`, health check on `/health`. Keys are set in the Render dashboard
(`sync: false` — never commit them; `.env` is gitignored). Render never runs
`ingesta.py`; generate and commit `indice.sqlite` locally before deploying.

## Tips for AI Agents

- **Name state**: public name is **Blemish**; checkout dir is `humanizar`.
  The only remaining LLMario trace is `ingesta.py`'s module docstring. Don't
  rename piecemeal without deciding scope.
- **Never hand-edit** generated data: `indice.sqlite` (rebuild with
  `ingesta.py`). `referencia/` is user-owned input — don't modify, delete, or
  add files there without being asked.
- `indice.sqlite` is loaded once at import; after re-running `ingesta.py`,
  restart the server or the new index won't be visible.
- `.doc` extraction needs LibreOffice (`soffice`) on PATH, else the file is
  skipped with an `AVISO`. `sqlite_vec` needs `enable_load_extension` on the
  connection; a broken install fails at import, not lazily.
- The UI lives inline in `main.py` (`PAGINA`): title, tone buttons, and error
  message text all go there. Keep the input a plain `<textarea>` — the product
  promise is that native shortcuts (Ctrl+Z, Ctrl+V) work untouched; the page
  only intercepts Ctrl+Enter.
- There is **no agent loop, no tools, no chat history**: do not reintroduce
  tool-calling or `conversaciones.sqlite` for Blemish.
- The reference base is optional at runtime: empty/missing corpus or missing
  `GEMINI_API_KEY` → humanize without references. Only `GROQ_API_KEY` is
  required for `/humanizar` to succeed.
- The 4-session gate is **in-memory** (`SESIONES` dict + a lock): it only works
  with a single uvicorn worker. Don't scale to multiple workers without moving
  it to shared storage.
- CORS is `allow_origins=["*"]` on purpose (comment in `main.py`).
- Shell here is PowerShell: use `$env:VAR = "..."`, not `export`.
- Text files are decoded utf-8 then latin-1; Markdown files only contribute
  their `## ` sections as fragments.

## Cache Stability

Keep these byte-stable between edits (config/persona/instructions): `AGENTS.md`,
`README.md`, `render.yaml`, `requirements.txt`, `personalidad.md`. High-churn,
never cache-assume: `indice.sqlite`, `referencia/`, and the end of `main.py`
(the inline HTML page). Append new context instead of reordering the request
preamble.

- **CodeWhale reads this file as:** AGENTS.md (canonical cross-agent project
  instructions). Authority policy goes in `.codewhale/constitution.json`, which
  stays versioned while the rest of `.codewhale/` is gitignored.
