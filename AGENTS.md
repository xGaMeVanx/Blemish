# AGENTS.md — LLMario

A Spanish-language RAG chatbot with a fixed persona ("Mario") that answers
questions from a personal document library, citing source, page, or chapter.
Single Python service: FastAPI app + tool-calling agent loop + embedded chat
page, all in `main.py`; the search corpus is built by `ingesta.py` into
`indice.sqlite` (SQLite + sqlite-vec).

## Run / Build / Verify

There is **no test suite, linter, or formatter**. Verification is manual.

```powershell
# Install (PowerShell; README uses bash `export`, which won't work here)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY = "..."; $env:GROQ_API_KEY = "..."

# 1. Build/rebuild the search index from biblioteca/ and docs/
python ingesta.py                 # needs GEMINI_API_KEY; rebuilds indice.sqlite from scratch
python ingesta.py --solo-probar   # dry run: shows chunking, no API calls

# 2. Run the server
uvicorn main:app --reload         # http://localhost:8000
```

Smoke test: `curl http://localhost:8000/health` → `{"status":"ok"}`; POST to
`/preguntar` with `{"pregunta": "..."}`.

## Architecture

```
personalidad.md ─┐
negocio.json   ──┼──► main.py (FastAPI + agent loop) ──► Groq (tool calls) / Gemini (embeddings)
indice.sqlite  ─┘
biblioteca/ + docs/ ──► ingesta.py ──► indice.sqlite (fragmentos + vec_fragmentos)
```

- **main.py** — the whole server: config constants, the three tools, the agent
  loop (`correr_agente`), FastAPI routes, and the inline HTML chat page
  (`PAGINA`). Loads `personalidad.md`, `negocio.json`, and `indice.sqlite` at
  import time.
- **ingesta.py** — ingestion pipeline: extract text from PDF/EPUB/DOCX/DOC/TXT/MD
  in `biblioteca/` and `docs/`, chunk (~1200 chars, 200 overlap), embed with
  Gemini (`gemini-embedding-001`, 768-dim), write to `indice.sqlite` (cosine,
  sqlite-vec). Deletes and rebuilds the DB on every run.
- **personalidad.md** — the persona text injected into the system prompt.
- **negocio.json** — articles, discounts, tax for `calcular_precio` /
  `consultar_stock`.
- **biblioteca/**, **docs/** — source documents. `biblioteca/` is gitignored
  (synced from Google Drive); `docs/` holds loose Markdown notes.
- **render.yaml**, **requirements.txt** — Render deployment config; deps
  pinned with `==`.

Data flow: POST `/preguntar` → `correr_agente` builds messages from
`INSTRUCCIONES` + last 20 messages of the `chat_id` history
(`conversaciones.sqlite`) → agent loop (max 6 turns) where Groq
(`openai/gpt-oss-120b`) picks tools → tool results (or exception text) go back
into the conversation → final answer returned with `trayectoria` (tool call
log), `vueltas`, and `chat_id`.

## Tools

- `buscar_en_corpus(consulta)` — vector search, top-3 fragments, gated by
  `UMBRAL_SIMILITUD = 0.68` in `main.py`; below it, the agent says the question
  isn't documented.
- `calcular_precio(articulo, descuento?)` — line-item price breakdown from
  `negocio.json`.
- `consultar_stock(articulo)` — stock lookup. `ESCENARIO` constant simulates
  timeout / HTTP-500 failures for demos; keep it `"ok"` when deployed.

## Conventions

- Code, docstrings, comments, commit messages, and UI text are all **Spanish**;
  technical identifiers stay as-is (`buscar_en_corpus`, `chat_id`). Match this.
- Single-file style; sections delimited by wide `# ===` comment banners.
- Tool errors never raise: the agent loop catches everything per tool call and
  feeds `"ERROR al ejecutar <nombre>: <type>: <msg>"` back to the model.
- Missing optional inputs degrade loudly: `personalidad.md`, `negocio.json`,
  `indice.sqlite`, and both API keys print an `ADVERTENCIA:` at import, and the
  app still boots; only the affected path fails at request time.
- Tunables are constants at the top of `main.py`: `MODELO`, `MODELO_EMBEDDINGS`,
  `UMBRAL_SIMILITUD`, `MAX_VUELTAS`, `MAX_HISTORIAL`.
- No pyproject/lockfile; `requirements.txt` with `==` pins is the single
  dependency source.

## Git Workflow

- Single `main` branch; commits land directly on it. No PRs or feature branches.
- Commit style: `<area>: <description>` in lowercase Spanish, present tense —
  `web: ...`, `persona: ...`, `chore: ...`, `README: ...`. No ticket refs.

## CI/CD

No CI config in the repo. Deployment is Render (`render.yaml`): builds with
`pip install -r requirements.txt`, starts `uvicorn main:app --host 0.0.0.0
--port $PORT`, health check on `/health`. `GROQ_API_KEY` / `GEMINI_API_KEY`
are set in the Render dashboard (`sync: false` — never commit them; `.env` is
gitignored).

## Agent Gotchas

- **Never edit** generated/runtime data: `indice.sqlite` (rebuild with
  `ingesta.py`), `conversaciones.sqlite` (gitignored), and `biblioteca/`
  contents (gitignored, synced from Drive).
- `indice.sqlite` is loaded once at import; after re-running `ingesta.py`,
  restart the server or the new index won't be visible.
- `.doc` extraction needs LibreOffice (`soffice`) on PATH, else the file is
  skipped with an `AVISO`.
- `sqlite_vec` needs `enable_load_extension` on the connection; a broken
  install fails at import, not lazily.
- The chat UI lives inline in `main.py` (the `PAGINA` constant, including the
  `<title>` and error message text) — UI fixes go there, not in a template dir.
- History is server-side only: the localStorage/history-endpoint commit was
  reverted (`3bb0074`). Persist via `chat_id`; don't reintroduce client-side
  storage without checking why it was reverted.
- API keys are read at import; missing keys leave `cliente` / `cliente_gemini`
  as `None` and the relevant tool returns a setup hint instead of crashing.
- CORS is `allow_origins=["*"]` on purpose (comment in `main.py` explains why).
- Shell here is PowerShell: use `$env:VAR = "..."`, not `export`.
- Text files are decoded utf-8 then latin-1; Markdown files only contribute
  their `## ` sections as fragments.

## Cache Stability

Keep these byte-stable between edits (config/persona/instructions): `AGENTS.md`,
`README.md`, `render.yaml`, `requirements.txt`, `personalidad.md`,
`negocio.json`. High-churn, never cache-assume: `indice.sqlite`,
`conversaciones.sqlite`, `biblioteca/`, and the end of `main.py` (the inline
HTML page). Append new context instead of reordering the request preamble.

- **CodeWhale reads this file as:** AGENTS.md (canonical cross-agent project
  instructions). Authority policy goes in `.codewhale/constitution.json`, which
  stays versioned while the rest of `.codewhale/` is gitignored.
