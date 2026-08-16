# Blemish — humaniza tu texto

Pega un texto, elige un tono y recibe una versión **humanizada**: reescrita
para sonar escrita por una persona, con el registro que elijas y sin cambiar
el significado.

## Cómo se usa

1. Pega tu texto en la página (solo pegado: no se suben ni se reciben archivos).
2. Elige el tono de salida: **Formal**, **Científico**, **Tarea** o **Casual**.
3. Pulsa *Humanizar* y copia el resultado.

Sin historial ni conversaciones: un pegado, un resultado. El textarea es
nativo, así que Ctrl+V, Ctrl+Z y todos los atajos de Windows funcionan igual.

## La base de referencia

Blemish humaniza a partir de una **base de textos de referencia**: los
documentos que pongas en `referencia/` (ya creada en el repo, con tres textos
de ejemplo). Ahí dejas los ejemplos cuyo estilo y estructura quieres que
Blemish imite. Genera el índice:

```bash
export GEMINI_API_KEY=...
python ingesta.py               # extrae, trocea, embebe → indice.sqlite
python ingesta.py --solo-probar # ve el troceado sin gastar la API
```

Al humanizar, Blemish busca en esa base los fragmentos más parecidos a tu
texto y los usa como ejemplo de estilo. Si la base está vacía o no hay clave
de Gemini, humaniza igual: con su personalidad, pero sin referencias.

Formatos soportados: PDF, EPUB, DOCX, DOC (requiere LibreOffice), TXT y
Markdown (cada sección `##` es un fragmento). `docs/` también se indexa.

## Requisitos

Dos claves de API gratuitas: `GROQ_API_KEY` (generación) y `GEMINI_API_KEY`
(embeddings de la base de referencia). Se pasan como variables de entorno;
nunca se suben.

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate   # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export GROQ_API_KEY=...
export GEMINI_API_KEY=...
uvicorn main:app --reload    # http://localhost:8000
```

## API

| Endpoint     | Método | Descripción                                            |
|--------------|--------|--------------------------------------------------------|
| `/`          | GET    | Página web (textarea + tono + resultado)               |
| `/health`    | GET    | Healthcheck — `{"status":"ok"}`                        |
| `/humanizar` | POST   | `{"texto": "...", "tono": "casual"}` → texto humanizado|

```bash
curl -X POST http://localhost:8000/humanizar \
  -H "Content-Type: application/json" \
  -d '{"texto": "El presente documento tiene como finalidad...", "tono": "casual"}'
```

Respuesta: `{"texto_humanizado": "...", "tono": "casual", "referencias_usadas": 0, "session_id": "..."}`.
Tonos válidos: `formal`, `cientifico`, `tarea`, `casual` (acepta también
"Científico" con acento y mayúsculas).

El uso está limitado a **4 sesiones simultáneas**; cada una se cierra sola tras
7 minutos sin actividad. Con 4 sesiones activas, el siguiente intento recibe
un 429 y debe esperar a que se libere una. La página gestiona la sesión sola
(`session_id` en localStorage); por API el `session_id` es opcional (si se
omite, el servidor genera uno).

## Despliegue en Render

El repo incluye `render.yaml` (el nombre del servicio aún está pendiente de
renombrar). Render no ejecuta `ingesta.py`: genera `indice.sqlite` en tu
máquina (con tus textos en `referencia/`) y súbelo al repo antes de
desplegar. Las claves se cargan en el
dashboard con `sync: false`.

## Estructura

| Archivo           | Contenido                                              |
|-------------------|--------------------------------------------------------|
| `main.py`         | El humanizador: tonos, referencias, API y página       |
| `ingesta.py`      | Extrae, trocea, embebe y escribe `indice.sqlite`       |
| `personalidad.md` | La voz de Blemish: divertida y sarcástica              |
| `referencia/`     | Tus textos de ejemplo: el corpus de estilo que Blemish imita |
| `docs/`           | Notas sueltas en Markdown (también se indexan)           |
| `indice.sqlite`   | Índice vectorial de la base de referencia              |
| `render.yaml`     | Receta de despliegue para Render                       |
| `requirements.txt`| Dependencias                                           |

## Licencia

PolyForm Noncommercial 1.0.0. Uso, copia y modificación libres para fines no
comerciales; la explotación comercial queda reservada a Mario Aguirre Rivera.
Texto completo en [`LICENSE`](LICENSE).
