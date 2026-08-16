# LLMario — tu biblioteca con personalidad

Un asistente con personalidad propia que convierte tus libros, notas y
documentos en conocimiento consultable. Otras personas le preguntan y
reciben la respuesta con la **fuente**: archivo y, cuando aplica, página o
capítulo.

## El flujo completo

1. Deja tus archivos en `biblioteca/` (libros) y `docs/` (notas sueltas).
2. *(Opcional)* sincroniza `biblioteca/` con Google Drive usando Drive de
   escritorio o `rclone`. El código no depende de Drive: lee la carpeta local.
3. Genera el índice:

   ```bash
   export GEMINI_API_KEY=...
   python ingesta.py               # extrae, trocea, embebe → indice.sqlite
   python ingesta.py --solo-probar # ve el troceado sin gastar la API
   ```

4. Arranca y pregunta:

   ```bash
   export GROQ_API_KEY=...
   uvicorn main:app --reload
   ```

Abre <http://localhost:8000>.

## Formatos soportados

| Formato | Extracción                              | Cita             |
|---------|-----------------------------------------|------------------|
| PDF     | `pypdf`, página por página              | `p. N`           |
| EPUB    | `ebooklib`, por capítulo                | título de cap.   |
| DOCX    | `python-docx`, párrafos y tablas        | documento        |
| DOC     | requiere LibreOffice (se convierte)     | documento        |
| TXT     | nativo                                  | documento        |
| Markdown| nativo; cada sección `##` es un fragmento | sección        |

## Cómo funciona

El agente recibe una pregunta y entra en un bucle de hasta 6 vueltas: el
modelo decide qué herramienta(s) llamar, las ejecuta y vuelve a decidir con
el resultado. Cada respuesta expone la **trayectoria** de llamadas.

Las conversaciones tienen memoria: un `chat_id` identifica la charla y el
agente recuerda los últimos 20 mensajes de esa conversación (persistidos en
`conversaciones.sqlite`).

### Herramientas

| Herramienta       | Qué hace                                                          |
|-------------------|-------------------------------------------------------------------|
| `buscar_en_corpus`| Búsqueda vectorial en tu biblioteca (RAG)                         |
| `calcular_precio` | Desglose de un artículo: base, descuento, subtotal, impuesto, total |
| `consultar_stock` | Existencias de un artículo (simula el inventario)                 |

## Datos

- `indice.sqlite` — SQLite + `sqlite-vec`. Tabla `fragmentos`
  (id, fuente, titulo, pagina, texto) y tabla `vec_fragmentos`
  (embeddings `float[768]`, distancia coseno). Se genera con `ingesta.py`.
- Fragmentos de ~1200 caracteres con 200 de solape.
- Umbral de similitud `0.68` en `main.py` — si el mejor fragmento no lo
  alcanza, el agente admite que la pregunta no está documentada.

## Personalidad

`personalidad.md` define quién es el agente y cómo habla. `negocio.json`
define los artículos para `calcular_precio` y `consultar_stock` (opcional;
si no te interesa vender, déjalo con los ejemplos).

## Stack

- **FastAPI** + **uvicorn** — API y servidor
- **Groq** (`openai/gpt-oss-120b`) — tool calling
- **google-genai** (`gemini-embedding-001`) — embeddings
- **sqlite-vec** — búsqueda vectorial sin cargar todo en RAM
- **pypdf / ebooklib / python-docx / beautifulsoup4** — extracción de texto
- **numpy** — normalización de vectores

## Requisitos

Dos claves de API gratuitas: `GROQ_API_KEY` (Groq) y `GEMINI_API_KEY`
(Google AI Studio). Se pasan como variables de entorno; nunca se suben.

## API

| Endpoint     | Método | Descripción                                         |
|--------------|--------|-----------------------------------------------------|
| `/`          | GET    | Página web de prueba                                |
| `/health`    | GET    | Healthcheck — `{"status":"ok"}`                     |
| `/preguntar` | POST   | Pregunta → respuesta + trayectoria + `chat_id`      |

```bash
curl -X POST http://localhost:8000/preguntar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué dice el libro sobre la fotosíntesis?"}'
```

La respuesta incluye un `chat_id`. Envíalo en la siguiente pregunta para
continuar la misma conversación:

```bash
curl -X POST http://localhost:8000/preguntar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Y qué más?", "chat_id": "1d394c69..."}'
```

Omite `chat_id` (o usa el botón "Nueva conversación" en la página) para
empezar de cero.

## Despliegue en Render

El repo incluye `render.yaml`. En Render: **New + → Blueprint**, conecta el
repositorio y pega las dos claves.

> Render no ejecuta `ingesta.py`. Genera `indice.sqlite` en tu máquina y
> súbelo al repo antes de desplegar. `biblioteca/` está en `.gitignore`
> (los binarios se sincronizan con Drive, no se versionan).

## Estructura

| Archivo             | Contenido                                                  |
|---------------------|------------------------------------------------------------|
| `main.py`           | El agente: contenido, herramientas, bucle y API            |
| `ingesta.py`        | Extrae, trocea, embebe y escribe `indice.sqlite`           |
| `personalidad.md`   | Tu voz: quién es el agente y cómo habla                    |
| `negocio.json`      | Artículos para precio y stock (opcional)                   |
| `biblioteca/`       | Tus libros y documentos (PDF, EPUB, DOCX, DOC, TXT, MD)    |
| `docs/`             | Notas sueltas en Markdown                                  |
| `indice.sqlite`     | Índice vectorial, generado por `ingesta.py`                |
| `conversaciones.sqlite` | Historial de conversaciones, se crea solo por `chat_id` |
| `render.yaml`       | Receta de despliegue para Render                           |
| `requirements.txt`  | Dependencias                                               |

## En el radar (pendiente)

- NotebookLM como capa de lectura (no como backend: no expone API).
- API de Google Drive para ingerir sin carpeta local.
- Chat con historial y memoria de conversación.
- Reranking de resultados con un segundo modelo.
