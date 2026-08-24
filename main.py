"""
Copyright Mario Aguirre Rivera. Licenciado bajo PolyForm Noncommercial 1.0.0
(https://polyformproject.org/licenses/noncommercial/1.0.0). Ver LICENSE.

Blemish — un humanizador de texto con personalidad, todo en un archivo.

El usuario pega un texto, elige un tono de salida (Formal, Científico, Tarea
o Casual) y recibe una versión "humanizada": reescrita para sonar escrita por
una persona, con el registro pedido y sin cambiar el significado.

Estructura, de arriba a abajo:

  1. Configuración: modelo, umbral de similitud, temperatura y tonos
  2. Contenido: personalidad.md (la voz) e indice.sqlite (la base de
     referencia, generada por ingesta.py desde referencia/ y docs/)
  3. Referencias: búsqueda vectorial de fragmentos de estilo para imitar
  4. Sesiones: máximo 4 concurrentes, con cierre por inactividad
  5. El humanizador: una sola llamada al modelo, sin herramientas ni historial
  6. La API: POST /humanizar, GET /health, GET /

Para correrlo:
  $env:GROQ_API_KEY = "..."; $env:GEMINI_API_KEY = "..."
  uvicorn main:app --reload
"""

import os
import pathlib
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from typing import Optional

import numpy as np
import sqlite_vec
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from groq import Groq
from pydantic import BaseModel

# =====================================================================================
# 1. CONFIGURACIÓN
# =====================================================================================

MODELO = "openai/gpt-oss-120b"
MODELO_EMBEDDINGS = "gemini-embedding-001"
DIMENSIONES = 768
# Parecido mínimo para tomar un fragmento como referencia de estilo. La base
# es un corpus de estilo, no de respuestas: un umbral bajo alcanza.
UMBRAL_SIMILITUD = 0.55
MAX_REFERENCIAS = 5
# Un poco alto a propósito: con temperatura 0 el texto sale plano y "de IA",
# justo lo que este producto quiere evitar.
TEMPERATURA = 0.7

# Los cuatro tonos de salida. Las claves van sin acentos (científico →
# cientifico) porque también son valores de API; las etiquetas van con acentos.
TONOS = {
    "formal": {
        "etiqueta": "Formal",
        "instruccion": (
            "Registro formal: profesional, neutro y estructurado. Vocabulario "
            "cuidado y preciso, oraciones completas, sin coloquialismos, "
            "contracciones ni muletillas."
        ),
    },
    "cientifico": {
        "etiqueta": "Científico",
        "instruccion": (
            "Registro científico: técnico y preciso, con vocabulario académico "
            "del tema. Impersonal (se observa, se concluye, se demuestra), sin "
            "adornos, sin opiniones personales y sin coloquialismos."
        ),
    },
    "tarea": {
        "etiqueta": "Tarea",
        "instruccion": (
            "Como una tarea escolar bien hecha: claro, ordenado y directo, con "
            "vocabulario accesible. Suena a estudiante que entiende el tema, "
            "sin tecnicismos innecesarios ni florituras."
        ),
    },
    "casual": {
        "etiqueta": "Casual",
        "instruccion": (
            "Registro casual: relajado y coloquial, como si le contaras el "
            "tema a un amigo. Contracciones y frases cortas con naturalidad, "
            "sin vulgaridades."
        ),
    },
}

REGLAS = (
    "Tu trabajo es humanizar texto: reescríbelo para que suene escrito por "
    "una persona real, conservando TODO el significado, los datos, los "
    "nombres y las ideas originales. Nunca inventes información, no cambies "
    "el sentido y no dejes fuera contenido importante. La respuesta es SOLO "
    "el texto humanizado: sin títulos, sin comentarios, sin explicaciones, "
    "sin markdown (nada de #, *, **, `, - ni enlaces) y sin repetir el texto "
    "original. Párrafos separados por una línea en blanco y sin dobles "
    "espacios."
)

# --- Sesiones: tope de uso y cierre por inactividad --------------------------
MAX_SESIONES = 4
TIEMPO_INACTIVIDAD_SESION = 7 * 60  # segundos sin actividad antes de liberar el cupo


# =====================================================================================
# 2. CONTENIDO — la voz y la base de referencia
# =====================================================================================

DIRECTORIO = pathlib.Path(__file__).parent

# --- La voz: personalidad.md -------------------------------------------------
ARCHIVO_PERSONALIDAD = DIRECTORIO / "personalidad.md"
if ARCHIVO_PERSONALIDAD.exists():
    PERSONALIDAD = ARCHIVO_PERSONALIDAD.read_text(encoding="utf-8").strip()
else:
    PERSONALIDAD = (
        "Eres Blemish, un humanizador de textos con humor seco y sarcasmo "
        "cariñoso: no te tomas en serio, pero tu trabajo sí. Reescribes "
        "textos para que suenen humanos."
    )

# --- La base de referencia: indice.sqlite, generado por ingesta.py -------------
ARCHIVO_INDICE = DIRECTORIO / "indice.sqlite"
_db = None
_db_lock = threading.Lock()
if ARCHIVO_INDICE.exists():
    _db = sqlite3.connect(ARCHIVO_INDICE, check_same_thread=False)
    _db.enable_load_extension(True)
    sqlite_vec.load(_db)
    _db.enable_load_extension(False)
else:
    print(
        "ADVERTENCIA: falta indice.sqlite - la base de referencia estará "
        "vacía. Corre 'python ingesta.py' con tu GEMINI_API_KEY."
    )

# --- Clientes (las claves pueden faltar en desarrollo) ------------------------
try:
    cliente = Groq(api_key=os.environ["GROQ_API_KEY"])
except KeyError:
    cliente = None
    print("ADVERTENCIA: falta GROQ_API_KEY - /humanizar fallará hasta configurarla.")

try:
    cliente_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    cliente_gemini = None
    print("ADVERTENCIA: falta GEMINI_API_KEY - no habrá textos de referencia.")


# =====================================================================================
# SESIONES — tope de MAX_SESIONES concurrentes, cierre por inactividad
# =====================================================================================

SESIONES = {}  # session_id -> timestamp de última actividad
_sesiones_lock = threading.Lock()


def _permitir_sesion(session_id):
    """Registra (o actualiza) una sesión. Devuelve (permitida, mensaje de rechazo)."""
    ahora = time.time()
    with _sesiones_lock:
        # Libera los cupos de las sesiones que llevan demasiado sin actividad.
        for sid in list(SESIONES):
            if ahora - SESIONES[sid] > TIEMPO_INACTIVIDAD_SESION:
                del SESIONES[sid]

        if session_id in SESIONES:
            SESIONES[session_id] = ahora
            return True, ""

        if len(SESIONES) >= MAX_SESIONES:
            return False, (
                f"Máximo {MAX_SESIONES} sesiones activas. Espera a que se "
                "libere una e inténtalo de nuevo."
            )

        SESIONES[session_id] = ahora
        return True, ""


# =====================================================================================
# 3. REFERENCIAS — fragmentos de estilo desde indice.sqlite
# =====================================================================================


def _normalizar_tono(texto):
    """'Científico' → 'cientifico': minúsculas y sin acentos."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def buscar_referencias(texto):
    """Devuelve fragmentos de estilo de la base de referencia, ya formateados."""
    if _db is None or cliente_gemini is None:
        return []
    respuesta = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDINGS,
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONES,
        ),
    )
    vector = np.array(respuesta.embeddings[0].values, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)
    blob = sqlite_vec.serialize_float32(vector)

    with _db_lock:
        filas = _db.execute(
            "SELECT f.fuente, f.titulo, f.pagina, f.texto, v.distance "
            "FROM vec_fragmentos v "
            "JOIN fragmentos f ON f.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance",
            (blob, MAX_REFERENCIAS),
        ).fetchall()

    referencias = []
    for fuente, titulo, pagina, texto_fragmento, distancia in filas:
        if 1 - distancia < UMBRAL_SIMILITUD:
            continue
        etiqueta = f"{fuente} — {titulo}"
        if pagina:
            etiqueta += f", p. {pagina}"
        referencias.append(f"[fuente: {etiqueta}]\n{texto_fragmento}")
    return referencias


# =====================================================================================
# 4. EL HUMANIZADOR — una llamada al modelo, sin herramientas ni historial
# =====================================================================================


def limpiar_texto(texto):
    """Quita restos de markdown y normaliza espacios para pegar en Word."""
    lineas = []
    for linea in texto.split("\n"):
        linea = linea.strip()
        if not linea:
            continue
        # Encabezados, citas y marcadores de lista que el modelo pueda dejar.
        linea = re.sub(r"^(#{1,6})\s+", "", linea)
        linea = re.sub(r"^>\s?", "", linea)
        linea = re.sub(r"^([-*+]|\d{1,3}\.)\s+", "", linea)
        # Reglas horizontales (---, ***, ===) en su propia línea.
        if re.fullmatch(r"[-*_=\s]{3,}", linea):
            continue
        # Negritas (**x**, __x__) y código (`x`): se quita la marca, queda el texto.
        linea = re.sub(r"\*\*|__", "", linea)
        linea = re.sub(r"`([^`]*)`", r"\1", linea)
        # Enlaces [texto](url) → texto.
        linea = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", linea)
        # Énfasis suave (*x*, _x_) solo con límites de espacio, para no tocar
        # "3*4" ni palabras con guion bajo.
        linea = re.sub(r"(?<!\S)\*([^*\n]+)\*(?!\S)", r"\1", linea)
        linea = re.sub(r"(?<!\S)_([^_\n]+)_(?!\S)", r"\1", linea)
        # Dobles espacios y tabuladores → un solo espacio.
        linea = re.sub(r"[ \t]{2,}", " ", linea).strip()
        lineas.append(linea)
    return "\n".join(lineas)


def a_rtf(texto):
    """Texto limpio → RTF con Times New Roman a 12pt, para pegar en .docx."""
    parrafos = [p for p in (p.strip() for p in texto.split("\n")) if p]
    cuerpo = (
        p.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r", "")
        for p in parrafos
    )
    return (
        r"{\rtf1\ansi\deff0{\fonttbl{\f0\froman Times New Roman;}}"
        r"\f0\fs24\lang3082 "
        + r"\par ".join(cuerpo)
        + r"\par}"
    )


def humanizar(texto, tono):
    """Reescribe `texto` en el tono elegido. Devuelve (texto, nº de referencias)."""
    if cliente is None:
        raise RuntimeError(
            "Falta GROQ_API_KEY en el entorno. Configúrala y reinicia el servidor."
        )

    tono = _normalizar_tono(tono)
    if tono not in TONOS:
        raise ValueError(f"Tono desconocido: '{tono}'. Válidos: {', '.join(TONOS)}")

    sistema = (
        f"{PERSONALIDAD}\n\n{REGLAS}\n\n"
        f"Tono de salida elegido por el usuario:\n{TONOS[tono]['instruccion']}"
    )
    referencias = buscar_referencias(texto)
    if referencias:
        sistema += (
            "\n\nTextos de referencia para imitar su estilo (imita el "
            "registro, nunca el contenido):\n\n"
            + "\n\n".join(referencias)
        )

    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": sistema},
            {"role": "user", "content": f"Humaniza este texto:\n\n{texto}"},
        ],
        temperature=TEMPERATURA,
    )
    return limpiar_texto(respuesta.choices[0].message.content), len(referencias)


# =====================================================================================
# 5. LA API
# =====================================================================================

app = FastAPI(title="Blemish")

# Abierto en desarrollo: permite probar la página desde otro puerto o máquina.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Peticion(BaseModel):
    texto: str
    tono: str = "formal"
    session_id: Optional[str] = None


class Respuesta(BaseModel):
    texto_humanizado: str
    texto_rtf: str
    tono: str
    referencias_usadas: int
    session_id: str


@app.post("/humanizar", response_model=Respuesta)
def humanizar_endpoint(peticion: Peticion):
    """Texto pegado → versión humanizada en el tono elegido."""
    if not peticion.texto.strip():
        raise HTTPException(
            status_code=400,
            detail="El texto está vacío: pega algo para humanizar.",
        )

    tono = _normalizar_tono(peticion.tono)
    if tono not in TONOS:
        raise HTTPException(
            status_code=400,
            detail=f"Tono desconocido: '{peticion.tono}'. Válidos: {', '.join(TONOS)}.",
        )

    session_id = peticion.session_id or uuid.uuid4().hex
    permitida, mensaje = _permitir_sesion(session_id)
    if not permitida:
        raise HTTPException(status_code=429, detail=mensaje)

    try:
        texto_humanizado, cuantas = humanizar(peticion.texto, tono)
        texto_humanizado = limpiar_texto(texto_humanizado)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Error al humanizar: {type(error).__name__}: {error}")

    return Respuesta(
        texto_humanizado=texto_humanizado,
        texto_rtf=a_rtf(texto_humanizado),
        tono=tono,
        referencias_usadas=cuantas,
        session_id=session_id,
    )


@app.get("/health")
def health():
    """Render pega aquí para saber si el servicio sigue vivo."""
    return {"status": "ok"}


PAGINA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blemish — humaniza tu texto</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f6f5fb;
    --superficie: #ffffff;
    --borde: #e3e0f0;
    --texto: #211f33;
    --texto-tenue: #6b6880;
    --acento: #7c5cf0;
    --acento-oscuro: #6a48e6;
    --acento-tenue: #f1eefe;
    --error: #d1435b;
    --error-bg: #fdeef1;
  }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; min-height: 100vh; margin: 0;
         background: var(--bg); color: var(--texto);
         display: flex; flex-direction: column; }
  header { max-width: 40rem; width: 100%; margin: 0 auto; padding: 2.25rem 1.25rem .5rem;
            text-align: center; }
  h1 { font-size: 1.9rem; margin: 0 0 .3rem; letter-spacing: -.02em; color: var(--acento-oscuro); }
  header p { margin: 0; color: var(--texto-tenue); }
  main { flex: 1; max-width: 40rem; width: 100%; margin: 0 auto;
         padding: 1.25rem 1.25rem 3rem; display: flex; flex-direction: column; }
  .columnas { display: flex; flex-direction: column; gap: 1rem; }
  .tarjeta { background: var(--superficie); border: 1px solid var(--borde);
             border-radius: 1rem; padding: 1.25rem;
             box-shadow: 0 1px 2px rgba(33, 31, 51, .04), 0 8px 24px rgba(33, 31, 51, .05);
             display: flex; flex-direction: column; gap: 1rem; }
  textarea { width: 100%; min-height: 9rem; resize: vertical; padding: .85rem;
             font: inherit; line-height: 1.55; border: 1px solid var(--borde);
             border-radius: .75rem; background: var(--bg); color: inherit; }
  #entrada, #salida { flex: 1; }
  textarea:focus { outline: none; border-color: var(--acento); background: var(--superficie);
                   box-shadow: 0 0 0 3px var(--acento-tenue); }
  .tonos { display: flex; flex-wrap: wrap; gap: .5rem; }
  .tono { padding: .45rem 1.05rem; border: 1px solid var(--borde); border-radius: 1.5rem;
          background: var(--superficie); color: var(--texto-tenue); cursor: pointer;
          font: inherit; font-size: .9rem; transition: background .15s, border-color .15s, color .15s; }
  .tono:hover { border-color: var(--acento); color: var(--acento-oscuro); }
  .tono.activo { background: var(--acento); border-color: var(--acento); color: #fff; }
  #humanizar { padding: .75rem 1.5rem; border: 0; border-radius: .75rem;
               background: var(--acento); color: #fff; cursor: pointer;
               font: inherit; font-size: 1rem; font-weight: 600;
               transition: background .15s; }
  #humanizar:hover:not(:disabled) { background: var(--acento-oscuro); }
  #humanizar:disabled { opacity: .5; cursor: wait; }
  #error { display: none; color: var(--error); background: var(--error-bg);
           border-radius: .6rem; margin: 0; padding: .6rem .85rem;
           white-space: pre-wrap; font-size: .9rem; }
  #copiar { align-self: flex-start; padding: .5rem 1.1rem; border-radius: .75rem;
            border: 1px solid var(--acento); background: var(--superficie);
            color: var(--acento-oscuro); cursor: pointer; font: inherit;
            font-size: .9rem; transition: background .15s; }
  #copiar:hover { background: var(--acento-tenue); }
  .oculto { display: none !important; }

  @media (min-width: 860px) {
    header, main { max-width: 68rem; }
    .columnas { flex-direction: row; align-items: stretch; }
    .columnas > .tarjeta { flex: 1 1 0; min-width: 0; }
    textarea { min-height: 22rem; }
  }
</style>
</head>
<body>
  <header>
    <h1>Blemish</h1>
    <p>Pega tu texto, elige un tono y dale vida.</p>
  </header>
  <main>
    <div class="columnas">
      <div class="tarjeta">
        <textarea id="entrada" placeholder="Pega aquí el texto que quieres humanizar..."></textarea>
        <div class="tonos" id="tonos">
          <button type="button" class="tono activo" data-tono="formal">Formal</button>
          <button type="button" class="tono" data-tono="cientifico">Científico</button>
          <button type="button" class="tono" data-tono="tarea">Tarea</button>
          <button type="button" class="tono" data-tono="casual">Casual</button>
        </div>
        <button id="humanizar" type="button">Humanizar</button>
        <p id="error"></p>
      </div>
      <div class="tarjeta">
        <textarea id="salida" readonly placeholder="Tu texto humanizado aparecerá aquí..."></textarea>
        <button id="copiar" type="button" class="oculto">Copiar resultado (Times New Roman)</button>
      </div>
    </div>
  </main>
<script>
const entrada = document.getElementById("entrada");
const tonos = document.getElementById("tonos");
const boton = document.getElementById("humanizar");
const salida = document.getElementById("salida");
const copiar = document.getElementById("copiar");
const error = document.getElementById("error");
let tono = "formal";
let ultimoRtf = null;
let sessionId = localStorage.getItem("blemish_sesion") ||
                (crypto.randomUUID ? crypto.randomUUID() : (Date.now().toString(36) + Math.random().toString(36).slice(2)));
localStorage.setItem("blemish_sesion", sessionId);

tonos.addEventListener("click", (evento) => {
  const botonTono = evento.target.closest(".tono");
  if (!botonTono) return;
  tonos.querySelectorAll(".tono").forEach((b) => b.classList.remove("activo"));
  botonTono.classList.add("activo");
  tono = botonTono.dataset.tono;
});

async function humanizar() {
  const texto = entrada.value.trim();
  if (!texto) {
    error.textContent = "Pega un texto primero.";
    error.style.display = "block";
    return;
  }
  error.style.display = "none";
  salida.value = "";
  copiar.classList.add("oculto");
  boton.disabled = true;
  try {
    const peticion = await fetch("/humanizar", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({texto, tono, session_id: sessionId})
    });
    const datos = await peticion.json();
    if (!peticion.ok) {
      const e = new Error(datos.detail || ("El servidor respondió " + peticion.status));
      e.status = peticion.status;
      throw e;
    }
    salida.value = datos.texto_humanizado;
    ultimoRtf = datos.texto_rtf || null;
    copiar.classList.remove("oculto");
  } catch (err) {
    error.textContent = (err.status === 429)
      ? err.message
      : "Algo salió mal: " + err.message;
    error.style.display = "block";
  } finally {
    boton.disabled = false;
  }
}

boton.addEventListener("click", humanizar);
entrada.addEventListener("keydown", (evento) => {
  // Ctrl+Enter también humaniza; el resto de atajos (Ctrl+Z, Ctrl+V…) quedan nativos.
  if ((evento.ctrlKey || evento.metaKey) && evento.key === "Enter") humanizar();
});

copiar.addEventListener("click", async () => {
  try {
    if (ultimoRtf && navigator.clipboard.write && window.ClipboardItem) {
      // Copia con formato RTF (Times New Roman) para pegar en Word, con
      // texto plano de respaldo para cualquier otro destino.
      const rtf = new Blob([ultimoRtf], {type: "text/rtf"});
      const plano = new Blob([salida.value], {type: "text/plain"});
      await navigator.clipboard.write([
        new ClipboardItem({"text/rtf": rtf, "text/plain": plano})
      ]);
    } else {
      await navigator.clipboard.writeText(salida.value);
    }
  } catch {
    salida.select();
    document.execCommand("copy");
  }
  copiar.textContent = "¡Copiado!";
  setTimeout(() => { copiar.textContent = "Copiar resultado (Times New Roman)"; }, 1500);
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def inicio():
    """La página: pegar texto → elegir tono → humanizar."""
    return PAGINA
