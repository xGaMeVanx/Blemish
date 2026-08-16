"""
LLMario — un agente con personalidad propia, todo en un archivo.

Estructura, de arriba a abajo:

  1. Configuración: modelo, umbral de similitud, límite de vueltas
  2. Contenido: personalidad.md (tu voz), negocio.json (tus datos) e
     indice.sqlite (tu corpus, generado por ingesta.py desde biblioteca/ y docs/)
  3. Las tres herramientas: buscar_en_corpus, calcular_precio, consultar_stock
  4. El loop del agente, con límite de vueltas y excepciones convertidas en texto
  5. La API: POST /preguntar, GET /health, GET /

Para hacerlo tuyo (ver README.md):
  - personalidad.md  → escribe quién eres y cómo hablas
  - negocio.json     → tus productos o servicios, precios, stock y descuentos
  - biblioteca/      → tus libros y documentos (PDF, EPUB, DOCX, DOC, TXT, MD)
  - docs/*.md        → tus notas sueltas
  - python ingesta.py → genera indice.sqlite desde biblioteca/ y docs/ (usa GEMINI_API_KEY)

Para correrlo:
  export GROQ_API_KEY=...
  export GEMINI_API_KEY=...
  uvicorn main:app --reload
"""

import json
import os
import pathlib
import sqlite3
import threading
import unicodedata
import uuid
from typing import Optional

import numpy as np
import sqlite_vec
from fastapi import FastAPI
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
# Si el mejor fragmento no llega a este parecido, el agente admite que la
# pregunta no está documentada. 0.68 es un punto de partida: mídelo con
# preguntas de tu propio corpus (las de adentro suelen quedar arriba de ~0.70
# y las ajenas abajo de ~0.66) y ajústalo.
UMBRAL_SIMILITUD = 0.68
MAX_VUELTAS = 6
# Cuantos mensajes pasados se recuerdan por conversacion.
MAX_HISTORIAL = 20

REGLAS = (
    "Respondes SIEMPRE en primera persona, con la personalidad descrita arriba. "
    "Usa SIEMPRE las herramientas para datos del negocio o del corpus: nunca "
    "inventes precios, stock ni información que no esté documentada. Si necesitas "
    "varios datos, pide varias herramientas a la vez. Responde en español, "
    "breve y concreto."
)


# =====================================================================================
# 2. CONTENIDO — lo que hace único a este agente
# =====================================================================================

DIRECTORIO = pathlib.Path(__file__).parent

# --- La voz: personalidad.md ------------------------------------------------
ARCHIVO_PERSONALIDAD = DIRECTORIO / "personalidad.md"
if ARCHIVO_PERSONALIDAD.exists():
    PERSONALIDAD = ARCHIVO_PERSONALIDAD.read_text(encoding="utf-8").strip()
else:
    PERSONALIDAD = (
        "No tienes personalidad definida todavía: falta personalidad.md "
        "(ver README.md). Mientras tanto sé neutral y directo."
    )

INSTRUCCIONES = f"{PERSONALIDAD}\n\n{REGLAS}"

# --- El negocio: negocio.json ------------------------------------------------
ARCHIVO_NEGOCIO = DIRECTORIO / "negocio.json"
if ARCHIVO_NEGOCIO.exists():
    try:
        NEGOCIO = json.loads(ARCHIVO_NEGOCIO.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        NEGOCIO = None
        print(f"ADVERTENCIA: negocio.json no es JSON válido: {error}")
else:
    NEGOCIO = None
    print("ADVERTENCIA: falta negocio.json - precio y stock no tienen datos.")

# --- El corpus: indice.sqlite, generado por ingesta.py ---------------------------
ARCHIVO_INDICE = DIRECTORIO / "indice.sqlite"
_db = None
_db_lock = threading.Lock()
if ARCHIVO_INDICE.exists():
    _db = sqlite3.connect(ARCHIVO_INDICE, check_same_thread=False)
    _db.enable_load_extension(True)
    sqlite_vec.load(_db)
    _db.enable_load_extension(False)
else:
    print("ADVERTENCIA: falta indice.sqlite - corre 'python ingesta.py' con tu GEMINI_API_KEY.")

# --- Conversaciones: historial por chat_id, en su propio SQLite ---------------
ARCHIVO_CONVERSACIONES = DIRECTORIO / "conversaciones.sqlite"
_db_conv = None
_db_conv_lock = threading.Lock()
try:
    _db_conv = sqlite3.connect(ARCHIVO_CONVERSACIONES, check_same_thread=False)
    _db_conv.execute(
        "CREATE TABLE IF NOT EXISTS mensajes ("
        " id INTEGER PRIMARY KEY,"
        " chat_id TEXT NOT NULL,"
        " rol TEXT NOT NULL,"
        " contenido TEXT NOT NULL,"
        " creado_en TEXT DEFAULT (datetime('now')))"
    )
    _db_conv.execute("CREATE INDEX IF NOT EXISTS idx_mensajes_chat ON mensajes(chat_id, id)")
    _db_conv.commit()
except sqlite3.Error as error:
    _db_conv = None
    print(f"ADVERTENCIA: no se pudo abrir conversaciones.sqlite: {error}")

# --- Clientes (las claves pueden faltar en desarrollo) ------------------------
try:
    cliente = Groq(api_key=os.environ["GROQ_API_KEY"])
except KeyError:
    cliente = None
    print("ADVERTENCIA: falta GROQ_API_KEY - el agente fallará hasta configurarla.")

try:
    cliente_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    cliente_gemini = None
    print("ADVERTENCIA: falta GEMINI_API_KEY - la búsqueda en el corpus fallará.")


# =====================================================================================
# 3. LAS HERRAMIENTAS
# =====================================================================================


def _normalizar(texto):
    """Minúsculas, sin acentos ni signos: 'Flor de Fuego' → 'flordefuego'."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return "".join(c for c in texto if c.isalnum())


def _buscar_articulo(texto):
    """Encuentra un artículo de negocio.json por clave o nombre, con tolerancia."""
    if not NEGOCIO:
        return None
    buscado = _normalizar(texto)
    for articulo in NEGOCIO.get("articulos", []):
        candidatos = {_normalizar(articulo["clave"]), _normalizar(articulo["nombre"])}
        if any(buscado == c or c in buscado or buscado in c for c in candidatos):
            return articulo
    return None


def _buscar_descuento(texto):
    """Encuentra un descuento de negocio.json por clave o nombre, con tolerancia."""
    if not NEGOCIO:
        return None
    buscado = _normalizar(texto)
    for descuento in NEGOCIO.get("descuentos", []):
        candidatos = {_normalizar(descuento["clave"]), _normalizar(descuento["nombre"])}
        if any(buscado == c or c in buscado or buscado in c for c in candidatos):
            return descuento
    return None


def buscar_en_corpus(consulta):
    """Busca en la biblioteca y devuelve los 3 fragmentos más parecidos."""
    if _db is None or cliente_gemini is None:
        return (
            "Todavía no tengo documentos: corre 'python ingesta.py' con tu "
            "GEMINI_API_KEY para generar indice.sqlite desde biblioteca/."
        )
    respuesta = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDINGS,
        contents=consulta,
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
            "WHERE v.embedding MATCH ? AND k = 3 "
            "ORDER BY v.distance",
            (blob,),
        ).fetchall()

    if not filas:
        return "No encontré información sobre eso en mis documentos."

    if 1 - filas[0][4] < UMBRAL_SIMILITUD:
        return "No encontré información sobre eso en mis documentos."

    partes = []
    for fuente, titulo, pagina, texto, distancia in filas:
        etiqueta = f"{fuente} — {titulo}"
        if pagina:
            etiqueta += f", p. {pagina}"
        partes.append(
            f"[fuente: {etiqueta} | parecido: {1 - distancia:.2f}]\n{texto}"
        )
    return "\n\n".join(partes)


def calcular_precio(articulo, descuento=None):
    """Devuelve el desglose del precio de un artículo o servicio, línea por línea."""
    if NEGOCIO is None:
        return "No tengo datos del negocio todavía: falta negocio.json (ver README.md)."

    encontrado = _buscar_articulo(articulo)
    if encontrado is None:
        lista = ", ".join(
            f"{a['clave']} ({a['nombre']})" for a in NEGOCIO.get("articulos", [])
        )
        lista = lista or "ninguno todavía"
        return f"No conozco '{articulo}'. Artículos disponibles: {lista}."

    simbolo = NEGOCIO.get("simbolo", "$")
    base = encontrado["precio"]
    lineas = [
        f"{encontrado['clave']} — {encontrado['nombre']}",
        f"  {'Precio base':<34} {simbolo}{base:>9,.2f}",
    ]

    monto_descuento = 0.0
    if descuento:
        encontrado_desc = _buscar_descuento(descuento)
        if encontrado_desc is None:
            lista_desc = ", ".join(
                d["nombre"] for d in NEGOCIO.get("descuentos", [])
            )
            lista_desc = lista_desc or "ninguno definido"
            return f"No existe el descuento '{descuento}'. Descuentos disponibles: {lista_desc}."
        monto_descuento = base * encontrado_desc["porcentaje"]
        etiqueta = f"{encontrado_desc['nombre']} ({encontrado_desc['porcentaje']:.0%})"
        lineas.append(f"  {etiqueta:<34}-{simbolo}{monto_descuento:>9,.2f}")

    subtotal = base - monto_descuento
    impuesto_pct = NEGOCIO.get("impuesto", 0.0)
    impuesto = subtotal * impuesto_pct
    total = subtotal + impuesto

    lineas.append(f"  {'Subtotal':<34} {simbolo}{subtotal:>9,.2f}")
    if impuesto_pct:
        lineas.append(
            f"  {'Impuesto ({:.0%})'.format(impuesto_pct):<34} {simbolo}{impuesto:>9,.2f}"
        )
    lineas.append("  " + "-" * 45)
    lineas.append(f"  {'TOTAL A PAGAR':<34} {simbolo}{total:>9,.2f}")
    return "\n".join(lineas)


# En clase esta constante se cambia en vivo para ver fallar al agente.
# En el servidor desplegado se queda en "ok".
ESCENARIO = "ok"  # "ok" | "timeout" | "error500"


def consultar_stock(articulo):
    """Consulta cuántas unidades quedan. Simula el inventario del negocio."""
    if ESCENARIO == "timeout":
        raise TimeoutError("El sistema de inventario no respondió (timeout de 5 s).")
    if ESCENARIO == "error500":
        raise RuntimeError("HTTP 500 del sistema de inventario. Intenta de nuevo.")

    if NEGOCIO is None:
        return "No tengo datos del negocio todavía: falta negocio.json (ver README.md)."

    encontrado = _buscar_articulo(articulo)
    if encontrado is None:
        lista = ", ".join(
            f"{a['clave']} ({a['nombre']})" for a in NEGOCIO.get("articulos", [])
        )
        lista = lista or "ninguno todavía"
        return f"No conozco '{articulo}'. Artículos disponibles: {lista}."

    stock = encontrado.get("stock")
    if stock is None:
        return f"{encontrado['clave']}: siempre disponible."

    total = encontrado.get("total")
    if stock == 0:
        nombre = f"{encontrado['clave']} ({encontrado['nombre']})"
        return f"{nombre}: SIN STOCK por ahora."
    if total:
        return f"{encontrado['clave']}: {stock} disponibles de {total}."
    return f"{encontrado['clave']}: {stock} disponibles."


ESQUEMA_HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_en_corpus",
            "description": (
                "Busca en los documentos propios del negocio (docs/): historia, "
                "servicios, reglas, formas de contacto, preguntas frecuentes. Úsala "
                "para cualquier pregunta de fondo o detalles. No sirve para calcular "
                "precios ni para consultar stock."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": (
                            "Qué quieres buscar, redactado como una frase completa. "
                            "Por ejemplo: 'qué incluye el servicio de mantenimiento'."
                        ),
                    }
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_precio",
            "description": (
                "Calcula el precio de un artículo o servicio con el desglose línea "
                "por línea: base, descuento (si aplica), subtotal, impuesto y total. "
                "Úsala siempre que pregunten un precio o cuánto sale algo. Nunca "
                "calcules de memoria."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "articulo": {
                        "type": "string",
                        "description": "Artículo o servicio a cotizar, por ejemplo 'EJEMPLO-01' o 'producto de ejemplo'.",
                    },
                    "descuento": {
                        "type": "string",
                        "description": (
                            "Descuento a aplicar. Omite este parámetro si la persona "
                            "no menciona ninguno."
                        ),
                    },
                },
                "required": ["articulo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_stock",
            "description": (
                "Consulta cuántas unidades quedan de un artículo o servicio. Es el "
                "único dato que cambia seguido, así que consúltalo siempre que "
                "pregunten si hay disponible o si quedan existencias."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "articulo": {
                        "type": "string",
                        "description": "Artículo a consultar, por ejemplo 'EJEMPLO-01'.",
                    }
                },
                "required": ["articulo"],
            },
        },
    },
]

HERRAMIENTAS = {
    "buscar_en_corpus": buscar_en_corpus,
    "calcular_precio": calcular_precio,
    "consultar_stock": consultar_stock,
}


# =====================================================================================
# 4. EL AGENTE — loop con límite de vueltas y errores convertidos en texto
# =====================================================================================


def _cargar_historial(chat_id, limite=MAX_HISTORIAL):
    """Los ultimos mensajes de una conversacion, del mas viejo al mas nuevo."""
    if _db_conv is None:
        return []
    with _db_conv_lock:
        filas = _db_conv.execute(
            "SELECT rol, contenido FROM mensajes "
            "WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limite),
        ).fetchall()
    return [{"rol": rol, "contenido": contenido} for rol, contenido in reversed(filas)]


def _guardar_mensaje(chat_id, rol, contenido):
    if _db_conv is None:
        return
    with _db_conv_lock:
        _db_conv.execute(
            "INSERT INTO mensajes (chat_id, rol, contenido) VALUES (?, ?, ?)",
            (chat_id, rol, contenido),
        )
        _db_conv.commit()


def correr_agente(pregunta, historial=None):
    """Corre el loop y devuelve (respuesta, trayectoria, vueltas)."""
    if cliente is None:
        raise RuntimeError(
            "Falta GROQ_API_KEY en el entorno. Configúrala y reinicia el servidor."
        )
    mensajes = [{"role": "system", "content": INSTRUCCIONES}]
    for mensaje in historial or []:
        mensajes.append({"role": mensaje["rol"], "content": mensaje["contenido"]})
    mensajes.append({"role": "user", "content": pregunta})
    trayectoria = []
    vuelta = 0

    while vuelta < MAX_VUELTAS:
        vuelta += 1

        respuesta = cliente.chat.completions.create(
            model=MODELO,
            messages=mensajes,
            tools=ESQUEMA_HERRAMIENTAS,
            temperature=0,
            seed=42,
            include_reasoning=False,
        )
        mensaje = respuesta.choices[0].message

        mensajes.append(
            {
                "role": "assistant",
                "content": mensaje.content,
                "tool_calls": [
                    {
                        "id": llamada.id,
                        "type": "function",
                        "function": {
                            "name": llamada.function.name,
                            "arguments": llamada.function.arguments,
                        },
                    }
                    for llamada in (mensaje.tool_calls or [])
                ],
            }
        )

        if not mensaje.tool_calls:
            return mensaje.content, trayectoria, vuelta

        for llamada in mensaje.tool_calls:
            nombre = llamada.function.name

            # Todo lo que pueda tronar, truena aquí adentro y sale como texto:
            # argumentos mal formados, herramienta inexistente, servicio caído.
            try:
                argumentos = json.loads(llamada.function.arguments)
                resultado = HERRAMIENTAS[nombre](**argumentos)
            except Exception as error:
                argumentos = {"_sin_parsear": llamada.function.arguments}
                resultado = f"ERROR al ejecutar {nombre}: {type(error).__name__}: {error}"

            trayectoria.append(
                {"vuelta": vuelta, "herramienta": nombre, "argumentos": argumentos}
            )
            mensajes.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada.id,
                    "name": nombre,
                    "content": resultado,
                }
            )

    # Se acabaron las vueltas: una última llamada sin herramientas, para abstenerse bien.
    mensajes.append(
        {
            "role": "user",
            "content": (
                "Ya no puedes usar más herramientas. Responde con la información que SÍ "
                "lograste obtener y di claramente qué dato no pudiste conseguir y por qué. "
                "No inventes el dato que falta."
            ),
        }
    )
    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        temperature=0,
        seed=42,
        include_reasoning=False,
    )
    return respuesta.choices[0].message.content, trayectoria, vuelta


# =====================================================================================
# 5. LA API
# =====================================================================================

app = FastAPI(title="LLMario")

# Abierto a propósito: el cliente de Streamlit corre en localhost, en otra máquina.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Peticion(BaseModel):
    pregunta: str
    chat_id: Optional[str] = None


class Respuesta(BaseModel):
    respuesta: str
    trayectoria: list
    vueltas: int
    chat_id: str


@app.post("/preguntar", response_model=Respuesta)
def preguntar(peticion: Peticion):
    """Pregunta con memoria: con chat_id continúa la conversación; sin él, crea una."""
    chat_id = peticion.chat_id or uuid.uuid4().hex
    historial = _cargar_historial(chat_id)
    contenido, trayectoria, vueltas = correr_agente(peticion.pregunta, historial)
    _guardar_mensaje(chat_id, "user", peticion.pregunta)
    _guardar_mensaje(chat_id, "assistant", contenido)
    return Respuesta(respuesta=contenido, trayectoria=trayectoria, vueltas=vueltas, chat_id=chat_id)


@app.get("/health")
def health():
    """Render pega aquí para saber si el servicio sigue vivo."""
    return {"status": "ok"}


PAGINA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLMario 💀</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; height: 100vh; margin: 0;
         display: flex; flex-direction: column; }
  h1 { font-size: 1.4rem; color: #e52521; text-align: center; margin: 1rem 0 .5rem; }
  #chat { flex: 1; overflow-y: auto; padding: 0 1rem 1rem; display: flex;
          flex-direction: column; gap: .6rem; max-width: 46rem; width: 100%;
          margin: 0 auto; }
  .msg { max-width: 85%; padding: .6rem .9rem; border-radius: 1rem;
         white-space: pre-wrap; line-height: 1.5; }
  .msg.user { align-self: flex-end; background: #e52521; color: #fff;
              border-bottom-right-radius: .25rem; }
  .msg.asistente { align-self: flex-start; background: #8882;
                   border-bottom-left-radius: .25rem; }
  .msg.pensando { align-self: flex-start; opacity: .6; font-style: italic;
                  background: transparent; }
  form { display: flex; gap: .5rem; padding: 1rem; max-width: 46rem;
         width: 100%; margin: 0 auto; }
  input { flex: 1; padding: .75rem; font-size: 1rem; border: 1px solid #8888;
          border-radius: 1.5rem; background: transparent; color: inherit; }
  button { padding: .75rem 1.4rem; font-size: 1rem; border: 0; border-radius: 1.5rem;
           background: #e52521; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
</style>
</head>
<body>
  <h1>LLMario 💀</h1>

  <div id="chat"></div>

  <form id="formulario">
    <input id="pregunta" autofocus autocomplete="off"
           placeholder="Pregúntale a Mario: ¿qué dice el libro sobre la fotosíntesis?">
    <button id="boton">Preguntar</button>
  </form>

<script>
const formulario = document.getElementById("formulario");
const entrada = document.getElementById("pregunta");
const boton = document.getElementById("boton");
const chat = document.getElementById("chat");
let chatId = null;

function agregarMensaje(clase, texto) {
  const div = document.createElement("div");
  div.className = "msg " + clase;
  div.textContent = texto;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

formulario.addEventListener("submit", async (evento) => {
  evento.preventDefault();
  const pregunta = entrada.value.trim();
  if (!pregunta) return;

  agregarMensaje("user", pregunta);
  entrada.value = "";
  boton.disabled = true;
  const pensando = agregarMensaje("pensando", "Pensando...");

  try {
    const peticion = await fetch("/preguntar", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(chatId ? {pregunta, chat_id: chatId} : {pregunta})
    });
    if (!peticion.ok) throw new Error("El servidor respondió " + peticion.status);
    const datos = await peticion.json();
    chatId = datos.chat_id;
    pensando.remove();
    agregarMensaje("asistente", datos.respuesta);
  } catch (error) {
    pensando.remove();
    agregarMensaje("asistente", "Aún no tengo datos para responderte sobre ese tema, ¡pero! cuando actualice mi información podremos platicar ampliamente. Por ahora soy una versión bb de lo que voy a llegar a ser. 😅");
  } finally {
    boton.disabled = false;
    entrada.focus();
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def inicio():
    """Una página mínima para probar el agente desde el navegador, sin instalar nada."""
    return PAGINA
