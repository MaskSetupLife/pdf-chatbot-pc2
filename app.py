import streamlit as st
import pymongo
import gridfs
from google import genai
from PyPDF2 import PdfReader
import cohere
import time
from streamlit_pdf_viewer import pdf_viewer

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]
COHERE_API_KEY = st.secrets["app"]["COHERE_API_KEY"]
USER = st.secrets["app"].get("USER", "")

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan GOOGLE_API_KEY o MONGODB_URI en secrets")
    st.stop()

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

# MongoDB
client = pymongo.MongoClient(MONGODB_URI)
db = client["pdf_embeddings_db"]
collection = db["pdf_vectors"]
fs = gridfs.GridFS(db)


def crear_indice_vectorial():
    from pymongo.operations import SearchIndexModel
    # Conexión a MongoDB Atlas
    client = pymongo.MongoClient(MONGODB_URI)
    db = client.pdf_embeddings_db
    collection = db.pdf_vectors
    collection.insert_one({"a":"sample"})

    existing_indexes = [idx["name"] for idx in collection.list_search_indexes()]
    if "vector_index" in existing_indexes:
        return

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "similarity": "cosine",
                    "numDimensions": 1024,
                }
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )

    collection.create_search_index(model=search_index_model)
    time.sleep(20)


crear_indice_vectorial()

# =======================
# FUNCIONES PDF + EMBEDDING
# =======================

def leer_pdf(archivo):
    reader = PdfReader(archivo)
    texto = ""
    for page in reader.pages:
        texto += (page.extract_text() or "") + "\n"
    return texto.strip()


def crear_embedding(texto, input_type="search_document"):
    """Genera embeddings usando Cohere v3 (multilenguaje, 1024 dim)."""
    resp = co.embed(
        model="embed-multilingual-v3.0",
        texts=[texto],
        input_type=input_type,
    )
    return resp.embeddings[0]


def procesar_pdf(archivo_pdf, nombre_pdf):
    """Lee PDF, genera embeddings y guarda texto + PDF en MongoDB."""
    st.info("📄 Leyendo PDF...")

    texto = leer_pdf(archivo_pdf)
    if not texto:
        st.error("El PDF no contiene texto.")
        return None

    trozos = [texto[i:i + 1000] for i in range(0, len(texto), 1000)]

    documentos = []
    for i, chunk in enumerate(trozos):
        embedding = crear_embedding(chunk)
        documentos.append({
            "pdf": nombre_pdf,
            "id": i,
            "texto": chunk,
            "embedding": embedding,
        })

    collection.insert_many(documentos)

    # Guardar PDF en GridFS (reemplaza a Backblaze)
    st.info("📤 Guardando PDF en MongoDB...")
    if fs.exists({"filename": nombre_pdf}):
        for f in fs.find({"filename": nombre_pdf}):
            fs.delete(f._id)
    fs.put(archivo_pdf.getvalue(), filename=nombre_pdf, content_type="application/pdf")

    return len(documentos)

# =======================
# VISOR PDF DESDE MONGODB
# =======================

def obtener_pdf(nombre_pdf):
    """Devuelve los bytes del PDF almacenado en GridFS."""
    archivo = fs.find_one({"filename": nombre_pdf})
    return archivo.read() if archivo else None


def mostrar_pdf(pdf_bytes):
    """Muestra PDF embebido con pdf.js (compatible con Chrome)."""
    pdf_viewer(input=pdf_bytes, width=700)

# =======================
# VECTOR SEARCH + CHAT
# =======================

def buscar_similares(embedding, k=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def generar_respuesta(pregunta, contextos):
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""
Usa el contexto para responder la pregunta.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde en español, de forma clara.
"""
    respuesta = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return respuesta.text

# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(page_title="ChatBot", page_icon="📚")
st.title("📚 Chat con PDFs en MongoDB + Gemini + Cohere: " + USER)

archivo_pdf = st.file_uploader("📤 Sube un PDF", type=["pdf"])

if archivo_pdf:
    if st.button("Procesar y guardar PDF"):
        with st.spinner("Procesando PDF..."):
            cantidad = procesar_pdf(archivo_pdf, archivo_pdf.name)
            st.success(f"Procesado: {cantidad} fragmentos generados y PDF guardado.")

        st.info("📖 Vista previa del PDF desde MongoDB:")
        pdf_bytes = obtener_pdf(archivo_pdf.name)
        if pdf_bytes:
            mostrar_pdf(pdf_bytes)

# ---------------- Chat ----------------

st.subheader("💬 Pregunta sobre el contenido del PDF")

if "historial" not in st.session_state:
    st.session_state.historial = []

pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    with st.spinner("Buscando en el PDF..."):
        emb = crear_embedding(pregunta, input_type="search_query")
        similares = buscar_similares(emb)

        if not similares:
            respuesta = "No encontré información relevante."
        else:
            respuesta = generar_respuesta(pregunta, similares)

        st.session_state.historial.append({"rol": "usuario", "texto": pregunta})
        st.session_state.historial.append({"rol": "bot", "texto": respuesta})

for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])
