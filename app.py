import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.document_loaders import TextLoader
from langchain_classic.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.text_splitter import CharacterTextSplitter

from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="Abuelito Sabio", page_icon="👴🏼")

# ======= SIDEBAR =======
with st.sidebar:
    st.image("abuelito-sabio.png", width=200)

    archivo_subido = st.file_uploader(
        "Entrégale un documento al abuelito",
        type=["txt", "pdf"]
    )

    if st.button("Limpiar chat"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.documentos = None

# ======= INICIALIZACIÓN =======
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("👴🏼 Abuelito Sabio")
st.subheader("Tu asistente con sabiduría de biblioteca")

# ======= CARGA Y INDEXACIÓN =======
if archivo_subido:
    # Cargar TXT
    if archivo_subido.type == "text/plain":
        with open("temp.txt", "wb") as f:
            f.write(archivo_subido.getvalue())
        loader = TextLoader("temp.txt")
        documentos = loader.load()

    # Cargar PDF usando pypdf
    else:
        with open("temp.pdf", "wb") as f:
            f.write(archivo_subido.getvalue())

        reader = PdfReader("temp.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        documentos = [{"page_content": text}]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(textos, embeddings)

    st.session_state.vectorstore = vectorstore

    # Resumen automático
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    resumen = llm.invoke({
        "input": "Resume este texto en 3 líneas:\n\n" + documentos[0]["page_content"]
    })
    st.success("📌 Resumen del documento:")
    st.write(resumen["output_text"])

# ======= CHAT =======
if st.session_state.vectorstore:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever()
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if pregunta := st.chat_input("¿Qué quieres saber, hija/o?"):
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            respuesta = qa.invoke(pregunta)

            # Personalidad del abuelito
            respuesta_final = f"Escucha bien, hija/o... {respuesta['result']}\n\n🧓🏼 *Consejo del abuelito:* “No dejes para mañana lo que puedes preguntar hoy.”"
            st.markdown(respuesta_final)

            st.session_state.messages.append({
                "role": "assistant",
                "content": respuesta_final
            })

else:
    st.info("Sube un archivo .txt o .pdf en el panel izquierdo.")
