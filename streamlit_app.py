import os
import tempfile
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import Document

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Voice PDF Assistant", layout="centered")
st.title("📄🎙️ Voice-Powered PDF Assistant")

# Sidebar PDF upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# TTS using gTTS
def speak(text):
    st.markdown(f"**🤖 Response:** {text}")
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format='audio/mp3')

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and embed PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Ensure metadata is safe
    clean_docs = [Document(page_content=d.page_content, metadata=d.metadata or {}) for d in docs]

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Use Chroma in-memory (no persist_directory!)
    vectorstore = Chroma.from_documents(
        clean_docs,
        embedding=embeddings
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    st.success("✅ PDF loaded and indexed.")

    # Mode selector
    mode = st.radio("Choose input mode:", ["Text", "Voice"])

    # Query and response
    query = ""
    if mode == "Text":
        query = st.text_input("Type your question:")
    else:
        if st.button("🎤 Start Recording"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                query = recognizer.recognize_google(audio)
                st.markdown(f"**You asked:** {query}")
            except:
                st.error("❌ Could not recognize speech")

    if query:
        with get_openai_callback() as cb:
            result = qa_chain.invoke({"query": query})
            response = result["result"]
            speak(response)
            inr_cost = cb.total_cost * 88
            st.markdown(f"📊 **Tokens used:** {cb.total_tokens} | **Cost:** ₹{inr_cost:.2f}")
else:
    st.warning("👈 Please upload a PDF to begin.")