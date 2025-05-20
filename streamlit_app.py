import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
import tempfile

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("🤖 Chat with Your PDF")
st.write("Upload a PDF, ask questions, and get AI-powered answers — with page references and live INR cost tracking.")

# Upload PDF file
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if uploaded_file:
    if uploaded_file.type != "application/pdf":
        st.error("❌ Only PDF files are allowed.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name


    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Step 2: Chunk PDF
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Step 3: Embeddings + FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # Step 4: GPT + Retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")  # or "gpt-4"

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Step 5: User input
    question = st.text_input("📝 Ask a question from your PDF:")
    if question:
        with st.spinner("Thinking..."):
            with get_openai_callback() as cb:
                result = qa_chain({"query": question})

            answer = result["result"]
            top_doc = result["source_documents"][0]
            page = top_doc.metadata.get("page_label") or top_doc.metadata.get("page", "?")

            vague_phrases = [
                "no information", "not mentioned", "i don't know",
                "sorry", "not provided", "couldn't find"
            ]

            st.markdown("### 🧠 Answer")
            st.write(answer)

            if not any(v in answer.lower() for v in vague_phrases):
                st.markdown(f"**📄 Source Page:** {page}")

            inr = cb.total_cost * 83.5
            st.markdown(f"💰 Tokens used: `{cb.total_tokens}` — Approx. cost: **₹{inr:.2f}**")
