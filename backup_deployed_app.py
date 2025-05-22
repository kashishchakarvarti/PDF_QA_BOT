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

# Streamlit page config
st.set_page_config(page_title="PDF Q&A", layout="centered")
st.title("📘 Ask Questions About Your PDF")
st.write("Upload a PDF file and ask any question. You'll get accurate answers and page references instantly.")

# File upload
uploaded_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"])

if uploaded_file:
    if uploaded_file.type != "application/pdf":
        st.error("❌ Only PDF files are allowed.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split into text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    if not docs:
        st.error("❌ No readable text found in the PDF. Please upload a valid text-based PDF.")
        st.stop()

    # Generate embeddings and create vector index
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Language model
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Input question
    question = st.text_input("📝 Ask a question from the document:")
    if question:
        with st.spinner("Searching..."):
            with get_openai_callback() as cb:
                result = qa_chain({"query": question})

            answer = result["result"]
            top_doc = result["source_documents"][0]
            page = top_doc.metadata.get("page_label") or top_doc.metadata.get("page", "?")

            vague_phrases = [
                "no information", "not mentioned", "i don't know",
                "sorry", "not provided", "couldn't find"
            ]

            st.markdown("### 📌 Answer")
            st.write(answer)

            if not any(v in answer.lower() for v in vague_phrases):
                st.markdown(f"**📄 Source Page:** {page}")

            inr = cb.total_cost * 83.5
            st.markdown(f"💰 Tokens used: `{cb.total_tokens}` — Estimated cost: **₹{inr:.2f}**")
