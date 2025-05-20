import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
import tempfile

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF GPT Chatbot", layout="centered")

st.title("🤖 Chat with Your PDF")
st.write("Upload a PDF, ask questions, and get smart GPT answers — with page references and cost in INR.")

# File upload
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Embed and index
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")  # Change to "gpt-4" if needed

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Chat UI
    question = st.text_input("📝 Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            with get_openai_callback() as cb:
                result = qa_chain({"query": question})

            answer = result["result"]
            top_doc = result["source_documents"][0]
            page = top_doc.metadata.get("page_label") or top_doc.metadata.get("page")

            # Detect vague answers
            vague = any(p in answer.lower() for p in [
                "no information", "not mentioned", "i don't know", "sorry", "not provided", "couldn't find"
            ])

            st.markdown("### 🧠 GPT's Answer")
            st.write(answer)

            if not vague:
                st.markdown(f"**📄 Source Page:** {page}")

            # Show cost
            usd = cb.total_cost
            inr = usd * 83.5
            st.markdown(f"💰 Tokens used: `{cb.total_tokens}` — Approx. cost: **₹{inr:.2f}**")

