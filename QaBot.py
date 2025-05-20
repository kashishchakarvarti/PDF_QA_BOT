import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback  # 👈 for cost tracking

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load PDF
loader = PyPDFLoader("/Users/kashish/Downloads/instruction.pdf")
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Embed + store in Chroma
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# ⚡ Switched to GPT-3.5-turbo
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Chat loop
print("\n🤖 Chat with your PDF! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    with get_openai_callback() as cb:  # 🔍 Track usage
        result = qa_chain({"query": query})

        answer = result["result"].strip()
        top_chunk = result["source_documents"][0]
        page = top_chunk.metadata.get("page_label") or top_chunk.metadata.get("page")

        vague_phrases = [
            "doesn't provide specific information",
            "not mentioned",
            "no information available",
            "i don't know",
            "couldn’t find",
            "sorry"
        ]

        print("\n🧠 GPT's Answer:\n" + answer)

        if not any(phrase in answer.lower() for phrase in vague_phrases):
            print(f"📄 Source Page: {page}")

        # 💸 Print token and cost info (converted to INR)
        usd_cost = cb.total_cost
        inr_cost = usd_cost * 88
        print(f"\n🔢 Tokens Used: {cb.total_tokens}")
        print(f"💰 Approx. Cost: ₹{inr_cost:.4f}")
        print("-" * 50)
