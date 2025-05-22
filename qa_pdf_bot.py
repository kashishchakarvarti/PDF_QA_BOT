import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# 💰 Exchange rate
USD_TO_INR = 88.0

# 🔐 Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 📄 Load and split PDF
loader = PyPDFLoader("/Users/kashish/Downloads/instruction.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
docs = splitter.split_documents(pages)

# 🧠 Embed and store in Chroma
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")

# 🤖 LLM setup
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

# 📚 Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 🛠️ PDF QA Tool
def answer_from_pdf(query: str) -> str:
    result = qa_chain.invoke({"query": query})
    return result["result"]

pdf_tool = Tool.from_function(
    func=answer_from_pdf,
    name="PDFBot",
    description="Use this to answer questions from the uploaded PDF."
)

# ➗ Calculator Tool
def calculator_function(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

calculator_tool = Tool.from_function(
    func=calculator_function,
    name="calculator_tool",
    description="Use this to evaluate math expressions like '45 * 6'."
)

# 🧰 Combine Tools
tools = [pdf_tool, calculator_tool]

# 🤖 Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🎙️ Voice Config
recognizer = sr.Recognizer()
tts = pyttsx3.init()
tts.setProperty("rate", 160)  # Optional: speed

# Set preferred voice (change "Veena" to any installed voice name)
for voice in tts.getProperty("voices"):
    if "Samantha" in voice.name:  # You can try: Alex, Samantha, etc.
        tts.setProperty("voice", voice.id)
        break

def speak(text):
    print("🤖:", text)
    tts.say(text)
    tts.runAndWait()

# 🔁 Main Loop
print("\n🧠 Assistant Ready (Text + Voice). Say 'exit' to quit.\n")

while True:
    mode = input("Choose input mode [text/voice]: ").strip().lower()

    if mode == "voice":
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("🎙️ Listening...")
            audio = recognizer.listen(source)

        try:
            query = recognizer.recognize_google(audio)
            print("You (voice):", query)
        except sr.UnknownValueError:
            print("❌ Couldn't understand audio.")
            continue
        except sr.RequestError:
            print("⚠️ Speech API error.")
            continue

    elif mode == "text":
        query = input("You (text): ").strip()
    else:
        print("❗ Please enter 'text' or 'voice'.")
        continue

    if query.lower() == "exit":
        speak("Goodbye!")
        break

    with get_openai_callback() as cb:
        response = agent.run(query)
        speak(response)

        usd_cost = cb.total_cost
        inr_cost = usd_cost * USD_TO_INR
        print(f"🧾 Tokens: {cb.total_tokens} | USD: ${usd_cost:.4f} | INR ₹{inr_cost:.2f}")
