# 🤖 GPT-Powered PDF Chatbot

This terminal-based AI bot lets you chat with any PDF using GPT-3.5 or GPT-4.  
It performs semantic search, generates smart answers, shows source page numbers, and even displays the cost of each question in INR 💸

---

## 🧠 Features

- Chat with any PDF file using natural language
- Fast semantic retrieval using ChromaDB
- Accurate answers from GPT-3.5 or GPT-4
- Shows the exact page number the answer came from
- Tracks OpenAI token usage and real-time cost in INR
- Keeps your API key safe with `.env`

---

## ⚙️ Setup Instructions (All in one flow)

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-chatbot-gpt.git
cd pdf-chatbot-gpt

STEP 2 

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows


Step 3

pip install -r requirements.txt


Step 4 

cp .env.example .env


Step 5 


loader = PyPDFLoader("instruction.pdf")



Step 6 

python QABot.py




