import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File to store sessions
SESSIONS_FILE = "sessions.json"

# Load existing sessions from file
def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}

# Save sessions to file
def save_sessions(sessions):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

# Load into in-memory dict
user_sessions = load_sessions()

# Handle single user's session
def chat_with_user(user_id):
    print(f"\n💬 Chatting with {user_id} — type 'exit' to switch users.")

    if user_id not in user_sessions:
        user_sessions[user_id] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    messages = user_sessions[user_id]

    while True:
        user_input = input(f"{user_id}: ")
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4",  # or gpt-3.5-turbo to save cost
            messages=messages
        )

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        # Save to file after every interaction
        user_sessions[user_id] = messages
        save_sessions(user_sessions)

        print("Assistant:", reply)

# Main loop for switching users
def start_multi_session_chat():
    print("👋 Welcome to Multi-User GPT Chatbot!")
    while True:
        user_id = input("\n🪪 Enter a user ID (or type 'quit' to exit): ")
        if user_id.lower() == "quit":
            break
        chat_with_user(user_id)

# Run the chatbot
start_multi_session_chat()
