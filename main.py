from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

# Enable CORS for your WordPress
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary memory store
conversations = {}

@app.get("/")
def root():
    return {"status": "FastAPI running!", "note": "Chat endpoint is /chat (POST only)"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")  # WordPress widget can send a session/user id
    user_message = data.get("message", "")

    # Initialize conversation if new
    if user_id not in conversations:
        conversations[user_id] = [
            {"role": "system", "content": "You are a helpful shopping assistant."}
        ]

    # Add user message
    conversations[user_id].append({"role": "user", "content": user_message})

    # Send to Groq
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": conversations[user_id]
        }
    ).json()

    bot_reply = response["choices"][0]["message"]["content"]

    # Save bot reply to conversation
    conversations[user_id].append({"role": "assistant", "content": bot_reply})

    return {"reply": bot_reply}
