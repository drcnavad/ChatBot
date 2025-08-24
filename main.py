from fastapi import FastAPI, Request
import requests
from dotenv import load_dotenv
import os

# Load .env automatically
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": user_message}
            ]
        }
    )
    return response.json()
