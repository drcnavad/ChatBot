from fastapi import FastAPI, Request
import requests
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware




# Load .env automatically
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

# 👇 Add this right after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or replace "*" with ["https://your-wordpress-site.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root GET route for health check
@app.get("/")
def root():
    return {"status": "FastAPI running!", "note": "Chat endpoint is /chat (POST only)"}

# Chat POST route
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
