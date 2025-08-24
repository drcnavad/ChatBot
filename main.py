from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import os

# Load .env automatically
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

# Enable CORS so your WordPress site can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing, allow all; later replace "*" with ["https://your-wordpress-site.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "FastAPI running!", "note": "Chat endpoint is /chat (POST only)"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    # Call Groq API
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": """
                    You are a helpful shopping assistant.
                    
                    - If user asks for product suggestions, return JSON array format:
                      [
                        {"title": "Product Name", "image": "https://link-to-image", "price": "$XX.XX"},
                        {"title": "Another Product", "image": "https://link-to-image", "price": "$YY.YY"}
                      ]
                    
                    - If user asks for anything else (like greetings, questions, order status, etc.),
                      return JSON object format:
                      {"reply": "Your text response here"}
                      
                    Do NOT include explanations outside JSON. Only valid JSON output is allowed.
                    """
                },
                {"role": "user", "content": user_message}
            ]
        }
    )

    try:
        return response.json()
    except Exception as e:
        return {"reply": "Sorry, I couldn't process the request.", "error": str(e)}
