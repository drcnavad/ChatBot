from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

# Env variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
WC_SITE = os.getenv("WC_SITE")
WC_KEY = os.getenv("WC_KEY")
WC_SECRET = os.getenv("WC_SECRET")

app = FastAPI()

# CORS for your WordPress site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with ["https://your-wordpress-site.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory conversation store (per session if needed)
conversations = {}

@app.get("/")
def root():
    return {"status": "FastAPI running!", "note": "Chat endpoint is /chat (POST only)"}

def get_order_status(order_number):
    url = f"{WC_SITE}/wp-json/wc/v3/orders/{order_number}"
    resp = requests.get(url, auth=(WC_KEY, WC_SECRET))
    if resp.status_code == 200:
        order = resp.json()
        return f"Your order #{order_number} is currently '{order['status']}'."
    return f"Sorry, I could not find order #{order_number}."

def get_products(keyword):
    url = f"{WC_SITE}/wp-json/wc/v3/products"
    params = {"search": keyword}
    resp = requests.get(url, auth=(WC_KEY, WC_SECRET), params=params)
    if resp.status_code == 200:
        products = resp.json()
        result = []
        for p in products:
            result.append({
                "title": p["name"],
                "image": p["images"][0]["src"] if p.get("images") else "",
                "price": p.get("price_html", p.get("price", "N/A"))
            })
        return result
    return []

def query_llama(messages):
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama3-70b-8192",
                "messages": messages
            }
        )
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except:
            return {"reply": content}
    except:
        return {"reply": "Sorry, I couldn't process your request."}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    # Order status
    if "order" in user_message.lower() or "track" in user_message.lower():
        match = re.search(r'\b\d{3,10}\b', user_message)
        if match:
            order_number = match.group()
            return {"reply": get_order_status(order_number)}
        else:
            return {"reply": "Please provide your order number to check status."}

    # Product search
    product_keywords = ["product", "cable", "charger", "phone", "accessory"]
    if any(word in user_message.lower() for word in product_keywords):
        products = get_products(user_message)
        if products:
            return products
        else:
            return {"reply": f"Sorry, no products found for '{user_message}'."}

    # LLaMA fallback (chat)
    messages = [
        {"role": "system", "content": "You are a helpful shopping assistant."},
        {"role": "user", "content": user_message}
    ]
    return query_llama(messages)
