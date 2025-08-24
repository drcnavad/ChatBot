from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import os
import re

# Load .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
WC_SITE = os.getenv("WC_SITE")
WC_KEY = os.getenv("WC_KEY")
WC_SECRET = os.getenv("WC_SECRET")

app = FastAPI()

# Enable CORS for WordPress
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with ["https://your-wordpress-site.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "FastAPI running!", "note": "Chat endpoint is /chat (POST only)"}

def get_products(keyword=None):
    url = f"{WC_SITE}/wp-json/wc/v3/products"
    params = {}
    if keyword:
        params["search"] = keyword  # filter products by keyword
    resp = requests.get(url, auth=(WC_KEY, WC_SECRET), params=params)
    if resp.status_code == 200:
        products = resp.json()
        # Convert to format for chat widget
        result = []
        for p in products:
            result.append({
                "title": p["name"],
                "image": p["images"][0]["src"] if p.get("images") else "",
                "price": p["price_html"] if "price_html" in p else p["price"]
            })
        return result
    return []

def get_order_status(order_number):
    url = f"{WC_SITE}/wp-json/wc/v3/orders/{order_number}"
    resp = requests.get(url, auth=(WC_KEY, WC_SECRET))
    if resp.status_code == 200:
        order = resp.json()
        return f"Your order #{order_number} is currently '{order['status']}'."
    else:
        return "Sorry, I could not find your order. Please check the number."

def query_llama(user_message):
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
                    If user asks for product suggestions, return JSON array format:
                    [
                        {"title": "Product Name", "image": "https://link-to-image", "price": "$XX.XX"},
                        {"title": "Another Product", "image": "https://link-to-image", "price": "$YY.YY"}
                    ]
                    For FAQs or general chat, return JSON object:
                    {"reply": "Your text response here"}
                    Do NOT include any explanations outside JSON.
                    """
                },
                {"role": "user", "content": user_message}
            ]
        }
    )

    try:
        data = response.json()
        content = data.choices[0].message.content
        try:
            return JSON_parse_safe(content)
        except:
            return {"reply": content}
    except Exception as e:
        return {"reply": "Sorry, I couldn't process your request."}

def JSON_parse_safe(content):
    import json
    try:
        return json.loads(content)
    except:
        return {"reply": content}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    # Check if the user is asking about order status
    if "order" in user_message.lower() or "track" in user_message.lower():
        match = re.search(r'\b\d{3,10}\b', user_message)  # extract order number
        if match:
            order_number = match.group()
            bot_reply = get_order_status(order_number)
            return {"reply": bot_reply}
        else:
            return {"reply": "Please provide your order number so I can check it."}
        
    # Detect if user is asking for product list
     # 1️⃣ Detect if message is about products
    product_keywords = ["product", "cable", "charger", "phone", "accessory", "phone accessories"]  # add more if needed
    
    if any(word in user_message.lower() for word in product_keywords):
        keyword = user_message  # use the whole message as search keyword
        products = get_products(keyword)
        if products:
            return products  # send product array to frontend
        else:
            return {"reply": f"Sorry, no products found for '{keyword}'."}


    # For everything else, query LLaMA
    bot_reply = query_llama(user_message)
    return bot_reply
