# streamlit_app.py
# ============================================================
# Smart AI Chatbot - Web App using Streamlit + Groq + Llama 3
# Deploy free on Streamlit Cloud!
# ============================================================

import os
import re
import ast
import operator as op
import requests
import streamlit as st
from groq import Groq
from ddgs import DDGS

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Viraaya",
    page_icon="✨",
    layout="centered"
)

# ---------------------------
# Custom CSS (mobile friendly)
# ---------------------------
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: auto; }
    .user-msg {
        background: #0084ff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .bot-msg {
        background: #f0f0f0;
        color: black;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .clearfix { clear: both; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Setup Groq
# ---------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ No GROQ_API_KEY found! Add it in Streamlit secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)
MODEL = "llama-3.3-70b-versatile"

# ---------------------------
# Session State (Memory)
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "facts" not in st.session_state:
    st.session_state.facts = {}
if "tone" not in st.session_state:
    st.session_state.tone = "neutral"
if "messages" not in st.session_state:
    st.session_state.messages = []  # display messages

# ---------------------------
# Memory helpers
# ---------------------------
FAV_PAT = re.compile(r"my (?:favorite|best)\s+(\w+)\s+is\s+([A-Za-z ]+)", re.I)
LIKE_PAT = re.compile(r"\bi (?:like|love)\s+([A-Za-z ]+)", re.I)

TONE_RULE = {
    "neutral": "Use clear, helpful language.",
    "playful": "Be friendly and fun! Add light energy 😊",
    "formal":  "Be professional and concise.",
}

def store_fact(msg):
    m = FAV_PAT.search(msg)
    if m:
        st.session_state.facts[m.group(1).lower()] = m.group(2).strip(" .!?,")
        return
    m2 = LIKE_PAT.search(msg)
    if m2:
        st.session_state.facts["general_like"] = m2.group(1).strip()

def recall_fact(msg):
    msg_l = msg.lower()
    for key, val in st.session_state.facts.items():
        if key != "general_like" and key in msg_l:
            return f"I remember you told me your favorite {key} is {val} 😊"
    if any(w in msg_l for w in ["like", "love", "favorite", "remember"]) and st.session_state.facts:
        items = [f"{k}: {v}" for k, v in st.session_state.facts.items() if k != "general_like"]
        if items:
            return "Here's what I remember about you: " + ", ".join(items)
    return None

# ---------------------------
# Calculator
# ---------------------------
SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}

def _eval_ast(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return SAFE_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError("Unsupported")

def calculate(expr):
    try:
        expr = expr.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        return f"🧮 Result: {_eval_ast(tree.body)}"
    except:
        return "❌ Could not calculate that safely."

def is_math(q):
    return any(ch.isdigit() for ch in q) and any(s in q for s in "+-*/%()^")

# ---------------------------
# Web Search
# ---------------------------
def web_search(query):
    try:
        with DDGS() as ddg:
            results = list(ddg.text(keywords=query, max_results=3, safesearch="moderate"))
        if results:
            top = results[0]
            return f"{top.get('title','')}: {top.get('body','')}"
    except:
        pass
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '%20')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("extract"):
                return data["extract"].split(". ")[0] + "."
    except:
        pass
    return None

def should_search(q):
    ql = q.lower()
    return any(kw in ql for kw in [
        "search", "find", "latest", "news", "who is", "what is",
        "when did", "where is", "capital of", "define", "meaning of",
        "founder of", "tell me about"
    ])

# ---------------------------
# Groq LLM
# ---------------------------
def ask_groq(user_message, system_extra=""):
    system_prompt = (
        f"You are a smart, helpful and friendly AI assistant.\n"
        f"Tone: {st.session_state.tone}. {TONE_RULE[st.session_state.tone]}\n"
        f"Keep answers clear and concise.\n"
        f"{system_extra}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages += st.session_state.chat_history[-10:]
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Router
# ---------------------------
def set_tone_cmd(q):
    m = re.match(r"\s*tone\s+is\s+(neutral|playful|formal)\s*$", q, re.I)
    if m:
        st.session_state.tone = m.group(1).lower()
        return True
    return False

def is_joke_request(q):
    return any(w in q.lower() for w in ["joke", "funny", "make me laugh"])

def get_joke_topic(q):
    m = re.search(r"joke about (.+)", q, re.I)
    return m.group(1).strip() if m else "anything funny"

def is_summary_request(q):
    return any(w in q.lower() for w in ["summarize", "summary", "hashtags for", "one pager"])

def respond(user_input):
    if set_tone_cmd(user_input):
        return f"✅ Tone updated to: {st.session_state.tone}"

    store_fact(user_input)
    recalled = recall_fact(user_input)
    if recalled:
        return recalled

    if is_math(user_input):
        return calculate(user_input)

    if is_joke_request(user_input):
        topic = get_joke_topic(user_input)
        reply = ask_groq(
            f"Tell me one funny, clean joke about: {topic}. Format: Setup — Punchline.",
            system_extra="You are a witty comedian. Keep jokes clean and short."
        )
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        return f"😂 {reply}"

    if is_summary_request(user_input):
        reply = ask_groq(user_input, system_extra=(
            "Provide:\n1. One-sentence summary (max 18 words)\n"
            "2. Catchy 5-word title\n3. Three hashtags\n"
            "Format: Summary:, Title:, Hashtags:"
        ))
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        return f"📋 {reply}"

    if should_search(user_input):
        search_result = web_search(user_input)
        reply = ask_groq(
            user_input,
            system_extra=f"Use this to help answer:\n{search_result}" if search_result else ""
        )
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        return f"🔍 {reply}"

    reply = ask_groq(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    return reply

# ---------------------------
# UI
# ---------------------------
st.title("✨ Viraaya")
st.caption("Your personal AI assistant | Chat • Search • Calculator • Jokes • Memory")

# Tone selector in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_tone = st.selectbox("🎨 Tone", ["neutral", "playful", "formal"])
    st.session_state.tone = selected_tone

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.facts = {}
        st.rerun()

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        reply = respond(user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
