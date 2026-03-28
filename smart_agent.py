# smart_agent.py
# ============================================================
# Combined Smart AI Agent using Groq + Llama 3
# Features:
#   ✅ Intelligent chat (like ChatGPT!)
#   ✅ Memory (remembers your favorites & facts)
#   ✅ Web Search (DuckDuckGo + Wikipedia fallback)
#   ✅ Calculator (safe math)
#   ✅ Topic Summary (title + summary + hashtags)
#   ✅ Joke telling 😄
#   ✅ Tone control (neutral / playful / formal)
# ============================================================

import os
import re
import ast
import operator as op
import requests
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS

# ---------------------------
# 1. Setup
# ---------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise SystemExit("❌ No GROQ_API_KEY found. Add GROQ_API_KEY=... to your .env file.")

client = Groq(api_key=groq_api_key)
MODEL = "llama-3.3-70b-versatile"  # Free, fast, powerful!

# ---------------------------
# 2. Memory
# ---------------------------
chat_history = []   # Full conversation history sent to Groq
facts = {}          # Stored facts e.g. {"sport": "cricket", "food": "biryani"}
tone = "neutral"    # neutral | playful | formal

FAV_PAT = re.compile(r"my (?:favorite|best)\s+(\w+)\s+is\s+([A-Za-z ]+)", re.I)
LIKE_PAT = re.compile(r"\bi (?:like|love)\s+([A-Za-z ]+)", re.I)

TONE_RULE = {
    "neutral": "Use clear, helpful language.",
    "playful": "Be friendly and fun! Add light energy 😊",
    "formal":  "Be professional and concise.",
}

def store_fact(msg: str):
    """Save facts like 'my favorite sport is cricket'"""
    m = FAV_PAT.search(msg)
    if m:
        facts[m.group(1).lower()] = m.group(2).strip(" .!?,")
        return
    m2 = LIKE_PAT.search(msg)
    if m2:
        facts["general_like"] = m2.group(1).strip()

def recall_fact(msg: str):
    """Return memory answer if user asks about something stored"""
    msg_l = msg.lower()
    for key, val in facts.items():
        if key != "general_like" and key in msg_l:
            return f"I remember you told me your favorite {key} is {val} 😊"
    if any(w in msg_l for w in ["like", "love", "favorite", "remember"]) and facts:
        items = [f"{k}: {v}" for k, v in facts.items() if k != "general_like"]
        if items:
            return "Here's what I remember about you: " + ", ".join(items)
    return None

# ---------------------------
# 3. Groq LLM Call
# ---------------------------
def ask_groq(user_message: str, system_extra: str = "") -> str:
    """Send message to Groq Llama 3 and get a reply"""
    system_prompt = (
        f"You are a smart, helpful and friendly AI assistant.\n"
        f"Tone: {tone}. {TONE_RULE[tone]}\n"
        f"Keep answers clear and concise.\n"
        f"{system_extra}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history[-10:]  # last 10 turns for context
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# 4. Calculator
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

def calculate(expr: str) -> str:
    try:
        expr = expr.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        result = _eval_ast(tree.body)
        return f"🧮 Result: {result}"
    except:
        return "❌ Could not calculate that safely."

def is_math(q: str) -> bool:
    return any(ch.isdigit() for ch in q) and any(s in q for s in "+-*/%()^")

# ---------------------------
# 5. Web Search
# ---------------------------
def web_search(query: str):
    """DuckDuckGo search with Wikipedia fallback"""
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

def should_search(q: str) -> bool:
    ql = q.lower()
    return any(kw in ql for kw in [
        "search", "find", "latest", "news", "who is", "what is",
        "when did", "where is", "capital of", "define", "meaning of",
        "founder of", "tell me about"
    ])

# ---------------------------
# 6. Joke Feature
# ---------------------------
def is_joke_request(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["joke", "funny", "make me laugh", "tell me a joke"])

def get_joke_topic(q: str) -> str:
    m = re.search(r"joke about (.+)", q, re.I)
    if m:
        return m.group(1).strip()
    return "anything funny"

# ---------------------------
# 7. Topic Summary Feature
# ---------------------------
def is_summary_request(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["summarize", "summary", "topic summary", "one pager", "hashtags for"])

# ---------------------------
# 8. Tone Control
# ---------------------------
def set_tone_cmd(q: str) -> bool:
    global tone
    m = re.match(r"\s*tone\s+is\s+(neutral|playful|formal)\s*$", q, re.I)
    if m:
        tone = m.group(1).lower()
        return True
    return False

# ---------------------------
# 9. Main Router
# ---------------------------
def respond(user_input: str) -> str:
    global chat_history

    # Tone command
    if set_tone_cmd(user_input):
        return f"✅ Tone updated to: {tone}"

    # Store facts silently
    store_fact(user_input)

    # Recall from memory
    recalled = recall_fact(user_input)
    if recalled:
        return recalled

    # Calculator
    if is_math(user_input):
        return calculate(user_input)

    # Joke
    if is_joke_request(user_input):
        topic = get_joke_topic(user_input)
        reply = ask_groq(
            f"Tell me one funny, clean, original joke about: {topic}. Format: Setup — Punchline.",
            system_extra="You are a funny, witty comedian. Keep jokes clean and short."
        )
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        return f"😂 {reply}"

    # Topic Summary
    if is_summary_request(user_input):
        reply = ask_groq(
            user_input,
            system_extra=(
                "When asked to summarize a topic, provide:\n"
                "1. A one-sentence summary (max 18 words)\n"
                "2. A catchy 5-word title\n"
                "3. Three relevant hashtags\n"
                "Format clearly with labels: Summary:, Title:, Hashtags:"
            )
        )
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        return f"📋 {reply}"

    # Web Search
    if should_search(user_input):
        search_result = web_search(user_input)
        if search_result:
            reply = ask_groq(
                user_input,
                system_extra=f"Use this search result to help answer:\n{search_result}"
            )
        else:
            reply = ask_groq(user_input)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        return f"🔍 {reply}"

    # General chat via Groq
    reply = ask_groq(user_input)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": reply})
    return reply

# ---------------------------
# 10. Run
# ---------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("🤖 Smart AI Agent (Powered by Groq + Llama 3)")
    print("=" * 55)
    print("💬 Chat       - just type anything!")
    print("🧮 Calculator - '25 * 4 + 10'")
    print("🔍 Search     - 'what is machine learning'")
    print("😂 Joke       - 'tell me a joke about Python'")
    print("📋 Summary    - 'summarize the topic: AI'")
    print("🧠 Memory     - 'my favorite food is biryani'")
    print("🎨 Tone       - 'tone is playful / formal / neutral'")
    print("👋 Exit       - 'quit'")
    print("=" * 55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye! 👋")
                break
            if not user_input:
                continue
            reply = respond(user_input)
            print(f"\n🤖 Agent: {reply}\n")
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
