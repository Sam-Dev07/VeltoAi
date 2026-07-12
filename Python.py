import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv(override=True)
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

SYSTEM_INSTRUCTION = """
    You are Velto AI, a helpful and friendly AI assistant.
    Your creator and owner is Shameek Chaturvedi.
    Never mention Google, Gemini, or any company name.
    If anyone asks about age (yours or your creator's), reply with: "Sorry I can't help you with that!"
    if anyone asks about how you were made tell them "I was founded by Shameek Chaturvedi, the owner of this website, i was found on 7th July 2026 as a tool for homework assistance!"
"""

def build_contents(history, user_question):
    """
    Converts frontend history (list of {role, text}) into Gemini's
    'contents' format, then appends the new user question at the end.

    Expected history item shape from frontend:
        {"role": "user", "text": "hello my name is sam"}
        {"role": "assistant", "text": "Hi Sam! Nice to meet you."}
    """
    contents = []

    for turn in history:
        role = turn.get("role")
        text = turn.get("text", "")
        if not text:
            continue
        # Gemini only accepts "user" or "model" as roles
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": text}]
        })

    # Add the current question as the latest user turn
    contents.append({
        "role": "user",
        "parts": [{"text": user_question}]
    })

    return contents

def call_gemini(model_name, contents):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    return resp

@app.route('/ask-homework', methods=['POST'])
def ask_homework():
    data = request.json or {}
    user_question = data.get("question", "")
    history = data.get("history", [])  # <-- NEW: list of past turns from frontend

    if not user_question:
        return jsonify({"error": "Please type a question first!"}), 400

    contents = build_contents(history, user_question)

    # Model fallback chain (Ordered from lowest/worse tier up to the absolute best)
    models = [
        ("gemini-2.5-flash-lite", "Velto Lite Engine"),
        ("gemini-2.5-flash", "Velto Standard Engine"),
        ("gemini-3.1-flash-lite", "Velto Legacy Engine"),
        ("gemini-3.5-flash", "Velto Apex Engine")
    ]

    for model_name, engine_name in models:
        try:
            resp = call_gemini(model_name, contents)
            if resp.status_code == 429:
                continue  # quota or rate limit hit, skip to next model
            resp_json = resp.json()
            if resp.status_code != 200:
                error_msg = resp_json.get("error", {}).get("message", "Unknown error")
                error_str = error_msg.lower()
                if "quota" in error_str or "resourceexhausted" in error_str:
                    continue  # skip to next model if limited
                return jsonify({"error": f"Error: {error_msg}"}), 500
            answer = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            return jsonify({
                "answer": answer,
                "engine_used": engine_name
            })
        except Exception:
            # If a model completely fails to connect, skip to the next one
            continue

    # All 4 models failed
    return jsonify({
        "error": "All engines are currently at capacity. Please try again later."
    }), 503

if __name__ == '__main__':
    app.run(port=5000, debug=True)
