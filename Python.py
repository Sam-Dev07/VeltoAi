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


def call_gemini(model_name, user_question):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    payload = {
        "contents": [
            {"parts": [{"text": user_question}]}
        ],
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

    if not user_question:
        return jsonify({"error": "Please type a question first!"}), 400

    # Model fallback chain
    models = [
        ("gemini-flash-latest", "Velto Standard Engine"),
        ("gemini-flash-lite-latest", "Velto Lite Engine")
    ]

    for model_name, engine_name in models:
        try:
            resp = call_gemini(model_name, user_question)

            if resp.status_code == 429:
                continue  # quota hit, try next model

            resp_json = resp.json()

            if resp.status_code != 200:
                error_msg = resp_json.get("error", {}).get("message", "Unknown error")
                error_str = error_msg.lower()
                if "quota" in error_str or "resourceexhausted" in error_str:
                    continue
                return jsonify({"error": f"Error: {error_msg}"}), 500

            answer = resp_json["candidates"][0]["content"]["parts"][0]["text"]

            return jsonify({
                "answer": answer,
                "engine_used": engine_name
            })

        except Exception as e:
            return jsonify({"error": f"Error: {str(e)}"}), 500

    # All models failed
    return jsonify({
        "error": "All engines are currently at capacity. Please try again later."
    }), 503


if __name__ == '__main__':
    app.run(port=5000, debug=True)
