import json
import os
import time
import requests
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
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

def build_contents(chat_history, user_question):
    """
    The frontend sends 'chat_history' as the FULL conversation so far,
    including the just-typed user message as the last entry
    (see sendMessage() -> chats[activeChatId].history already has it pushed
    before the fetch call). So we just convert that array directly.

    Frontend item shape:
        {"role": "user",  "text": "hello my name is sam"}
        {"role": "velto", "text": "Hi Sam! Nice to meet you.", "engine": "..."}

    Fallback: if chat_history is empty/missing (e.g. old client), build
    a single-turn conversation from 'question' instead.
    """
    contents = []

    if chat_history:
        for turn in chat_history:
            role = turn.get("role")
            text = turn.get("text", "")
            if not text:
                continue
            gemini_role = "model" if role == "velto" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": text}]
            })
    else:
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

def stream_answer(answer, engine_name):
    """Yield SSE-style chunks so the browser can render the reply progressively."""
    chunk_size = 18
    for index in range(0, len(answer), chunk_size):
        chunk = answer[index:index + chunk_size]
        payload = json.dumps({"delta": chunk})
        yield f"data: {payload}\n\n"
        time.sleep(0.02)
    yield f"data: {json.dumps({'done': True, 'engine_used': engine_name})}\n\n"


@app.route('/ask-homework', methods=['POST'])
def ask_homework():
    data = request.json or {}
    user_question = data.get("question", "")
    chat_history = data.get("chat_history", [])  # matches frontend's key name

    if not user_question:
        return jsonify({"error": "Please type a question first!"}), 400

    contents = build_contents(chat_history, user_question)
    accepts_stream = 'text/event-stream' in request.headers.get('Accept', '')

    models = [
        ("gemini-2.5-flash-lite", "Velto Lite Engine"),
        ("gemini-2.5-flash", "Velto Standard Engine"),
        ("gemini-3.1-flash-lite", "Velto Legacy Engine"),
        ("gemini-3.5-flash", "Velto Apex Engine")
    ]

    for model_name, engine_name in models:
        try:
            app.logger.info(f"Trying model {model_name} ({engine_name})")
            resp = call_gemini(model_name, contents)
            # Log status and a short snippet of the response for debugging
            try:
                resp_text = resp.text
            except Exception:
                resp_text = '<no-text>'
            app.logger.info(f"Model {model_name} returned status {resp.status_code}: {resp_text[:400]}")

            if resp.status_code == 429:
                app.logger.warning(f"Model {model_name} rate limited (429), trying next model")
                continue

            # Safely parse JSON; if parsing fails, log and continue to next model
            try:
                resp_json = resp.json()
            except ValueError:
                app.logger.warning(f"Model {model_name} returned non-JSON response; skipping. Status: {resp.status_code}")
                continue

            if resp.status_code != 200:
                error_msg = resp_json.get("error", {}).get("message", "Unknown error")
                error_str = (error_msg or '').lower()
                app.logger.warning(f"Model {model_name} error: {error_msg}")
                if "quota" in error_str or "resourceexhausted" in error_str or resp.status_code in (429, 503):
                    # try next model when quota/capacity errors occur
                    continue
                # For other kinds of errors, return them to client
                if accepts_stream:
                    payload = json.dumps({ 'error': f'Error: {error_msg}', 'model': model_name })
                    return Response(
                        stream_with_context(iter([f"data: {payload}\n\n"])),
                        mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache'}
                    )
                return jsonify({"error": f"Error: {error_msg}", "model": model_name}), 500

            # Successful response
            answer = resp_json.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "")
            app.logger.info(f"Model {model_name} succeeded; engine: {engine_name}")
            if accepts_stream:
                return Response(
                    stream_with_context(stream_answer(answer, engine_name)),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache'}
                )
            return jsonify({
                "answer": answer,
                "engine_used": engine_name
            })
        except Exception as e:
            app.logger.exception(f"Exception while calling model {model_name}: {e}")
            # continue to next model
            continue

    if accepts_stream:
        return Response(
            stream_with_context(iter([f"data: {json.dumps({'error': 'All engines are currently at capacity. Please try again later.'})}\n\n"])),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache'}
        )

    return jsonify({
        "error": "All engines are currently at capacity. Please try again later."
    }), 503


@app.route('/debug-gemini', methods=['GET'])
def debug_gemini():
    """Lightweight debug endpoint to test direct calls to the Gemini endpoint from the running host.
    Use query `?model=gemini-3.5-flash` to test a specific model. Returns status and a short body.
    """
    model = request.args.get('model', 'gemini-3.5-flash')
    test_contents = [{"role": "user", "parts": [{"text": "Hello from debug check"}]}]
    try:
        app.logger.info(f"Debug: calling Gemini model {model}")
        resp = call_gemini(model, test_contents)
        text = None
        try:
            text = resp.text
        except Exception:
            text = '<no-text>'
        result = {
            'model': model,
            'status': resp.status_code,
            'body_snippet': text[:2000]
        }
        return jsonify(result)
    except Exception as e:
        app.logger.exception('Debug Gemini call failed')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
