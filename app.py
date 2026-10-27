"""Flask web UI for RAG ML Docs.

Usage:  ~/workspace/ml_env/bin/python app.py
Then open http://127.0.0.1:5000

Chat history is kept per browser session (Flask session cookie, capped at
6 exchanges) so follow-up questions can use conversation memory.
"""

import os

from flask import Flask, jsonify, render_template, request, session

from rag_pipeline import RAGIndex, ChatSession, answer_question

app = Flask(__name__)
app.secret_key = "rag-ml-docs-local-demo"

INDEX = None


def get_index():
    global INDEX
    if INDEX is None:
        INDEX = RAGIndex()
    return INDEX


def get_session():
    chat = ChatSession()
    chat.history = [tuple(x) for x in session.get("history", [])]
    return chat


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400
    generative = bool(data.get("generative")) or bool(
        os.environ.get("RAG_GENERATIVE"))
    chat = get_session()
    result = answer_question(get_index(), question, session=chat,
                             generative=generative, debug=True)
    session["history"] = chat.history
    return jsonify({
        "question": result["question"],
        "sub_questions": result["sub_questions"],
        "used_history": result["used_history"],
        "mode": result["mode"],
        "generative_text": result["generative_text"],
        "citations": [
            {"rank": c["rank"], "title": c["title"], "section": c["section"],
             "url": c["url"], "source": c["source"],
             "rerank_score": c["rerank_score"],
             "text": c["text"][:700]}
            for c in result["citations"]],
        "citations_verified": result["citations_verified"],
        "debug": result.get("debug"),
    })


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("history", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Loading index (first run downloads the reranker, ~80MB)...")
    get_index()
    app.run(host="127.0.0.1", port=5000, debug=False)
