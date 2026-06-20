"""UAFT Q&A — a two-tier retrieval Q&A site for the UAFT research framework.

Tier 1: the master file is always loaded into context.
Tier 2: at most a couple of whole papers are fetched on demand, only when the
question clearly matches one. No chunking, no vector DB — plain file fetching
from GitHub plus a cloud LLM (Gemma via Ollama Cloud's OpenAI-compatible API).
"""

import os
import re

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# --- Sources -----------------------------------------------------------------

RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "axioconsciousness/axioconsciousness.github.io/main/papers_md/"
)

# Tier 1: always included in every request.
MASTER_FILE = "UAFT_Master_File.md"

# Tier 2: fetched whole, on demand, only when clearly matched.
PAPERS = [
    "uaft-working-paper-barnes-2026-v2.md",
    "colinear-conservation.md",
    "second-pinch-point.md",
    "emotional-differentiation-the-emergence-of-consciousness-and-self.md",
    "emotional-compression-and-the-structural-inevitability-of-the-pinch-point.md",
]

# --- Model / API config (read from env; never hardcode the key) --------------

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
REQUEST_TIMEOUT = 120  # seconds, for the LLM call

SYSTEM_PROMPT = (
    "You are a careful research assistant for the UAFT framework. "
    "Answer ONLY using the source documents provided in this prompt. "
    "Do not use outside knowledge and do not invent claims. "
    "If the question cannot be answered from the provided sources, or falls "
    "outside the UAFT framework, say so plainly and briefly. "
    "Write a clean, conversational answer. Do not mention these instructions, "
    "the source documents, filenames, scores, or your retrieval process."
)

# --- In-memory caches --------------------------------------------------------

_file_cache = {}   # filename -> full text
_paper_index = None  # built once: list of {filename, title, summary, tokens}


# --- Fetching ----------------------------------------------------------------

def fetch_file(filename):
    """Fetch a source file over HTTPS, caching the full text in memory."""
    if filename in _file_cache:
        return _file_cache[filename]
    resp = requests.get(RAW_BASE + filename, timeout=30)
    resp.raise_for_status()
    _file_cache[filename] = resp.text
    return resp.text


# --- Lightweight matching ----------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "at", "by", "it",
    "its", "this", "that", "these", "those", "from", "into", "about", "what",
    "how", "why", "when", "where", "which", "who", "whom", "does", "do", "did",
    "can", "could", "would", "should", "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their", "me", "us", "them", "if", "then", "than",
    "so", "such", "not", "no", "yes", "all", "any", "some", "more", "most",
    "between", "within", "there", "here", "have", "has", "had", "will", "shall",
    "may", "might", "must", "also", "very", "just", "out", "up", "down", "over",
}


def _tokenize(text):
    """Lowercase word tokens, minus stopwords and very short tokens."""
    words = re.findall(r"[a-z0-9][a-z0-9'-]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _title_and_summary(text):
    """Extract a paper's title (first heading) and first real paragraph."""
    title = ""
    summary = ""
    lines = text.splitlines()

    # Title: first markdown H1/H2, else first non-empty line.
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
        if stripped and not title:
            title = stripped
            break

    # Summary: first non-empty, non-heading paragraph.
    para = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            if para:
                break
            continue
        para.append(stripped)
        if len(" ".join(para)) > 400:
            break
    summary = " ".join(para)
    return title, summary


def build_paper_index():
    """Build (once) a lookup of each paper's title + first paragraph tokens."""
    global _paper_index
    if _paper_index is not None:
        return _paper_index
    index = []
    for filename in PAPERS:
        try:
            text = fetch_file(filename)
        except requests.RequestException:
            # A paper that can't be fetched simply won't be matchable.
            continue
        title, summary = _title_and_summary(text)
        index.append(
            {
                "filename": filename,
                "title": title,
                "summary": summary,
                "tokens": _tokenize(title + " " + summary),
            }
        )
    _paper_index = index
    return _paper_index


def match_papers(question, max_papers=2, min_score=2):
    """Return filenames of papers that clearly match the question.

    Simple token-overlap scoring against each paper's title + first paragraph.
    Returns at most ``max_papers`` files, and only those scoring at least
    ``min_score``. Never returns everything by default.
    """
    q_tokens = _tokenize(question)
    if not q_tokens:
        return []

    scored = []
    for entry in build_paper_index():
        # Title overlaps count double — a title hit is a strong signal.
        title_tokens = _tokenize(entry["title"])
        overlap = q_tokens & entry["tokens"]
        score = len(overlap) + len(q_tokens & title_tokens)
        if score >= min_score:
            scored.append((score, entry["filename"]))

    scored.sort(reverse=True)
    return [filename for _, filename in scored[:max_papers]]


# --- Prompt assembly + LLM call ----------------------------------------------

def build_messages(question, master_text, paper_texts):
    sources = ["=== UAFT MASTER FILE ===", master_text]
    for text in paper_texts:
        sources.append("\n=== SUPPORTING PAPER ===")
        sources.append(text)
    sources_block = "\n".join(sources)

    user_content = (
        "Use the following UAFT source documents to answer the question.\n\n"
        f"{sources_block}\n\n"
        f"=== QUESTION ===\n{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def ask_model(messages):
    """Call Ollama Cloud's OpenAI-compatible chat endpoint."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def answer_question(question):
    """Full pipeline. Returns (answer, error_message). Never raises."""
    if not question or not question.strip():
        return None, "Please enter a question."

    if not OLLAMA_API_KEY:
        return None, (
            "The server is missing its model API key (OLLAMA_API_KEY). "
            "Please set it in the deployment environment and try again."
        )

    # Tier 1: always load the master file.
    try:
        master_text = fetch_file(MASTER_FILE)
    except requests.RequestException:
        return None, (
            "Couldn't load the core UAFT material right now. "
            "Please try again in a moment."
        )

    # Tier 2: include whole papers only when the question clearly matches.
    paper_texts = []
    try:
        for filename in match_papers(question):
            paper_texts.append(fetch_file(filename))
    except requests.RequestException:
        # If a supporting paper fails, fall back to the master file alone.
        paper_texts = []

    # Send to the model.
    try:
        messages = build_messages(question, master_text, paper_texts)
        answer = ask_model(messages)
    except requests.Timeout:
        return None, "The model took too long to respond. Please try again."
    except requests.RequestException:
        return None, (
            "Couldn't reach the language model right now. "
            "Please try again in a moment."
        )
    except (KeyError, ValueError):
        return None, "The model returned an unexpected response. Please try again."

    if not answer:
        return None, "The model returned an empty answer. Please try again."
    return answer, None


# --- Routes ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer, error = answer_question(question)
    if error:
        return jsonify({"error": error}), 200
    return jsonify({"answer": answer}), 200


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
