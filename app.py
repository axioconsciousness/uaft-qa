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
    "symmetry-as-appearance-of-asymmetry.md",
    "quantum-differentiation-resonance.md",
    "differentiation-boundary.md",
]

# --- Model / API config (read from env; never hardcode the key) --------------

# Standard engine: Gemma via Ollama Cloud (OpenAI-compatible).
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")

# Comprehensive engine: the real OpenAI API, used for harder questions.
# If OPENAI_API_KEY is unset, the app behaves exactly as before (Gemma only).
# OPENAI_MODEL is overridable so you can match whatever your credits cover
# (e.g. gpt-5.5 for hardest, gpt-5.4-mini to conserve credits).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

# Deep engine: Kimi K3 (Moonshot), also OpenAI-compatible. Reserved for the
# hardest questions. If KIMI_API_KEY is unset, routing collapses to the two
# engines above and nothing else changes.
KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
KIMI_BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-k3")

# Routing across the three engines, by question-hardness score band:
#   score <  ROUTER_THRESHOLD       -> standard      (Gemma)
#   score <  ROUTER_DEEP_THRESHOLD  -> comprehensive (OpenAI)
#   score >= ROUTER_DEEP_THRESHOLD  -> deep          (Kimi K3)
# Raise a threshold to send fewer questions to that tier (saves credits).
# ROUTE_MODE = auto (default) | standard | comprehensive | deep  (forces one engine)
ROUTE_MODE = os.environ.get("ROUTE_MODE", "auto").strip().lower()
ROUTER_THRESHOLD = int(os.environ.get("ROUTER_THRESHOLD", "3"))
ROUTER_DEEP_THRESHOLD = int(os.environ.get("ROUTER_DEEP_THRESHOLD", "6"))

# Seconds to wait on a single LLM call. Keep the gunicorn --timeout in the
# start command comfortably ABOVE this, or the worker is killed mid-request
# and returns an uncatchable 500 instead of falling back.
REQUEST_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

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

def build_messages(question, master_text, paper_texts, thorough=False):
    sources = ["=== UAFT MASTER FILE ===", master_text]
    for text in paper_texts:
        sources.append("\n=== SUPPORTING PAPER ===")
        sources.append(text)
    sources_block = "\n".join(sources)

    system = SYSTEM_PROMPT
    if thorough:
        system += (
            " Be thorough and comprehensive: draw connections across the "
            "provided sources and explain your reasoning step by step, while "
            "staying strictly within what the sources support."
        )

    user_content = (
        "Use the following UAFT source documents to answer the question.\n\n"
        f"{sources_block}\n\n"
        f"=== QUESTION ===\n{question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# --- Engines + routing -------------------------------------------------------

def _engines():
    """The three engines, cheapest first (fallback tries them in this order).

    'standard' = Gemma/Ollama, 'comprehensive' = OpenAI, 'deep' = Kimi K3.
    """
    return {
        "standard": {
            "base_url": OLLAMA_BASE_URL,
            "api_key": OLLAMA_API_KEY,
            "model": OLLAMA_MODEL,
        },
        "comprehensive": {
            "base_url": OPENAI_BASE_URL,
            "api_key": OPENAI_API_KEY,
            "model": OPENAI_MODEL,
        },
        "deep": {
            "base_url": KIMI_BASE_URL,
            "api_key": KIMI_API_KEY,
            "model": KIMI_MODEL,
        },
    }


def _chat_completion(engine, messages):
    """Call any OpenAI-compatible chat endpoint (Ollama Cloud or OpenAI)."""
    resp = requests.post(
        f"{engine['base_url'].rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {engine['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": engine["model"],
            "messages": messages,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    # Parse defensively: reasoning models may return null content, put the text
    # under another key, or return content as a list of parts.
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"no choices in response; keys={sorted(data)}")
    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    if not content:
        content = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"empty content; message keys={sorted(message)}")
    return content.strip()


# Words/phrases that signal a question wants depth, synthesis, or comparison.
# ('why'/'how' are common, so the depth contribution is capped in route_engine.)
_DEPTH_PATTERN = re.compile(
    r"\b(compare|comparison|contrast|versus|vs|differ|differs|difference|"
    r"differences|relationship|relate|relates|related|connection|connections|"
    r"reconcile|synthesi\w+|integrate|integration|unify|unified|implication|"
    r"implications|consequence|consequences|derive|derivation|prove|proof|"
    r"mechanism|mechanisms|justify|critique|evaluate|analy[sz]e|analysis|"
    r"trade-?off|across|comprehensive|in-depth|elaborate|rigorous|formal|"
    r"why|how|explain|explains)\b",
    re.IGNORECASE,
)


def route_engine(question, matched_count):
    """Pick the preferred engine by scoring the question's 'hardness'.

    Transparent and tunable: returns (engine_name, human_reason). ROUTE_MODE
    can force a single engine; otherwise a score >= ROUTER_THRESHOLD routes to
    the comprehensive (OpenAI) engine.
    """
    if ROUTE_MODE in ("standard", "comprehensive", "deep"):
        return ROUTE_MODE, f"forced by ROUTE_MODE={ROUTE_MODE}"

    reasons = []
    score = 0

    depth_hits = {m.lower() for m in _DEPTH_PATTERN.findall(question)}
    if depth_hits:
        contribution = min(len(depth_hits), 3)  # cap so common words don't run away
        score += contribution
        reasons.append(f"depth/compare terms {sorted(depth_hits)} (+{contribution})")

    if matched_count >= 2:
        score += 2
        reasons.append("spans multiple papers (+2)")

    words = len(re.findall(r"\w+", question))
    if words >= 45:
        score += 2
        reasons.append("long question (+2)")
    elif words >= 25:
        score += 1
        reasons.append("medium-length question (+1)")

    if question.count("?") >= 2:
        score += 1
        reasons.append("multiple sub-questions (+1)")

    if score >= ROUTER_DEEP_THRESHOLD:
        engine = "deep"
    elif score >= ROUTER_THRESHOLD:
        engine = "comprehensive"
    else:
        engine = "standard"
    reason = (
        f"score={score} (bands: <{ROUTER_THRESHOLD} standard, "
        f"<{ROUTER_DEEP_THRESHOLD} comprehensive, else deep): "
        + ("; ".join(reasons) or "no depth signals")
    )
    return engine, reason


def _try_order(preferred):
    """Ordered [(name, engine)] to attempt: preferred first, then any other
    engine that has a key configured (so one engine can cover for the other)."""
    engines = _engines()
    names = [preferred] + [n for n in engines if n != preferred]
    return [(n, engines[n]) for n in names if engines[n]["api_key"]]


def answer_question(question):
    """Full pipeline. Returns (answer, engine_used, model, error_message). Never raises."""
    if not question or not question.strip():
        return None, None, None, "Please enter a question."

    if not any(e["api_key"] for e in _engines().values()):
        return None, None, None, (
            "The server has no model API key configured. Please set "
            "OLLAMA_API_KEY (and optionally OPENAI_API_KEY / KIMI_API_KEY) "
            "in the environment."
        )

    # Tier 1: always load the master file.
    try:
        master_text = fetch_file(MASTER_FILE)
    except requests.RequestException:
        return None, None, None, (
            "Couldn't load the core UAFT material right now. "
            "Please try again in a moment."
        )

    # Tier 2: include whole papers only when the question clearly matches.
    matched = match_papers(question)
    paper_texts = []
    try:
        for filename in matched:
            paper_texts.append(fetch_file(filename))
    except requests.RequestException:
        # If a supporting paper fails, fall back to the master file alone.
        paper_texts = []

    # Route to an engine, then try it — falling back to the other on failure.
    preferred, reason = route_engine(question, len(matched))
    app.logger.info("route %r -> %s [%s]", question[:80], preferred, reason)

    order = _try_order(preferred)
    if not order:
        return None, None, None, (
            "The selected model isn't configured. Please check the server's API keys."
        )

    last_error = "The model returned an empty answer. Please try again."
    for name, engine in order:
        detail = "empty answer"
        try:
            messages = build_messages(
                question, master_text, paper_texts,
                thorough=(name in ("comprehensive", "deep")),
            )
            answer = _chat_completion(engine, messages)
            if answer:
                return answer, name, engine["model"], None
        except requests.Timeout:
            last_error = "The model took too long to respond. Please try again."
            detail = "timeout"
        except requests.HTTPError as exc:
            last_error = (
                "Couldn't reach the language model right now. "
                "Please try again in a moment."
            )
            resp = exc.response
            status = getattr(resp, "status_code", "?")
            body = resp.text[:300] if resp is not None else str(exc)
            detail = f"HTTP {status}: {body}"
        except requests.RequestException as exc:
            last_error = (
                "Couldn't reach the language model right now. "
                "Please try again in a moment."
            )
            detail = f"{type(exc).__name__}: {str(exc)[:200]}"
        except (KeyError, ValueError, TypeError, AttributeError, IndexError) as exc:
            last_error = "The model returned an unexpected response. Please try again."
            detail = f"bad response shape: {type(exc).__name__}: {str(exc)[:200]}"
        except Exception as exc:  # last resort: one engine must never 500 the request
            last_error = "Something went wrong reaching the model. Please try again."
            detail = f"unexpected {type(exc).__name__}: {str(exc)[:200]}"
        app.logger.warning("engine '%s' (model=%s) failed: %s", name, engine["model"], detail)

    return None, None, None, last_error


# --- Routes ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer, engine, model, error = answer_question(question)
    if error:
        return jsonify({"error": error}), 200
    # 'engine' (kind) and 'model' (exact model id) drive the badge in the UI.
    return jsonify({"answer": answer, "engine": engine, "model": model}), 200


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
