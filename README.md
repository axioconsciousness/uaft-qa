# UAFT Q&A

A small two-tier retrieval Q&A site for the UAFT research framework. No chunk
retrieval and no vector database — it fetches whole Markdown papers from GitHub
and sends them to a cloud LLM (Gemma via Ollama Cloud's OpenAI-compatible API).

## How it works

- **Tier 1 — always loaded:** `UAFT_Master_File.md` is included in the context
  of every request.
- **Tier 2 — on demand:** the question is matched (by simple token overlap)
  against a lookup built from each paper's title and first paragraph. If one (or
  at most two) papers clearly match, those *whole* papers are added to the
  context. If nothing clearly matches, the master file is used alone. Papers are
  never all loaded by default, and nothing is ever chunked.
- The master file (plus any matched paper) and the question are sent to the
  model. The model is instructed to answer **only** from the supplied sources,
  not to invent claims, and to say so when a question falls outside the
  framework. The user sees only the clean conversational answer — no scores, no
  filenames.
- Source files are fetched over HTTPS and cached in memory after first fetch.

## Two engines (standard / comprehensive)

The app can use two LLMs and auto-route between them by question difficulty:

| Tier | Model | Score band |
| --- | --- | --- |
| **Standard** | Gemma via Ollama Cloud | below `ROUTER_THRESHOLD` (default < 3) |
| **Comprehensive** | OpenAI | at/above `ROUTER_THRESHOLD` (3+) |

An **auto-router** scores each question for "hardness": depth/comparison words
(*compare, relationship, implications, derive, why, how* — capped at +3), whether
it spans two papers (+2), length (+1/+2), and multiple sub-questions (+1). Both
engines get the same grounded context and the same "answer only from the sources"
rule; the comprehensive tier also gets a "be thorough" nudge. The router decision
is logged server-side; the UI shows the answer plus a small badge naming the
model that replied.

> A third "deep" tier using Kimi (K3, then K2.6 via Moonshot) was trialled and
> removed: it answered in 110–170s against ~8s for the models above, and
> disabling thinking mode, capping output tokens, and trimming input did not
> close the gap. Adding a provider back is just another entry in `_engines()`.

Every engine is a plain OpenAI-compatible chat endpoint, so adding another
provider is just another entry in `_engines()`.

Tiers whose key is unset are simply skipped — with only `OLLAMA_API_KEY` set, the
app behaves as a single-engine app. If a chosen engine fails, the app falls back
to the other configured engines (cheapest first) before showing an error.

## Configuration

The app reads these environment variables (the API key is **never** hardcoded):

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OLLAMA_API_KEY` | yes* | — | Ollama Cloud API key (standard engine). Set it in the host's dashboard. |
| `OLLAMA_MODEL` | no | `gemma4:31b-cloud` | Standard model tag. Override if the tag differs. |
| `OLLAMA_BASE_URL` | no | `https://ollama.com/v1` | OpenAI-compatible base URL for Ollama Cloud. |
| `OPENAI_API_KEY` | no | — | OpenAI API key (comprehensive engine). If unset, that tier is skipped. |
| `OPENAI_MODEL` | no | `gpt-5.5` | OpenAI model for hard questions. Set to e.g. `gpt-5.4-mini` to conserve credits. |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | OpenAI chat-completions base URL. |
| `ROUTE_MODE` | no | `auto` | `auto`, or force one engine: `standard`, `comprehensive`. |
| `ROUTER_THRESHOLD` | no | `3` | Score at which routing moves up to the comprehensive tier. |
| `LLM_TIMEOUT` | no | `120` | Seconds to wait on one model call. Keep gunicorn's `--timeout` above this. |
| `PORT` | no | `5000` | Port to listen on (set automatically by Render). |

\* At least one engine key is required. `OLLAMA_API_KEY` alone reproduces the
original single-engine behavior; add `OPENAI_API_KEY` to enable auto-routing.

On any model, key, or network failure the app returns a clear message and never
crashes.

## Deploying on Render

This repo includes `render.yaml`, so Render can deploy it as a Blueprint:

1. In Render, create a new **Blueprint** from this GitHub repo (or a Web Service
   using the start command below).
2. Set **`OLLAMA_API_KEY`** in the service's Environment settings.
3. Render builds with `pip install -r requirements.txt` and starts with:

   ```
   gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 180 --graceful-timeout 30
   ```

   The app listens on the port Render provides via `PORT`.

## Running locally (optional)

> Note: `gunicorn` does not run on Windows. On Windows, use the Flask dev server
> below; on Linux/macOS you can also use the gunicorn command above.

```bash
pip install -r requirements.txt
export OLLAMA_API_KEY=your-key-here   # PowerShell: $env:OLLAMA_API_KEY="your-key-here"
python app.py                          # serves on http://localhost:5000
```

## Project layout

```
app.py                # Flask app: fetching, matching, prompt assembly, LLM call
templates/index.html  # Single-page UI (Source Serif / Source Sans, 760px)
requirements.txt      # Flask, requests, gunicorn
render.yaml           # Render Blueprint (web service + env vars)
```
