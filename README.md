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

## Configuration

The app reads these environment variables (the API key is **never** hardcoded):

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OLLAMA_API_KEY` | **yes** | — | Ollama Cloud API key. Set it in the host's dashboard. |
| `OLLAMA_MODEL` | no | `gemma4:31b-cloud` | Model tag to call. Override if the tag differs. |
| `OLLAMA_BASE_URL` | no | `https://ollama.com/v1` | OpenAI-compatible base URL. |
| `PORT` | no | `5000` | Port to listen on (set automatically by Render). |

On any model, key, or network failure the app returns a clear message and never
crashes.

## Deploying on Render

This repo includes `render.yaml`, so Render can deploy it as a Blueprint:

1. In Render, create a new **Blueprint** from this GitHub repo (or a Web Service
   using the start command below).
2. Set **`OLLAMA_API_KEY`** in the service's Environment settings.
3. Render builds with `pip install -r requirements.txt` and starts with:

   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
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
