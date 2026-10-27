# SmartStack

SmartStack cuts the cost of answering questions with language models. Instead of sending every question to the biggest model, it checks how hard the question is, reuses answers it already has, sends the model only the documents that help, and reports what each answer cost compared with always using the large model.

- Site: https://naman-singh-777.github.io/SmartStack01/
- API docs: https://smartstack01-api.onrender.com/docs (free hosting, so the first request after a quiet period can take up to a minute)

## What is real AI here

| Step | What it does | Technique | File |
| --- | --- | --- | --- |
| Route | Labels a question simple, moderate or complex and picks a small, medium or large model | TF-IDF, a few hand-made shape features and logistic regression, trained on 120 labelled questions | `backend/app/classifier.py` |
| Cache | Reuses the answer to a question it has seen before | Character n-gram similarity plus a key-word check | `backend/app/cache.py` |
| Retrieve | Finds the passages that help, first in the project documents, then on Wikipedia | Hybrid search: word TF-IDF mixed with character TF-IDF. Wikipedia candidates are re-ranked by SmartStack | `backend/app/retriever.py`, `backend/app/wikipedia.py` |
| Answer | Writes the reply | Calls the Anthropic API when a key is available, otherwise quotes the best sentences from the sources | `backend/app/llm.py` |
| Plan | Suggests where to run the work | Rules over a price table (GPU class, region, spot or on-demand) | `backend/app/scheduler.py` |

`backend/app/engine.py` connects the five steps and is the best file to read first.

## Run it on your computer

You need Python 3.12 or newer.

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload        # API on http://localhost:8000, docs at /docs
```

In a second terminal:

```bash
cd frontend
python -m http.server 8080           # site on http://localhost:8080
```

Run the tests with `pytest` inside `backend`.

### Or with Docker

```bash
docker compose up --build
```

The site is then on http://localhost:8080 and the API on http://localhost:8000/docs.

## Turn on live model answers

Without a key the server runs in offline mode: routing, caching, retrieval and cost estimates all work, but the answer text is sentences quoted from Wikipedia or the project documents. That is fine for simple facts and weak for anything that needs reasoning.

There are two ways to get real model answers:

1. **Server key.** Set an environment variable before starting the server (on Render, add it under Environment):

   ```bash
   export ANTHROPIC_API_KEY=your-key-here
   ```

2. **Visitor key.** On the Try it page, open "Use your own language model key" and paste a key. It is sent in a request header over HTTPS, used for that one request, and never stored or logged by the server. The browser tab remembers it until the tab closes.

Wikipedia lookups need no key. Questions your project documents cannot answer are sent to the public Wikipedia API as search words, so do not type secrets into the question box. Set `WEB_LOOKUP=off` to disable this.

The three model sizes come from `MODEL_SMALL`, `MODEL_MEDIUM` and `MODEL_LARGE` (see `backend/app/config.py`). In live mode, token counts come from the provider instead of being estimated.

## Teach it your own documents

Put Markdown files in `backend/app/data/knowledge/` and restart. The first line of each file is its title.

## Measure the saving

```bash
cd backend
python -m benchmarks.run_benchmark
```

It sends 24 questions the classifier never saw through the pipeline in offline mode. In the last run the classifier picked the intended model size for all 24, and the estimated cost was 63% lower than sending everything to the large model. Treat that as an example: it depends on the mix of questions (12 easy, 7 medium and 5 hard here), on the example prices in `config.py`, and on the assumption that the large model would write an answer of the same length.

## Deploy

- **Site:** GitHub Actions publishes the `frontend` folder to GitHub Pages on every push (`.github/workflows/pages.yml`). In the repository settings, set Pages to use GitHub Actions as its source.
- **API:** `render.yaml` describes the backend for Render. If you deploy it under another name, change `API_BASE` in `frontend/js/config.js` and add the new site address to `ALLOWED_ORIGINS`.

## Limits

- 120 training questions is a small set. Add examples to `backend/app/data/training_queries.json` for your own traffic.
- Prices and machine costs are example numbers.
- Without a language model key, answers are quoted sentences. The Wikipedia lookup picks sources with simple rules, so it can pick the wrong page for unusual questions.
- The compute planner recommends a placement and does not start machines.
- Cache entries and dashboard totals live in memory and reset when the server restarts.
- The cache is shared by every visitor, so do not use it as is for answers that depend on private data.
