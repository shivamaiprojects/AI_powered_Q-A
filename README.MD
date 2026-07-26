# RAG-Based FAQ Assistant

Ask natural-language questions against a 5,100-document knowledge base of data science, statistics, and machine learning Q&A. Describe a problem in your own words — *"my model does great on training data but terribly on new data"* — and get a grounded answer synthesised from the most relevant retrieved passages, with sources.

## Overview

Keyword search fails when a user describes a problem rather than naming a concept. A question about "my model failing on unseen data" shares no words with the answer that discusses "overfitting." This system embeds a Stack Exchange corpus into a vector space where semantic similarity is measurable, retrieves the closest passages, and passes them to a Llama-based LLM constrained to answer only from that context.

Retrieval runs locally on a consumer GPU; generation runs on a hosted inference provider. This split keeps retrieval latency in the single-digit milliseconds without needing local GPU capacity for an 8B model.

## Features

- **Semantic retrieval** over 10,014 chunks via dense embeddings and FAISS
- **Grounded generation** — the LLM answers only from retrieved context and refuses when nothing relevant is found
- **Calibrated hallucination guard** — a similarity threshold derived from measured score distributions, tuned against adversarial queries
- **Source attribution** — every answer carries the passages it was built from
- **Provider-agnostic LLM client** — swap inference providers with one config value, no code change
- **Full evaluation harness** — Hit Rate, MRR, NDCG, faithfulness, and latency percentiles

## Architecture

```
                ┌──────────────────┐
                │  Streamlit UI    │  :8501
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │    RAG chain     │
                └────────┬─────────┘
             ┌───────────┴───────────┐
    ┌────────▼────────┐    ┌─────────▼─────────┐
    │  Retriever      │    │   LLM client      │
    │  MiniLM + FAISS │    │  Llama 3.1 8B     │
    │  (local GPU)    │    │  (hosted API)     │
    └─────────────────┘    └───────────────────┘
```

Ingestion (clean → chunk → embed → index) runs offline and is allowed to be slow. Inference (embed query → search → ground → generate) runs per request and stays under a second. That asymmetry drives the design: the index is built once and persisted, so the app loads it in milliseconds rather than re-embedding on startup.

## Tech Stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Selected over bge-small/base by measured retrieval accuracy on this corpus |
| Vector store | FAISS `IndexFlatIP` | Exact search; approximation is unnecessary at 10K vectors |
| LLM | Llama 3.1 8B Instruct | Hosted via an OpenAI-compatible endpoint |
| Framework | LangChain | Provider-agnostic composition via LCEL |
| Frontend | Streamlit | Rapid UI iteration |
| Config | pydantic-settings | Typed, validated settings from a single `.env` |

## Pipeline & Results

### 1. Dataset selection

Source: [flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl), using the `datascience`, `ai`, and `stats` configurations — 141,945 title→answer pairs.

I evaluated and rejected the Bitext customer-support dataset first. It contains 27 intents rephrased ~1,000 ways each — ideal for fine-tuning, but poison for retrieval, since a RAG index of near-duplicate paraphrases returns the same answer many times and makes retrieval metrics meaningless. Stack Exchange gives the opposite profile: thousands of genuinely distinct questions with accepted answers as ground truth, which is what makes retrieval a real problem worth solving. Measured duplication confirmed it — only 0.016% of answers were exact duplicates.

### 2. Data cleaning

From 141,945 raw pairs, cleaning applied HTML entity unescaping (24.6% of answers affected), LaTeX normalisation to plain words (40.3% affected, rising to 45.8% on `stats`), code-block stripping for the embedding text, and length filtering. The minimum answer length of 250 characters was set from the observed distribution: answers below the 5th percentile were upvoted comments rather than answers ("Pi", "Try h2o, more information here"). With 141,945 pairs available and only ~5,000 needed, filtering aggressively for quality cost nothing.

The corpus was then stratified to 1,700 documents per site (5,100 total), balancing the three communities so no single one dominates — `stats` alone was 81.5% of the raw data.

Two versions of each answer are produced: a **display** version (HTML-cleaned, code and math intact) shown to the user, and an **embedding** version (code collapsed, LaTeX normalised) used to build vectors. What you search over is not what you return.

### 3. Chunking

Long answers are split with a recursive character splitter at 900 characters with 150 overlap, chosen to stay within MiniLM's 256-token limit. Undersized tail fragments are merged into their neighbour so no chunk is an unusable fragment. Result: 10,014 chunks, 1.96 per document on average.

### 4. Embedding model selection

Four configurations were benchmarked on 1,500 held-out question/passage pairs, measured by in-batch retrieval accuracy rather than MTEB leaderboard rank:

| Model | Accuracy@1 | MRR | Passages/s |
| --- | --- | --- | --- |
| bge-small (prefix) | 0.476 | 0.568 | 156 |
| bge-small (no prefix) | 0.457 | 0.549 | 159 |
| bge-base (768-dim) | 0.469 | 0.569 | 57 |
| **all-MiniLM-L6-v2** | **0.540** | **0.642** | **419** |

MiniLM won on every ranking metric and was 2.7× faster, despite a shorter context window. This contradicts general leaderboard rankings and is a domain-match effect: MiniLM was trained largely on Stack Exchange and similar Q&A text, which is exactly this corpus. Scaling bge to base size did not help, ruling out capacity as the limitation.

### 5. Vector index

FAISS `IndexFlatIP` over unit-normalised vectors, which makes inner product equal to cosine similarity. An A/B against IVF confirmed exact search is the right choice here: at 10K vectors, flat search completes in ~0.5ms at 100% recall, while retrieval is only ~0.1% of total response time. Approximation would trade recall for a speedup against a step that isn't the bottleneck. IVF becomes worthwhile past ~100K–1M vectors.

### 6. Retrieval threshold

A query below a similarity floor is treated as unanswerable, so the LLM is never asked to answer from irrelevant context. The threshold was calibrated from measured score distributions over 1,000 pairs (positive median 0.50, negative median 0.05). An initial value of 0.20 — set from in-corpus negatives — failed against adversarial testing: out-of-domain queries like "what time does the pizza shop close" scored ~0.33, above the floor. The threshold was raised to 0.40 after adversarial testing cleanly separated real questions (top hits ≥ 0.48) from nonsense.

### 7. Cross-encoder reranking (rejected)

A `ms-marco-MiniLM` cross-encoder was added over the top-20 candidates, expecting it to lift Hit Rate@1 by reordering correct-but-buried documents. A controlled A/B on 498 queries showed the opposite: it *reduced* Hit Rate@1 from 0.44 to 0.35 and MRR from 0.52 to 0.47. Reranking against clean rather than embedding-normalised passages ruled out text preprocessing as the cause — the regression held. The likely reason is distribution mismatch: the reranker was trained on natural-language web queries, while these queries are short (~57-character) technical titles that the domain-tuned bi-encoder already ranks better. Kept disabled. The code remains for reproducibility.

### 8. Evaluation

Ground truth is self-supervised: each question is paired with the `doc_id` of the answer it belongs to, so a retrieval is correct when that document appears. Since the query is the title and the document is the answer — sharing no vocabulary by construction — this measures genuine semantic retrieval, not lexical overlap.

**Retrieval** (498 queries, full 10K index):

| Metric | Value |
| --- | --- |
| Hit Rate@1 | 0.440 |
| Hit Rate@3 | 0.564 |
| Hit Rate@5 | 0.623 |
| Hit Rate@10 | 0.689 |
| MRR | 0.515 |
| NDCG@5 | 0.535 |

**Generation & latency** (Groq, 10 samples):

| Metric | Value |
| --- | --- |
| Latency p50 | 272 ms |
| Latency p95 | 532 ms |
| Latency p99 | 676 ms |
| Retrieval p50 | 20 ms |
| Faithfulness (mean cosine to context) | 0.66 |

Every latency percentile is sub-second. The retrieval/generation split shows retrieval is essentially free (20ms) and generation dominates (~250ms), so the "sub-second" claim rests on the inference provider's speed, not the search.

## Setup

```
git clone https://github.com/shivamaiprojects/AI_powered_Q-A.git
cd AI_powered_Q-A

python -m venv .venv
.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Create a `.env` with:

```
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

`HF_TOKEN` downloads the embedding model; `GROQ_API_KEY` serves generation. The client routes to whichever provider `LLM_BASE_URL` names.

## Running

Build the pipeline in order:

```
python scripts/download_data.py       # fetch the Stack Exchange corpus
python scripts/prepare_data.py         # clean, filter, balance, chunk
python scripts/build_embeddings.py     # encode chunks
python scripts/build_index.py          # build and persist the FAISS index
```

Evaluate:

```
python scripts/run_evaluation.py       # retrieval + generation metrics
```

Launch the app:

```
streamlit run app/streamlit_app.py
```

UI at `http://localhost:8501`.

## Project Structure

```
├── app/                  Streamlit frontend
├── config/               Central settings (pydantic-settings)
├── data/                 raw/, processed/, vectorstore/ (git-ignored)
├── notebooks/            Exploratory analysis
├── reports/metrics/      Evaluation results (JSON)
├── scripts/              Pipeline entry points
├── src/faq_rag/
│   ├── data/             Cleaning, chunking, pipeline
│   ├── embeddings/       Embedding model wrapper
│   ├── vectorstore/      FAISS store, retriever, reranker
│   ├── llm/              Provider-agnostic LLM client
│   ├── evaluation/       Metric computation
│   └── rag/              Chain, prompts
└── tests/                pytest suite
```

## Known Limitations

- **Queries are title-only.** The source dataset dropped question bodies, so queries are ~57-character titles with limited signal. This caps retrieval accuracy and is the likely reason cross-encoder reranking underperformed.
- **Faithfulness is a proxy.** It measures cosine similarity between answer and context, not a rigorous entailment check. An LLM-judge (RAGAS-style) would be more accurate but costs an API call per sample.
- **Latency measured at k=3.** Groq free-tier rate limits capped the evaluation at 3 retrieved passages; the app defaults to the same for consistency between measured and deployed behaviour.
- **LLM dependency risk.** `llama-3.1-8b-instant` is scheduled for deprecation on Groq; the provider-agnostic client mitigates this — switching to a replacement is a one-line config change.
- **31% of queries return no result in the top 10**, driven partly by the title-only limitation and near-duplicate topics competing for rank.

## Roadmap

- [ ] LLM-judge faithfulness (RAGAS)
- [ ] Hybrid retrieval (dense + BM25) to address short-query weakness
- [ ] Query expansion to compensate for title-only queries
- [ ] Docker deployment
- [ ] Deploy to Hugging Face Spaces

## Acknowledgements

Data: [flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl), derived from the [Stack Exchange data dumps](https://archive.org/details/stackexchange).

Licensed **CC-BY-NC-SA 4.0**. This is a non-commercial portfolio project. Original Stack Exchange content is CC-BY-SA.
