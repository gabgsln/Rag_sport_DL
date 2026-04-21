# RAG System — Sports Science Q&A
**Deep Learning Project — Centrale Lyon 2025-2026**

---

## What is this?

I built a RAG (Retrieval-Augmented Generation) system around sports science. The idea is simple: instead of asking a generic LLM questions about sprint biomechanics or athlete recovery, the system first retrieves relevant chunks from a curated knowledge base, then generates an answer grounded in that evidence.

The knowledge base covers sprint mechanics, ankle biomechanics, plyometrics, nutrition, recovery, and profiles of athletes like Mbappé, Lamine Yamal, Noah Lyles and Usain Bolt.

---

## How to run

```bash
pip install -r requirements.txt
python3 rag_pipeline.py       # runs the full pipeline with evaluation
python3 demo_interactive.py   # interactive Q&A in the terminal
```

---

## Project structure

```
rag_submission/
├── rag_pipeline.py         # main pipeline: data → embeddings → ChromaDB → LLM → eval
├── demo_interactive.py     # terminal Q&A interface
├── rag_notebook.ipynb      # step-by-step notebook walkthrough
├── requirements.txt
├── data/
│   └── sports_science_corpus.json   # 114 documents
└── chroma_db/              # vector store, auto-created on first run (not tracked)
```

---

## Technical choices

### Dataset
114 documents built manually — 51 scientific studies and 63 evidence chunks. Chunked with `RecursiveCharacterTextSplitter` (size 500, overlap 50).

### Embeddings
I compared two models:

| Model | Dimensions | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | main model, fast and accurate on English |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | tested as alternative, supports French too |

### Vector database
ChromaDB with cosine similarity and HNSW index. The collection is rebuilt on each run to stay in sync with the corpus.

### LLMs
Three options depending on what's available:

| Model | Size | How |
|---|---|---|
| `google/flan-t5-base` | 250M | runs fully local, default |
| `google/flan-t5-large` | 770M | local, slower |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | via HuggingFace free API, needs `HF_TOKEN` |

### Prompt templates
I tested three approaches and compared the outputs:
- **XML-style** — the format from the course examples
- **Structured** — frames the context as research excerpts, more neutral
- **Minimal** — just context + question, no framing

The structured template gave noticeably better answers, especially for open-ended questions.

---

## Results

| Metric | Value |
|---|---|
| Retrieval Precision@3 | ~0.75 avg across 8 test queries |
| ChromaDB query latency | ~1.2ms |
| Index build (114 docs) | ~2s |
| flan-t5-base generation | 2–4s per query |

---

## What I learned

Retrieval quality is solid — both embedding models found the right chunks. The bottleneck is clearly the generation side: flan-t5 is limited on open-ended questions, and Mistral-7B is a big step up when you have access to it. Prompt framing makes a real difference too, especially telling the model it's a sports science expert before asking.
