# 🚀 CosmoCite

> A space-biology research engine that reads 572 scientific papers, answers questions with cited evidence, and refuses to answer when the evidence is insufficient.

🔗 **Built with:** Python · FastAPI · PyTorch · Hugging Face Transformers · React · TypeScript

---

## 📌 Overview

**CosmoCite** transforms space-biology publications into a searchable, explorable, and evidence-grounded research system.

Starting with 607 PMC links from the `SB_publications` list, 572 successfully parsed papers produce **20,242 searchable passages**. The system combines scientific text processing, classification, named entity recognition, embeddings, information retrieval, and question answering in one application.

Unlike a conventional chatbot, CosmoCite retrieves evidence from the research corpus and links answers directly to their original publications. Its default engine selects sentences from the papers rather than freely generating claims. When the evidence is insufficient, it refuses to answer instead of guessing.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Cited Question Answering** | Selects relevant sentences from retrieved papers and links them to their sources |
| **Evidence Check** | Uses a retrieval threshold to determine whether a question can be answered |
| **Relevance Filtering** | Filters aims, methods, generic descriptions, and sentences unrelated to the requested tissue and stressor |
| **Organism Preference** | Prioritises evidence about the organism in the question |
| **Local Model Option** | Optional Ollama generation with source-consistency checks |
| **Decoding Lab** | Compares greedy, beam, top-k, and nucleus decoding with FLAN-T5 |
| **Full Citations** | Displays publication details and supports text and BibTeX copying |
| **Closest Papers** | Suggests related publications when evidence is insufficient |
| **Show in Context** | Opens the original passage with selected sentences highlighted |
| **Four Retrievers** | Compares BM25, dense, hybrid, and cross-encoder retrieval |
| **Paper Explorer** | Filters papers by organism, stressor, and year; displays summaries and related papers |
| **Research Gap Map** | Visualises organism × stressor publication counts |
| **Passage Analyzer** | Predicts section labels, visualises attention, and identifies entities |
| **Results Dashboard** | Displays model metrics, comparisons, and experiment plots |

---

## 🧠 Models

| Task | Baseline | Improved / Final |
|---|---|---|
| Section classification | TF-IDF + Logistic Regression | BiLSTM / GRU + attention · SciBERT |
| NER | Rules and dictionary | BioBERT |
| Titles and summaries | Key-sentence extraction | LSTM attention · T5 (implemented, not evaluated) |
| Retrieval | BM25 | Dense · Hybrid · Cross-encoder |
| Question answering | FLAN-T5 | Ollama · Evidence-checked sentence ranking |

- **Embeddings:** `all-MiniLM-L6-v2`
- **Reranking:** `ms-marco-MiniLM-L-6-v2`
- **Optional local model:** `qwen2.5:0.5b`
- **Default answering:** Direct sentence selection with evidence checking.

---

## 🔁 Pipeline

```text
SB_publications
        │
        ▼
    607 PMC links
        │
        ▼
 Fetch → Parse → Clean → Chunk
        │
        ▼
  572 Papers · 20,242 Passages
        │
        ├── REPRESENTATION
        │   TF-IDF · BM25 · Word2Vec · Embeddings
        │
        ├── TASK MODELS
        │   Classification · NER · Seq2seq
        │
        ├── EXPLORATION
        │   Summaries · t-SNE · Research gap map
        │
        └── RETRIEVAL & ANSWERING
            BM25 → Dense → Hybrid → Reranking
            → Evidence check → Sentence selection
            → Cited answer
        │
        ▼
 FastAPI Backend → React + TypeScript Frontend
```

---

## 🚀 How to Run

### Prerequisites

Python 3.11+, Node.js 22+, and approximately 16 GB RAM recommended. Ollama is optional.

### 1. Clone and install

```bash
git clone https://github.com/Srejoye/CosmoCite.git
cd CosmoCite

pip install -r requirements.txt
```

### 2. Generate embeddings

The processed corpus is included. Generate the passage embeddings locally:

```bash
python backend/features.py embed
```

### 3. Start the backend

```bash
uvicorn api:app --app-dir backend --reload --port 8000
```

### 4. Start the frontend

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite, usually **http://localhost:5173**.

API documentation: `http://localhost:8000/docs`

### Optional: Local model

```bash
ollama pull qwen2.5:0.5b
```

Start Ollama to use the optional local generation engine. The default sentence-ranking engine does not require it.

### Optional: Docker

```bash
docker compose up --build
```

Open **http://localhost:8000**.

See [`docs/deploy.md`](docs/deploy.md) for deployment and configuration.

---

## 🖥️ Pages and API

| Page | Endpoint | Purpose |
|---|---|---|
| Home | `GET /api/stats` | Corpus overview |
| Ask | `POST /api/ask` | Cited answers and evidence-based refusals |
| Search | `GET /api/search` | Compare retrieval methods |
| Explore | `GET /api/papers` | Browse and filter publications |
| Explore | `GET /api/gaps` | Organism × stressor map |
| Analyze | `POST /api/analyze` | Section classification and entity extraction |
| Results | `GET /api/results` | Evaluation metrics and plots |
| Health | `GET /api/health` | Application readiness |

### Example

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does microgravity affect bone density in mice?"}'
```

The response includes the answer, evidence status, citations, supporting passages, and answering engine.

The default engine is `rank`. Optional engines are `synth` (Ollama) and `flan` (FLAN-T5).

---

## 📊 Evaluation

All experiments use a shared paper-level split, configured in `config.yaml`.

| Component | Metric | Result |
|---|---|---:|
| Section classification | SciBERT macro-F1 | **0.691** |
| NER | Rule-based entity F1 | **0.650** |
| Retrieval | Dense nDCG@5 (12 judged questions, labels by Claude) | **0.817** |
| Retrieval | Reranked MRR (30 known-item questions) | **0.903** |
| Seq2seq | Luong attention ROUGE-L | **0.154** |
| Answer relevance | On-topic sentences after filtering (judged by Claude, an estimate) | **51/62 (82%)** |
| Source faithfulness | Sampled sentences matching source passages | **60/60** |
| Natural questions | Answered by default engine | **28/30** |

The relevance labels (answer relevance and the judged retrieval set) were assigned by an AI model, Claude, not by human annotators, and the sentence filter was developed on the same questions used to measure it. Treat those figures as rough estimates; see [`docs/report.md`](docs/report.md).

Detailed metrics, comparisons, and plots are available in `reports/`.

---

## 🗂️ Project Structure

```text
CosmoCite/
├── backend/
│   ├── api.py
│   ├── fetch.py
│   ├── data.py
│   ├── features.py
│   ├── retrieval.py
│   ├── index.py
│   ├── ingest.py
│   ├── classify.py
│   ├── ner.py
│   ├── seq2seq.py
│   ├── generate.py
│   ├── evidence.py
│   ├── llm.py
│   ├── metadata.py
│   ├── summaries.py
│   ├── build.py
│   ├── evaluate.py
│   ├── bench.py
│   ├── labels.py
│   └── loadtest.py
├── frontend/
│   └── src/
│       ├── components/
│       └── pages/
├── data/
│   ├── SB_publication_PMC.csv
│   ├── eval/
│   └── processed/
├── reports/
├── tests/
├── docs/
│   ├── deploy.md
│   └── report.md
├── config.yaml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🎨 Tech Stack

| Technology | Role |
|---|---|
| Python · PyTorch | Data processing, NLP, and model training |
| Hugging Face Transformers | SciBERT, BioBERT, FLAN-T5, and T5 |
| Sentence Transformers | Dense embeddings and cross-encoder reranking |
| scikit-learn · Gensim | Traditional ML and Word2Vec |
| NumPy · FAISS | Vector indexing and search |
| FastAPI | Backend API |
| React · TypeScript · Vite | Frontend |
| Ollama | Optional local generation |
| Docker | Deployment |
| pytest · Vitest · Playwright | Testing |

---

## 📖 Documentation

- [`docs/report.md`](docs/report.md) — Methodology, experiments, and results.
- [`docs/deploy.md`](docs/deploy.md) — Deployment and configuration.

---

## ⭐ Support

If you found this project useful or interesting, consider giving it a ⭐ on GitHub.