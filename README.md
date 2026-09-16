# ChGK Dual Latent Embedding Rating System

A production-ready Python neural rating system for **What? Where? When? (Что? Где? Когда? / ChGK)** based on a **Dual Latent Embedding Neural Rating Model** implemented in PyTorch.

The model models individual player knowledge skill, communication base vectors, format-dependent sensitivity (offline, sync, online), question requirements, difficulty scalar bias, softplus discrimination factors, and pairwise team communication synergy.

---

## Mathematical Architecture

Each player $i$ is represented by a dual trainable embedding:
- **Knowledge Latent Embedding**: $k_i \in \mathbb{R}^D$ ($D = 16$)
- **Communication Base Embedding**: $c_{base, i} \in \mathbb{R}^M$ ($M = 8$)
- **Format Sensitivity Embedding**: $c_{\delta, i} \in \mathbb{R}^M$

Each question $j$ is represented by:
- **Requirement Latent Embedding**: $q_j \in \mathbb{R}^D$
- **Difficulty Scalar Bias**: $b_j \in \mathbb{R}^1$
- **Discrimination Factor**: $a_j = \text{softplus}(\hat{a}_j) \in \mathbb{R}^+$

Format Constraint ($F \in \{\text{offline}, \text{sync}, \text{online}\}$):
- **Format Embedding**: $e_F \in \mathbb{R}^M$
- **Effective Communication Vector**: $c_i(F) = c_{base, i} + e_F \odot c_{\delta, i}$

### Forward Pass Pipeline for Team $T$ on Question $j$ in Format $F$:
1. **Team Knowledge Pooling**:
   $$K_{team} = \max_{i \in T} \{k_i\} \in \mathbb{R}^D$$
2. **Raw Generation Probability**:
   $$z_{know} = a_j \cdot (\langle K_{team}, q_j \rangle - b_j)$$
   $$P_{gen} = \sigma(z_{know})$$
3. **Team Synergy Matrix**:
   $$S_{im} = c_i(F)^T W_{synergy} c_m(F) \quad \text{for } i < m$$
   $$S_{team} = \sum_{i < m} S_{im}$$
4. **Online Anomaly Adjustment** (for online format):
   $$z_{anomaly} = \delta_{cheat} \quad (\text{if } F = \text{'online'}, \text{else } 0.0)$$
5. **Final Solved Probability**:
   $$P_{final} = \sigma(\text{logit}(P_{gen}) + S_{team} + z_{anomaly}) = \sigma(z_{know} + S_{team} + z_{anomaly})$$

---

## Project Structure

```
rating/
├── cli.py                     # Root CLI proxy
├── pyproject.toml             # Package definitions & dependencies
├── requirements.txt           # Pip dependencies
├── README.md                  # Documentation
├── chgk_rating/
│   ├── __init__.py
│   ├── config.py              # Configuration & format mappings
│   ├── data_loader.py         # Async ingestion, SQLite caching, Parquet conversion, synthetic generator
│   ├── model.py               # PyTorch DualEmbeddingChGKModel
│   ├── dataset.py             # PyTorch Dataset and bi-directional ID indexers
│   ├── train.py               # Training engine, AdamW, Cosine scheduler, LogLoss, ROC-AUC, Brier
│   ├── analytics.py           # Top players, team synergy matrix, online anomaly detection, UMAP/t-SNE exporter
│   ├── app.py                 # FastAPI application
│   └── cli.py                 # Click CLI interface
└── tests/
    ├── test_analytics.py      # Analytics & inspection tests
    ├── test_api_cli.py        # REST API & CLI tests
    ├── test_data_loader.py    # Caching & dataset generation tests
    ├── test_model.py          # Model architecture & math tests
    └── test_train.py          # Training loop & metric tests
```

---

## Quickstart & Environment Setup

### 1. Initialize Virtual Environment with `uv` (Python 3.13)
```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v
```

---

## Data Ingestion & Caching

The ingestion system asynchronously queries `https://api.rating.chgk.info/` with token-bucket rate limiting (10 req/s), exponential backoff, SQLite caching (`data/chgk_cache.db`), and flattens into Parquet (`data/dataset.parquet`):

```bash
# Fetch live data starting from date
python cli.py fetch-data --start-date 2023-01-01 --max-tournaments 30

# Or generate synthetic benchmark data offline
python cli.py fetch-data --synthetic
```

---

## Training & Evaluation

The training engine splits data chronologically (80% Train, 10% Validation, 10% Test), trains with `AdamW` and `CosineAnnealingLR`, regularizes embeddings with L2 weight decay, checkpoints best validation LogLoss, and outputs LogLoss, ROC-AUC, and Brier score:

```bash
# Train model
python cli.py train --dim-k 16 --dim-c 8 --epochs 20 --batch-size 4096

# Evaluate checkpointed model on test set
python cli.py evaluate
```

---

## Model Inspection & Analytics

### 1. Top Players by Knowledge Vector Norm $\|k_i\|$
```bash
python cli.py top-players --n 15
```

### 2. Team Synergy & Question Prediction
```bash
python cli.py predict-team --players 27403,28751,21026,74001,40393,7008 --format offline --question-id 1001_1
```

### 3. Detect Online Anomalies ($\delta_{cheat}$)
Detects teams whose online performance exceeds their offline baseline:
```bash
python cli.py detect-anomalies
```

### 4. Export Embeddings for UMAP / t-SNE
Exports embeddings to `player_embeddings.parquet`, `player_embeddings.csv`, `question_embeddings.parquet`, and `question_embeddings.csv`:
```bash
python cli.py export-embeddings --output-dir data/embeddings
```

---

## REST API (FastAPI)

### Start API Server
```bash
python cli.py run-api --host 127.0.0.1 --port 8000
```
Interactive Swagger docs available at `http://127.0.0.1:8000/docs`.

### Key Endpoints

#### 1. `POST /predict_match`
```json
// Request
{
  "player_ids": ["27403", "28751", "21026", "74001", "40393", "7008"],
  "question_ids": ["1001_1", "1001_2"],
  "format": "offline"
}

// Response
{
  "players": ["27403", "28751", "21026", "74001", "40393", "7008"],
  "format": "offline",
  "total_synergy": 0.0113,
  "predictions": [
    {
      "question_id": "1001_1",
      "solved_probability": 0.2972,
      "raw_generation_prob": 0.2948,
      "knowledge_logit": -0.8721,
      "team_synergy": 0.0113
    }
  ]
}
```

#### 2. `POST /team_synergy`
```json
// Request
{
  "player_ids": ["27403", "28751", "21026", "74001", "40393", "7008"],
  "format": "online"
}

// Response
{
  "format": "online",
  "total_synergy": 0.0152,
  "players": ["27403", "28751", "21026", "74001", "40393", "7008"],
  "synergy_matrix": [...],
  "pairwise_rankings": [
    {"player_1": "74001", "player_2": "7008", "synergy_score": 0.0140}
  ],
  "optimal_seating_order": ["27403", "74001", "7008", ...]
}
```

#### 3. `GET /player/{id}/profile`
Returns player's latent vector norms, format sensitivity, and top synergistic teammates.
