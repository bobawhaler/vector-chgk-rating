# ChGK Dual Latent Embedding Rating System

A production-ready Python neural rating system for **What? Where? When? (Что? Где? Когда? / ChGK)** based on a **Dual Latent Embedding Neural Rating Model** implemented in PyTorch, featuring **Time Decay & Dynamic Skill Tracking**.

The system models individual player knowledge skill, communication base vectors, format-dependent sensitivity (offline, sync, online), question requirements, difficulty scalar bias, softplus discrimination factors, pairwise team communication synergy, and temporal dynamics (inactivity degradation, asymmetric knowledge decay, synergy erosion, and Brownian variance expansion).

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

---

## Dynamic Time Decay & Temporal Tracking

### 1. Inactivity & Asymmetric Knowledge Decay
For a player $i$ evaluated at target timestamp $T_{target}$, given their last active tournament date $T_{last\_i}$:
$$\Delta t_i = \max(0, T_{target} - T_{last\_i}) \quad (\text{in days})$$
$$k_i(T_{target}) = k_i^{model} \cdot \exp\left(-\vec{\lambda}_{decay} \odot \max(0, \Delta t_i - G_{grace})\right)$$
- **Grace Period ($G_{grace}$)**: 365 days (1 year) where no skill decay occurs.
- **Asymmetric Knowledge Decay**:
  - Dimensions $0 \dots 7$: **Factual Knowledge** (fast decay: half-life $H_{fact} \approx 730$ days, $\approx 2$ years).
  - Dimensions $8 \dots 15$: **Logical Reasoning & Erudition** (slow decay: half-life $H_{logic} \approx 2920$ days, $\approx 8$ years).

### 2. Pairwise Team Synergy Decay
For any pair of players $(i, m)$ with elapsed time $\Delta t_{im} = \max(0, T_{target} - T_{joint\_last}(i, m))$ since their last joint game:
$$S_{im}(T_{target}) = \left(c_i(F)^\top W_{synergy} c_m(F)\right) \cdot \exp\left(-\gamma_{syn} \cdot \Delta t_{im}\right)$$
- $\gamma_{syn}$: Synergy decay rate with half-life $H_{syn} \approx 540$ days ($\approx 1.5$ years).

### 3. Brownian Variance Expansion (Training Uncertainty)
During training, returning inactive players have their updates scaled by their inactivity gap:
$$\eta_i(t) = \eta_0 \cdot \sqrt{1.0 + \alpha_{gap} \cdot \frac{\Delta t_i}{365.0}}$$

---

## Forward Pass Pipeline for Team $T$ on Question $j$ in Format $F$:
1. **Team Knowledge Pooling**:
   $$K_{team}(t) = \max_{i \in T} \{k_i(t)\} \in \mathbb{R}^D$$
2. **Raw Generation Probability**:
   $$z_{know} = a_j \cdot (\langle K_{team}(t), q_j \rangle - b_j)$$
   $$P_{gen} = \sigma(z_{know})$$
3. **Team Synergy Matrix & Total Synergy**:
   $$S_{im}(t) = c_i(F)^T W_{synergy} c_m(F) \cdot \exp(-\gamma_{syn} \cdot \Delta t_{im})$$
   $$S_{team}(t) = \sum_{i < m} S_{im}(t)$$
4. **Online Anomaly Adjustment** (for online format):
   $$z_{anomaly} = \delta_{cheat} \quad (\text{if } F = \text{'online'}, \text{else } 0.0)$$
5. **Final Solved Probability**:
   $$P_{final} = \sigma(\text{logit}(P_{gen}) + S_{team}(t) + z_{anomaly}) = \sigma(z_{know} + S_{team}(t) + z_{anomaly})$$

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
│   ├── time_decay.py          # TimeDecayModule, asymmetric decay, synergy erosion
│   ├── data_loader.py         # Async ingestion, SQLite caching, Parquet conversion, synthetic generator
│   ├── model.py               # PyTorch DualEmbeddingChGKModel with time decay
│   ├── dataset.py             # Chronological state tracking, Delta_t calculation, indexers
│   ├── train.py               # Training engine, AdamW, Cosine scheduler, LogLoss, ROC-AUC, Brier
│   ├── analytics.py           # Historical timelines, top players, synergy matrix, anomaly detection
│   ├── app.py                 # FastAPI application with timeline and match_date support
│   └── cli.py                 # Click CLI interface
└── tests/
    ├── test_analytics.py      # Analytics & inspection tests
    ├── test_api_cli.py        # REST API & CLI tests
    ├── test_data_loader.py    # Caching & dataset generation tests
    ├── test_model.py          # Model architecture & math tests
    ├── test_time_decay.py     # Inactivity, synergy erosion, Brownian variance, time-travel tests
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

# Or generate synthetic multi-year benchmark data offline
python cli.py fetch-data --synthetic
```

---

## Training & Evaluation

The training engine splits data chronologically (80% Train, 10% Validation, 10% Test), trains with `AdamW` and `CosineAnnealingLR`, regularizes embeddings with L2 weight decay, checkpoints best validation LogLoss, and outputs LogLoss, ROC-AUC, and Brier score:

```bash
# Train model with time decay
python cli.py train --dim-k 16 --dim-c 8 --epochs 20 --batch-size 4096

# Evaluate checkpointed model on test set
python cli.py evaluate
```

---

## Model Inspection & Analytics

### 1. Player Historical Skill Timeline & Inactivity Curve
Inspect how player skill evolves and how factual knowledge decays faster than logical reasoning during hiatuses:
```bash
python cli.py player-timeline --player-id 1 --start-date 2022-01-01 --end-date 2026-09-01 --step-days 180
```

### 2. Team Synergy & Time-Decayed Question Prediction
Predict question outcome and pairwise synergy with inactivity decay active for custom date:
```bash
python cli.py predict-team --players 1,2,3,4,5,6 --format offline --match-date 2026-09-01 --question-id 1001_1
```

### 3. Top Players by Knowledge Vector Norm $\|k_i\|$
```bash
python cli.py top-players --n 15
```

### 4. Detect Online Anomalies ($\delta_{cheat}$)
```bash
python cli.py detect-anomalies
```

### 5. Export Embeddings for UMAP / t-SNE
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
  "player_ids": ["1", "2", "3", "4", "5", "6"],
  "question_ids": ["1001_1", "1001_2"],
  "format": "offline",
  "match_date": "2026-09-01"
}

// Response
{
  "players": ["1", "2", "3", "4", "5", "6"],
  "format": "offline",
  "match_date": "2026-09-01",
  "total_synergy": -0.0047,
  "predictions": [
    {
      "question_id": "1001_1",
      "solved_probability": 0.2772,
      "raw_generation_prob": 0.2782,
      "knowledge_logit": -0.9536,
      "team_synergy": -0.0047
    }
  ]
}
```

#### 2. `POST /team_synergy`
```json
// Request
{
  "player_ids": ["1", "2", "3", "4", "5", "6"],
  "format": "offline",
  "match_date": "2026-09-01"
}

// Response
{
  "format": "offline",
  "target_date": "2026-09-01",
  "total_synergy": -0.0047,
  "players": ["1", "2", "3", "4", "5", "6"],
  "synergy_matrix": [...],
  "pairwise_rankings": [
    {"player_1": "3", "player_2": "4", "synergy_score": 0.0139}
  ],
  "optimal_seating_order": ["1", "3", "4", ...]
}
```

#### 3. `GET /player/{id}/historical_timeline`
Query parameters: `start_date`, `end_date`, `step_days`.
Returns trajectory data points:
```json
{
  "player_id": "1",
  "timeline": [
    {
      "date": "2022-01-01",
      "days_inactive": 0.0,
      "is_active": true,
      "effective_knowledge_norm": 0.3815,
      "factual_norm": 0.2110,
      "logical_norm": 0.3179,
      "retention_ratio": 1.0
    },
    {
      "date": "2024-06-19",
      "days_inactive": 900.0,
      "is_active": false,
      "effective_knowledge_norm": 0.3074,
      "factual_norm": 0.1269,
      "logical_norm": 0.2800,
      "retention_ratio": 0.806
    }
  ]
}
```

#### 4. `GET /player/{id}/profile`
Returns player's latent vector norms, format sensitivity, and top synergistic teammates.
