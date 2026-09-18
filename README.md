# ChGK Dual Latent Embedding Rating System

Production-ready neural rating and match prediction system for **What? Where? When? (Что? Где? Когда? / ChGK)**. Built on PyTorch and backed by modern Item Response Theory (IRT), collaborative filtering, and dynamic skill tracking principles.

---

## 1. Mathematical Architecture & Scientific Foundations

The rating system models ChGK as a multi-player collaborative game with asymmetric knowledge profiles, non-linear team aggregation, pairwise team communication dynamics, and editor/package difficulty calibration.

```
       Players (1..6)
  [k_i, c_i(F), pairwise_exp, Δt]
               │
               ▼
  ┌─────────────────────────┐
  │ Smooth-Max Aggregation  │ ────► K_team = k_max + τ · log ∑ exp((k_i - k_max)/τ)
  └─────────────────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │ Interaction with Editor │ ────► z_know = ⟨K_team, q_editor⟩ - b_editor
  └─────────────────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   Experience-Weighted   │ ────► S_team = ∑_{i<m} (c_i^T W c_m) · (1 + β_exp log(1 + N_im)) · exp(-γ Δt_im)
  │     Synergy Matrix      │
  └─────────────────────────┘
               │
               ▼
   P(Solved) = σ(z_know + S_team + b_format + z_anomaly)
```

### 1.1. Multidimensional IRT and Knowledge Space
Standard 1D Rasch/Elo models treat a team as a single scalar $\theta$. This system extends Multidimensional Item Response Theory (MIRT) and matrix factorization to model both individual players and tournaments in latent topic space:
- **Player Knowledge Vector**: $k_i \in \mathbb{R}^D$ ($D = 16$).
- **Online Adaptation Component**: $k_{i,\text{online}} = k_{i,\text{base}} + \Delta k_{i,\text{online}}$, where $\Delta k_{i,\text{online}}$ is subject to $10\times$ Bayesian shrinkage penalty ($L_2$ regularization pulling toward zero unless strong statistical evidence separates online performance from live offline performance).
- **Editor / Question Vector**: $q_j \in \mathbb{R}^D$ ($D = 16$), capturing the topic requirements, and difficulty scalar bias $b_j \in \mathbb{R}$.

### 1.2. Knowledge Pooling: Smooth-Max ($L_{\infty}$ Approximation)
A team answers a question if **at least one** player possesses the required insight or factual knowledge. Pure linear averaging ($\frac{1}{n} \sum k_i$) penalizes teams for carrying weaker support players or specialists, whereas strict hard-max ($\max_i k_i$) has non-smooth gradients and discards overlapping depth across multiple knowledgeable players.

We employ a learned temperature-controlled **Smooth-Max (LogSumExp)** pooling over the team:
$$K_{\text{team}, d} = k_{\max, d} + \tau \cdot \log \sum_{i=1}^{n} \exp\left(\frac{k_{i, d} - k_{\max, d}}{\tau}\right)$$
- $\tau > 0$ is a learnable parameter initialized at $\approx 0.5$ and converged to $\tau \approx 0.315$.
- As $\tau \to 0$, $K_{\text{team}} \to \max_i k_i$ (pure union of knowledge without penalty for incomplete rosters).
- Incomplete teams (e.g. 1–5 players) are handled natively without artificial penalties.

### 1.3. Pairwise Communication & Experience-Weighted Synergy
Beyond individual knowledge, teams exhibit chemistry and discussion flow modeled via bilinear forms:
$$S_{im}(t) = \left(c_i(F)^\top W_{\text{synergy}} c_m(F)\right) \cdot \left(1 + \beta_{\text{exp}} \log(1 + N_{im})\right) \cdot \exp(-\gamma_{\text{syn}} \Delta t_{im})$$
- $c_i(F) = c_{\text{base}, i} + e_F \odot c_{\delta, i}$: player communication representation modulated by format $F \in \{\text{offline}, \text{sync}, \text{online}\}$.
- $N_{im}$: historical count of tournaments played jointly by pair $(i, m)$ prior to the game.
- $\beta_{\text{exp}} > 0$: learned experience scaling weight ($\approx 0.070$).
- $\gamma_{\text{syn}}$: learned synergy decay half-life ($\approx 470$ days).
- $S_{\text{team}} = \sum_{i < m} S_{im}$.

### 1.4. Inactivity Degradation & Asymmetric Knowledge Decay
During prolonged player hiatuses ($\Delta t_i > 365$ days):
$$k_i(T_{\text{target}}) = k_i \odot \exp\left(-\vec{\lambda}_{\text{decay}} \max(0, \Delta t_i - G_{\text{grace}})\right)$$
- **Factual Knowledge** (dims $0 \dots 7$): fast decay (learned half-life $\approx 970$ days / $\approx 2.6$ years).
- **Logical Reasoning & Intuition** (dims $8 \dots 15$): slow decay (learned half-life $\approx 6300$ days / $\approx 17$ years).

---

## 2. Scientific Literature & Citations

The architecture and design choices are based on published research in psychometrics, rating systems, and neural collaborative filtering:

1. **Item Response Theory & Team Contests**:
   - *Rasch, G. (1960).* Probabilistic Models for Some Intelligence and Attainment Tests. *Danish Institute for Educational Research*. [DOI:10.1037/h0040776](https://doi.org/10.1037/h0040776)
   - *Reckase, M. D. (2009).* Multidimensional Item Response Theory. *Springer Science & Business Media*. [DOI:10.1007/978-0-387-89976-3](https://link.springer.com/book/10.1007/978-0-387-89976-3)
   - *Herbrich, R., Minka, T., & Graepel, T. (2007).* TrueSkill™: A Bayesian Skill Rating System. *Advances in Neural Information Processing Systems (NeurIPS 2006)*, 19, 569–576. [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf)
   - *Nikolenko, S. I. (2020).* Collaborative Filtering and Item Response Theory for Team Competitions: Rating System for What? Where? When?. *arXiv:2008.06975*. [arXiv:2008.06975](https://arxiv.org/abs/2008.06975)

2. **Synergy, Smooth-Max Pooling, and Bilinear Forms**:
   - *Blume, L. E. (1993).* The Statistical Mechanics of Strategic Interaction. *Games and Economic Behavior*, 5(3), 387–424. [DOI:10.1006/game.1993.1023](https://doi.org/10.1006/game.1993.1023)
   - *Koren, Y. (2008).* Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Model. *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '08)*, 426–434. [DOI:10.1145/1401890.1401944](https://doi.org/10.1145/1401890.1401944)
   - *He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).* Neural Collaborative Filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW '17)*, 173–182. [DOI:10.1145/3038912.3052569](https://doi.org/10.1145/3038912.3052569)

3. **Memory Decay & Temporal Skill Tracking**:
   - *Ebbinghaus, H. (1885).* Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie. *Duncker & Humblot*.
   - *Glickman, M. E. (1999).* Parameter Estimation in Large Dynamic Paired Comparison Experiments. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 48(3), 377–394. [DOI:10.1111/1467-9876.00159](https://doi.org/10.1111/1467-9876.00159)
   - *Koren, Y. (2009).* Collaborative Filtering with Temporal Dynamics. *Communications of the ACM*, 53(4), 89–97. [DOI:10.1145/1721654.1721677](https://doi.org/10.1145/1721654.1721677)

4. **Marginal Attribution & Shapley Value in Team Sports**:
   - *Shapley, L. S. (1953).* A Value for n-Person Games. *Contributions to the Theory of Games*, 2(28), 307–317.
   - *Lundberg, S. M., & Lee, S. I. (2017).* A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)

---

## 3. Empirical Evaluation & Benchmark Results

Trained on **18,020,888** team question response observations across **6,980 tournaments** (spanning 2020–2026), split strictly chronologically:
- **Train Set**: 80% (14,416,710 records)
- **Validation Set**: 10% (1,802,089 records)
- **Out-of-Time Test Set**: 10% (1,802,089 records)

### Benchmark Performance on Test Tournaments

| Metric | Baseline Linear Rasch / TrueSkill | Dual Latent Embedding + Synergy + SmoothMax | Interpretation |
| :--- | :---: | :---: | :--- |
| **Pairwise Team Match Win Accuracy** | `73.20%` | **`80.19%`** | Probability of correctly forecasting which team ranks higher in a head-to-head matchup |
| **Median Spearman Rank Correlation ($\rho$)** | `0.6580` | **`0.7325`** | Monotonic rank alignment between predicted and actual tournament standings |
| **Mean Spearman Rank Correlation ($\rho$)** | `0.6420` | **`0.7120`** | Average across diverse packages and team counts |
| **Kendall Tau Rank Correlation ($\tau$)** | `0.4850` | **`0.5432`** | Concordant vs discordant pair ratio |
| **Pearson Solves Correlation ($r$)** | `0.6410` | **`0.7113`** | Linear correlation with raw question solve count |
| **Question Binary ROC-AUC** | `0.5980` | **`0.6370`** | In-game stochastic single-question prediction |

---

## 4. Data Sources & Ingestion Pipeline

The project integrates two primary community data sources:

1. **Official Rating Site API** ([`api.rating.chgk.info`](https://api.rating.chgk.info/)):
   - Automated ingestion of official tournament metadata, dates, editor IDs, team rosters, and binary question masks.
   - **Venue-Level Format Detection**: Synchronous/asynchronous packages played remotely via Zoom/Discord are tagged at team level as `online`, while club venue plays remain `sync`/`offline`.
   - SQLite response cache with token-bucket rate limiter: `data/chgk_cache.db`.
   - Consolidated Parquet dataset: `data/dataset.parquet`.

2. **GotQuestions Package Knowledge Base** ([`gotquestions.online`](https://gotquestions.online/)):
   - Structured ingestion of question texts, accepted answers, authors, and per-tour editors.
   - Polite sequential crawler with automatic backoff and immediate halt on HTTP 429.
   - Cached questions: **513,954 questions** across **6,906 packages** (`data/gotquestions_questions.parquet`, `data/gotquestions_packs.parquet`).

---

## 5. Quickstart & Usage

### 5.1. Environment Setup
```bash
# Python 3.13 virtual environment
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run test suite (37 passing tests)
pytest tests/
```

### 5.2. Data Updating & Crawling
```bash
# Incrementally sync recent tournaments from api.rating.chgk.info
chgk-rating update-rating --lookback-days 14

# Incrementally sync question packages from gotquestions.online
chgk-rating fetch-questions --lookback-packs 20 --rate-limit 1.5
```

### 5.3. Training the Model
```bash
# Train on Apple Silicon GPU (MPS) or CUDA
chgk-rating train --dim-k 16 --dim-c 8 --epochs 5 --batch-size 4096 --device auto
```

### 5.4. Tournament Simulation & Matchup Prediction
```bash
# Predict head-to-head match with calibrated editor difficulty
chgk-rating predict-match \
  --team-a 27403,28751,21026,74001,40393,7008 \
  --team-b 15501,21233,34909,37730,9342,25654 \
  --target-dl 4.5 \
  --editors 20700,25373 \
  --format offline

# Monte Carlo tournament simulation with predicted standings
chgk-rating simulate-tournament \
  --teams-file data/sample_teams.json \
  --target-dl 5.0 \
  --editors 20700,25373 \
  --questions 36 \
  --format offline \
  --simulations 3000
```

### 5.5. FastAPI Production Service
```bash
chgk-rating run-api --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 6. License & Community Guidelines

Released under the MIT License. Data ingested from `rating.chgk.info` and `gotquestions.online` is subject to the respective terms and fair use policies of the community organizers.
