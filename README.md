# ChGK Dual Latent Embedding Rating System

Production-ready neural rating and tournament outcome prediction system for **What? Where? When? (Что? Где? Когда? / ChGK)**. Built on PyTorch and grounded in Multidimensional Item Response Theory (MIRT), collaborative filtering, and dynamic skill tracking principles.

---

## 1. Mathematical Architecture & Core Formulation

The rating system models ChGK as a collaborative team game with asymmetric knowledge profiles, non-linear team aggregation, pairwise team communication dynamics, and format-invariant question package difficulty.

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
  │ Package & Editor Model  │ ────► z_know = ⟨K_team, e_pkg⟩ - b_pkg(DL_intrinsic)
  └─────────────────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   Experience-Weighted   │ ────► S_team = 1/|Pairs| ∑_{i<m} (c_i^T W c_m) · (1 + β_exp tanh(log(1 + N_im))) · exp(-γ Δt_im)
  │     Synergy Matrix      │
  └─────────────────────────┘
               │
               ▼
   P(Solved) = σ(z_know + w_syn · S_team + b_format + z_anomaly)
```

### 1.1. Multidimensional IRT and Knowledge Space
Standard 1D Rasch or Elo models treat an entire team as a single scalar skill $\theta$. This system extends Multidimensional Item Response Theory (MIRT) to model individual players and tournament packages in latent factor space:
- **Player Base Knowledge Vector**: $k_i \in \mathbb{R}^D$ ($D = 6$: 1 general cognitive/analytical ability dimension + 5 macro-topic dimensions). Dimensionality reduction via SVD demonstrates that $D=6$ captures $82\%$ of variance while completely eliminating noise and overfitting on high-frequency synchronous players.
- **Online Divergence Component**: In remote online play (Discord/Zoom/asynchronous platforms), player performance can deviate due to search tools or unproctored taking:
  $$k_{i, \text{online}} = k_{i, \text{base}} + \Delta k_{i, \text{online}}$$
  where $\Delta k_{i, \text{online}}$ is subject to a $10\times$ $L_2$ Bayesian shrinkage penalty pulling toward zero. Only players with statistically significant and persistent divergence across multiple tournaments develop non-zero online delta vectors.
- **Editor / Package Vector**: $e_{\text{pkg}} \in \mathbb{R}^D$ ($D = 6$), pooling topic requirements and stylistic characteristics of the package editors.

### 1.2. Unconditional Format-Invariant Package Difficulty ($DL_{\text{intrinsic}}$)
The tournament website's raw `trueDL` metric is circular and format-polluted: because unproctored online taking enables higher solve rates, an identical package played synchronously vs. asynchronously receives a $\sim 0.64$ lower `trueDL` on the website.

The system decouples intrinsic question difficulty from play environment:
- **Synchronous / Offline Tournaments**: $DL_{\text{intrinsic}} = TrueDL_{\text{site}}$.
- **Asynchronous Mirror Packages**: Paired with their synchronous twins so that $DL_{\text{intrinsic}} = TrueDL_{\text{sync\_twin}}$.
- **Unpaired Online Asynchronous Packages**: $DL_{\text{intrinsic}} = TrueDL_{\text{site}} + 0.64$.
- **Stage 1 Bayesian Editor Calibration**:
  $$\beta_e = \frac{\sum (DL_{\text{intrinsic}} - DL_{\text{forecast}}) + n_0 \mu_0}{n + n_0}$$
  Predicting unconditional intrinsic tournament difficulty before the tournament is played:
  $$DL_{\text{calibrated}} = DL_{\text{forecast}} + \frac{1}{E} \sum_{e \in \text{Editors}} \beta_e$$
- **Intrinsic Difficulty Sensitivity in Model**:
  $$b_{\text{pkg}} = \bar{b}_{\text{editors}} + \alpha_{\text{dl}} \cdot (DL_{\text{intrinsic}} - DL_{\text{baseline}})$$

The higher solve rates of online tournaments are thus attributed strictly to the format baseline shift $b_{\text{format}}$, individual search factors $\Delta k_{i, \text{online}}$, and format-specific team communication $c_i(F)$, rather than artificially pretending the questions became easier.

### 1.3. Knowledge Pooling: Smooth-Max ($L_{\infty}$ Approximation)
A team solves a question if **at least one** player possesses the required insight or factual knowledge. Pure linear averaging ($\frac{1}{n} \sum k_i$) penalizes teams for carrying specialists or support players, while hard maximum ($\max_i k_i$) suffers from zero gradients for non-maximum players and ignores team depth:
$$K_{\text{team}, d} = k_{\max, d} + \tau \cdot \log \sum_{i=1}^{n} \exp\left(\frac{k_{i, d} - k_{\max, d}}{\tau}\right)$$
- $\tau = \text{softplus}(\text{raw\_temp}) + 0.05$: learnable temperature parameter.
- As $\tau \to 0$, Smooth-Max approaches exact hard maximum without penalizing incomplete rosters (1–5 players).

### 1.4. Bounded Synergy & Experience Saturation
Beyond individual knowledge, teams exhibit chemistry and discussion flow modeled via bilinear forms:
$$S_{im}(t) = \left(c_i(F)^\top W_{\text{synergy}} c_m(F)\right) \cdot \left(1 + \beta_{\text{exp}} \tanh(\log(1 + N_{im}))\right) \cdot \exp(-\gamma_{\text{syn}} \Delta t_{im})$$
- **Communication Vectors**: $c_i(F) = c_{\text{base}, i} + e_F \odot c_{\delta, i} \in \mathbb{R}^M$ ($M = 3$ core team roles: captain/selector, generator, critic).
- **Intensive Pairwise Average**: Total team synergy is scaled by available pairs:
  $$S_{\text{team}} = \frac{1}{\max(1, \binom{n}{2})} \sum_{i < m} S_{im}$$
- **Bounded Synergy Weight**: $w_{\text{syn}} = 0.25 \cdot \sigma(\text{raw\_synergy\_weight}) \in [0, 0.25]$. Constrains chemistry to realistic human bounds ($\approx \pm 2.5$ questions on a 120-question package) so team chemistry can never overpower individual knowledge.
- **Experience Saturation**: $\tanh(\log(1 + N_{im}))$ models rapid initial familiarity over the first 2–3 games ($\approx 80\%$ of potential synergy) with a hard mathematical ceiling for veteran pairs.
- **Decay Half-Life**: Elapsed time since the pair's last joint game decays synergy with a learned half-life ($\approx 845$ days).

### 1.5. Inactivity Degradation & Skill Tracking
During player hiatuses ($\Delta t_i > 365$ days):
$$k_i(T_{\text{target}}) = k_i \odot \exp\left(-\vec{\lambda}_{\text{decay}} \max(0, \Delta t_i - G_{\text{grace}})\right)$$
- **Factual Knowledge**: Fast decay (learned half-life $\approx 1600$ days / $\approx 4.4$ years).
- **Logical / Analytical Ability**: Slow decay (learned half-life $\approx 5600$ days / $\approx 15$ years).
- **Grace Period**: $G_{\text{grace}} \approx 365.6$ days.

---

## 2. Scientific Literature & Citations

1. **Item Response Theory & Team Contests**:
   - *Rasch, G. (1960).* Probabilistic Models for Some Intelligence and Attainment Tests. *Danish Institute for Educational Research*. [DOI:10.1037/h0040776](https://doi.org/10.1037/h0040776)
   - *Reckase, M. D. (2009).* Multidimensional Item Response Theory. *Springer Science & Business Media*. [DOI:10.1007/978-0-387-89976-3](https://link.springer.com/book/10.1007/978-0-387-89976-3)
   - *Herbrich, R., Minka, T., & Graepel, T. (2007).* TrueSkill™: A Bayesian Skill Rating System. *NeurIPS 2006*, 19, 569–576. [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf)
   - *Nikolenko, S. I. (2020).* Collaborative Filtering and Item Response Theory for Team Competitions: Rating System for What? Where? When?. *arXiv:2008.06975*. [arXiv:2008.06975](https://arxiv.org/abs/2008.06975)

2. **Synergy, Smooth-Max Pooling, and Bilinear Forms**:
   - *Blume, L. E. (1993).* The Statistical Mechanics of Strategic Interaction. *Games and Economic Behavior*, 5(3), 387–424. [DOI:10.1006/game.1993.1023](https://doi.org/10.1006/game.1993.1023)
   - *Koren, Y. (2008).* Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Model. *ACM SIGKDD (KDD '08)*, 426–434. [DOI:10.1145/1401890.1401944](https://doi.org/10.1145/1401890.1401944)
   - *He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).* Neural Collaborative Filtering. *WWW '17*, 173–182. [DOI:10.1145/3038912.3052569](https://doi.org/10.1145/3038912.3052569)

3. **Memory Decay & Temporal Skill Tracking**:
   - *Ebbinghaus, H. (1885).* Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie. *Duncker & Humblot*.
   - *Glickman, M. E. (1999).* Parameter Estimation in Large Dynamic Paired Comparison Experiments. *Journal of the Royal Statistical Society: Series C*, 48(3), 377–394. [DOI:10.1111/1467-9876.00159](https://doi.org/10.1111/1467-9876.00159)
   - *Koren, Y. (2009).* Collaborative Filtering with Temporal Dynamics. *Communications of the ACM*, 53(4), 89–97. [DOI:10.1145/1721654.1721677](https://doi.org/10.1145/1721654.1721677)

---

## 3. Data Sources & Ingestion Pipeline

The system continuously synchronizes two official community data sources:

1. **Official Rating Site API** ([`api.rating.chgk.info`](https://api.rating.chgk.info/)):
   - **Tournament Coverage**: **7,104 tournaments** spanning 2020–2026.
   - **Records**: **18,101,036 team question responses** stored in `data/dataset.parquet`.
   - **Metadata & Cache**: SQLite response cache (`data/chgk_cache.db`) containing raw tournament cards, question quantities, editor lists, rosters, and mask strings.
   - **Incremental Synchronization**: Automatically tracks `rating_last_sync` timestamp and invalidates cache only for edited tournaments.

2. **GotQuestions Knowledge Base** ([`gotquestions.online`](https://gotquestions.online/)):
   - **Packages**: **6,978 packages** in `data/gotquestions_packs.parquet`.
   - **Questions**: **518,454 questions** with texts, accepted answers, authors, and per-tour editors in `data/gotquestions_questions.parquet`.
   - **Linked Coverage**: **110,261 questions** directly matched between rating observations and question texts/author profiles.
   - **Polite Ingestion**: Rate-limited crawler with automatic backoff and immediate halt on HTTP 429.

---

## 4. Empirical Evaluation & Test Metrics

Model evaluated on **10,442,150 training observations** (95%), **274,793 validation observations** (2.5%), and **274,794 out-of-time test observations** (2.5%):

| Metric | Baseline Linear Rasch / Elo | Previous High-Dim Model | Dual Latent ($D=6, M=3, DL_{\text{intrinsic}}$) |
| :--- | :---: | :---: | :---: |
| **Test ROC-AUC** | 0.5980 | 0.6898 | **0.6923** |
| **Test LogLoss** | 0.6720 | 0.6358 | **0.6339** |
| **Test Brier Score** | 0.2410 | 0.2227 | **0.2218** |
| **Best Val LogLoss** | 0.6800 | 0.6436 | **0.6421** |

---

## 5. Quickstart & CLI Usage

### 5.1. Installation & Environment Setup
```bash
# Python 3.13 virtual environment
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run complete test suite (38 passing tests)
pytest tests/
```

### 5.2. Incremental Data Updates
```bash
# Incrementally update tournament results from api.rating.chgk.info
python cli.py update-rating --lookback-days 14

# Incrementally sync question packages from gotquestions.online
python cli.py fetch-questions --lookback-packs 20 --rate-limit 2.0
```

### 5.3. Training the Model
```bash
# Train on CPU or Apple Silicon
python cli.py train --dim-k 6 --dim-c 3 --epochs 5 --batch-size 4096 --device cpu
```

### 5.4. Universal Tournament Prediction (Any Tournament ID)
Predict tournament standings, expected score distribution, win probabilities, and generate interactive HTML reports for **any tournament ID**:

```bash
# Predict via CLI command
python cli.py predict-tournament 14149
python cli.py predict-tournament 10000

# Or run via universal prediction script
python scripts/predict_tournament.py 14149
python scripts/predict_tournament.py 10000 --simulations 10000
```

#### Available Prediction Options:
- `--simulations <N>`: Number of Monte Carlo simulation runs (default: 10,000).
- `--tactical / --no-tactical`: Optimize 6-person lineups per tour for teams with substitutions (>6 players).
- `--refresh`: Force refresh tournament data and rosters from the official API.
- `--dl-override <FLOAT>`: Override package TrueDL for sensitivity analysis.
- `--html / --no-html`: Automatically generates a responsive HTML report (`data/forecast_{ID}.html`).

### 5.5. REST API Service
```bash
python cli.py run-api --host 0.0.0.0 --port 8000
```
OpenAPI documentation available at `http://localhost:8000/docs`.

---

## 6. License & Community Guidelines

Released under the MIT License. Data ingested from `rating.chgk.info` and `gotquestions.online` is subject to the respective terms and fair use policies of the community organizers.
