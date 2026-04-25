# MOBA Draft Picker

A machine learning system for League of Legends professional draft analysis. The project trains a win probability model on completed team compositions and uses it to evaluate three winner-guided draft recommendation models.

**Authors:** Grace Ou, Jassem Alabdulrazaq  
**Dataset:** Oracle's Elixir — 2024 professional matches

---

## Overview

The draft phase in League of Legends determines which champions each team plays before the game begins. This project approaches draft recommendation as a two-stage pipeline:

1. **Win Probability Model** — trained on completed 5v5 compositions to predict Blue-side win probability. Logistic Regression, XGBoost, and a Feedforward Neural Network are compared against an always-Blue baseline.

2. **Draft Predictor** — three winner-guided models (MLP, LSTM, XGBoost) trained on winning-side draft actions only. A side indicator (Blue/Red) allows a single model to serve both sides. Evaluated by simulating 500 adversarial drafts and scoring the resulting compositions with the LR win probability model.

---

## Repository Structure

```
ML/
├── src/
│   ├── data_preprocessing.py             # Win probability feature encoding
│   ├── draft_data_preprocessing.py       # Draft sequence processing + winner subclass
│   ├── models_win_predictor.py           # AlwaysBlueBaseline, LR, XGBoost, FNN
│   ├── models_draft.py                   # WinnerDraftMLPModel, LSTMModel, XGBModel
│   ├── model_draft_mlp.py                # Plain draft MLP (no side indicator)
│   ├── train_all_models.py               # Train win probability models
│   ├── train_winner_model.py             # Train winner-guided draft models
│   ├── adversarial_evaluation_winner.py  # Adversarial simulation + evaluation
│   ├── plot_win_predictor_results.py     # Figures for win predictor results
│   └── plot_draft_predictor_results.py   # Figures for draft predictor results
│   └──Split2_2024.csv                   # Win predictor training data
│   └── 2024.csv                          # Draft model training data
├── data/
│   ├── metadata.pkl                      # Win predictor champion/patch encoders
│   └── draft_metadata.pkl               # Draft model champion vocab
└── models/

```

---

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### 1. Train win probability models

```bash
python src/train_all_models.py
```

### 2. Train winner-guided draft models

```bash
python src/train_winner_model.py
```

### 3. Run adversarial evaluation

Simulates 500 drafts per matchup. Each model plays both Blue and Red sides. The finished compositions are scored by the LR win probability model.

```bash
python src/adversarial_evaluation_winner.py
```

### 4. Generate figures

```bash
python src/plot_win_predictor_results.py --outdir results/
python src/plot_draft_predictor_results.py --outdir results/
```

---

## Data

Training data is sourced from [Oracle's Elixir](https://oracleselixir.com/). The dataset is restricted to 2024 matches to avoid Fearless Draft, which was adopted by all major leagues in 2025 and is incompatible with standard draft modelling.

---
