# Regime-Aware LSTM + GRPO for Stock Return Prediction

This repository implements an end-to-end pipeline for **stock return & direction prediction** using:

1. **Feature-rich preprocessing** (25–30 technical indicators + volatility features)
2. **HMM-style regime encoder** to tag each day with a latent market regime
3. A **two-head LSTM baseline**  
   - Head 1: next-day **log return** (regression)  
   - Head 2: next-day **direction** (up / down) with a dead-band
4. A **GRPO (Group Relative Policy Optimization) feature-block policy** that learns, per regime, which feature blocks to keep or drop (dynamic feature selection / gating).

The goal is to end up with a clean baseline, a regime-aware extension, and a GRPO-based feature gating mechanism.

---

## 1. Project Structure

High-level layout (logical, some files may be in subfolders):

```text
PricePrediction_2/
  config/
    __init__.py
    paths.py              # centralised paths and per-symbol helpers

  data/
    raw/                  # raw CSVs per ticker: AAPL.csv, MSFT.csv, ...
    processed/            # processed + regime-annotated CSVs
    loaders/
      __init__.py
      save_processed.py   # builds {SYM}_processed.csv from raw
      yfinance_loader.py  # optional: download from yfinance

  features/
    __init__.py
    technical_indicators.py   # adds 25–30 TA features using `ta`
    feature_normalization.py  # StandardScaler wrapper

  regime_encoder/
    __init__.py
    build_regimes.py          # reads {SYM}_processed → writes {SYM}_regime
    hmm_regime_encoder.py     # HMMRegimeEncoder class

  models/
    __init__.py
    two_head_lstm.py          # shared LSTM + (return, direction) heads
    lstm_baseline.py          # simpler baseline (if used)

  training/
    __init__.py
    dataset.py                # PriceDataset wrapper around sequences
    data_split.py             # train / val / test split helpers
    train_baseline.py         # baseline LSTM training helpers
    train_twohead.py          # (in your repo) two-head LSTM training script

  evaluation/
    __init__.py
    eval_twohead.py           # evaluate baseline / two-head LSTM

  grpo/
    __init__.py
    block_policy.py           # policy over feature-blocks
    feature_policy.py         # (optional) per-feature policy
    train_grpo_block_by_regime.py  # train GRPO block policy per regime
    eval_grpo_block_by_regime.py   # evaluate GRPO policy
    train_grpo_selector.py    # (optional) selector experiments
    eval_grpo_selector.py     # (optional) selector evaluation

  README.md
  requirements.txt
