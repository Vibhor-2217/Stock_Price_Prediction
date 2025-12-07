import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.two_head_lstm import TwoHeadLSTM

# ---------------- ----------
# Config
# ---------------- ----------

WINDOW = 60
HORIZON = 1
RET_THRESHOLD = 0.003  # threshold for "valid" days

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "AAPL_regime.csv"
LSTM_CKPT = ROOT / "models" / "twohead_best.pt"
POLICY_DIR = ROOT / "models"


# ---------------- ----------
# Helpers (same as train script)
# ---------------- ----------

def build_windows_with_regime(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    assert "direction" in df.columns
    assert "return" in df.columns
    assert "regime" in df.columns

    feature_cols = [c for c in df.columns if c not in ("date", "direction", "return", "regime")]

    n = len(df)
    m = n - WINDOW - HORIZON + 1

    X = np.zeros((m, WINDOW, len(feature_cols)), dtype=np.float32)
    y_ret = np.zeros(m, dtype=np.float32)
    y_dir = np.zeros(m, dtype=np.float32)
    regimes = np.zeros(m, dtype=np.int64)

    feat_vals = df[feature_cols].values
    ret_vals = df["return"].values.astype(np.float32)
    dir_vals = df["direction"].values.astype(np.float32)
    reg_vals = df["regime"].values.astype(np.int64)

    for i in range(m):
        j_end = i + WINDOW
        target_idx = j_end + HORIZON - 1
        X[i] = feat_vals[i:j_end]
        y_ret[i] = ret_vals[target_idx]
        y_dir[i] = dir_vals[target_idx]
        regimes[i] = reg_vals[target_idx]

    return X, y_ret, y_dir, regimes, feature_cols


def time_split_indices(m: int):
    idx = np.arange(m)
    n_train = int(m * 0.7)
    n_val = int(m * 0.85)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_val]
    idx_test = idx[n_val:]
    return idx_train, idx_val, idx_test


def compute_metrics(ret_true, ret_pred, dir_true, dir_logit_pred):
    ret_true = np.asarray(ret_true)
    ret_pred = np.asarray(ret_pred)
    dir_true = np.asarray(dir_true)
    dir_logit_pred = np.asarray(dir_logit_pred)

    mae = float(np.mean(np.abs(ret_pred - ret_true)))
    rmse = float(np.sqrt(np.mean((ret_pred - ret_true) ** 2)))

    if np.std(ret_true) < 1e-8 or np.std(ret_pred) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(ret_pred, ret_true)[0, 1])

    # "valid" days
    valid_mask = np.abs(ret_true) > RET_THRESHOLD
    if valid_mask.sum() == 0:
        acc = 0.0
        mean_pnl = 0.0
        sharpe = 0.0
    else:
        dir_pred = (dir_logit_pred > 0).astype(int)
        dir_true_valid = dir_true[valid_mask]
        dir_pred_valid = dir_pred[valid_mask]
        acc = float((dir_true_valid == dir_pred_valid).mean())

        pos = np.where(dir_pred_valid > 0, 1.0, -1.0)
        pnl = pos * ret_true[valid_mask]
        mean_pnl = float(pnl.mean())
        sharpe = float(mean_pnl / (pnl.std() + 1e-8))

    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "acc_valid": acc,
        "mean_pnl": mean_pnl,
        "sharpe": sharpe,
        "valid_days": int(valid_mask.sum()),
        "total_days": int(len(ret_true)),
    }


def make_block_mask_det(z: torch.Tensor, block_indices: List[List[int]], T: int, F: int) -> torch.Tensor:
    """
    Deterministic mask at eval time.
    z: [num_blocks] in {0,1}
    returns mask [B, T, F]
    """
    num_blocks = z.shape[0]
    device = z.device
    # We'll broadcast later; here we just build [1,T,F]
    mask = torch.ones(1, T, F, device=device)
    for b_idx, cols in enumerate(block_indices):
        if not cols:
            continue
        cols_t = torch.tensor(cols, device=device)
        gate = float(z[b_idx].item())
        mask[:, :, cols_t] = gate
    return mask  # [1,T,F]


# ---------------- ----------
# Main
# ---------------- ----------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading data from {DATA_PATH}")
    X, y_ret, y_dir, regimes, feature_cols = build_windows_with_regime(DATA_PATH)
    idx_train, idx_val, idx_test = time_split_indices(len(X))

    X_test = X[idx_test]
    yret_test = y_ret[idx_test]
    ydir_test = y_dir[idx_test]
    reg_test = regimes[idx_test]

    input_dim = X.shape[-1]
    lstm = TwoHeadLSTM(input_dim=input_dim).to(device)
    print(f"[INFO] Loading baseline LSTM from {LSTM_CKPT}")
    state = torch.load(LSTM_CKPT, map_location=device)
    lstm.load_state_dict(state)
    lstm.to(device)
    lstm.eval()

    # ---------- Baseline (all features ON) ----------
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        dir_full, ret_full = lstm(X_test_t)

    base_metrics = compute_metrics(
        yret_test, ret_full.cpu().numpy(), ydir_test, dir_full.cpu().numpy()
    )
    print("\n[BASELINE] All features ON")
    print(f"MAE   : {base_metrics['mae']:.6f}")
    print(f"RMSE  : {base_metrics['rmse']:.6f}")
    print(f"Corr  : {base_metrics['corr']:.4f}")
    print(f"Acc(valid days): {base_metrics['acc_valid']:.3f}")
    print(f"Mean PnL per day: {base_metrics['mean_pnl']:.6f}")
    print(f"Sharpe (unscaled): {base_metrics['sharpe']:.3f}")

    # ---------- Load per-regime policies ----------
    # We assume block_indices and block_names are the same for all policies.
    policies: Dict[int, dict] = {}
    for p in POLICY_DIR.glob("block_policy_regime*.pt"):
        state = torch.load(p, map_location="cpu")
        reg_id = int(state.get("regime_id"))
        policies[reg_id] = state

    if not policies:
        print("[WARN] No per-regime policies found in models/block_policy_regime*.pt")
        return

    any_state = next(iter(policies.values()))
    block_names: List[str] = any_state["block_names"]
    block_indices: List[List[int]] = any_state["block_indices"]
    num_blocks = len(block_names)

    print("\n[GRPO BLOCK] Gated feature blocks (per-regime policies)")
    # Evaluate per regime, then aggregate
    ret_pred_all = np.zeros_like(yret_test)
    dir_logit_all = np.zeros_like(ydir_test, dtype=np.float32)

    avg_block_probs = np.zeros(num_blocks, dtype=np.float64)
    regime_gate_stats = {}

    with torch.no_grad():
        for reg_id, state in policies.items():
            mask_r = reg_test == reg_id
            n_r = int(mask_r.sum())
            if n_r == 0:
                continue

            logits = torch.tensor(state["logits"], dtype=torch.float32, device=device)
            probs = torch.sigmoid(logits)  # [num_blocks]
            z_det = (probs > 0.5).float()  # deterministic gating
            mask_single = make_block_mask_det(z_det, block_indices, T=X_test.shape[1], F=input_dim)
            mask_r_full = mask_single.expand(n_r, -1, -1)  # [N_r,T,F]

            X_r = torch.tensor(X_test[mask_r], dtype=torch.float32, device=device)
            X_r_gated = X_r * mask_r_full

            dir_g, ret_g = lstm(X_r_gated)

            ret_pred_all[mask_r] = ret_g.cpu().numpy()
            dir_logit_all[mask_r] = dir_g.cpu().numpy()

            avg_block_probs += probs.cpu().numpy() * (n_r / len(X_test))
            regime_gate_stats[reg_id] = probs.cpu().numpy()

    grpo_metrics = compute_metrics(
        yret_test, ret_pred_all, ydir_test, dir_logit_all
    )
    print(f"MAE   : {grpo_metrics['mae']:.6f}")
    print(f"RMSE  : {grpo_metrics['rmse']:.6f}")
    print(f"Corr  : {grpo_metrics['corr']:.4f}")
    print(f"Acc(valid days): {grpo_metrics['acc_valid']:.3f}")
    print(f"Mean PnL per day: {grpo_metrics['mean_pnl']:.6f}")
    print(f"Sharpe (unscaled): {grpo_metrics['sharpe']:.3f}")

    # ---------- Block usage stats ----------
    print("\n[BLOCK USAGE] Avg gate probability per block (test set):")
    for name, p in zip(block_names, avg_block_probs):
        print(f"  {name:8s}: {p:0.3f}")

    print("\n[BLOCK USAGE BY REGIME] Avg gate prob per block:")
    print("Regime r (N) :", "  ".join(f"{name:>8s}" for name in block_names))
    for reg_id in sorted(regime_gate_stats.keys()):
        p = regime_gate_stats[reg_id]
        n_r = int((reg_test == reg_id).sum())
        row = "  ".join(f"{v:0.3f}" for v in p)
        print(f"Regime {reg_id} (N={n_r}): {row}")


if __name__ == "__main__":
    main()
