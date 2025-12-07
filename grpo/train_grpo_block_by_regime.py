import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.two_head_lstm import TwoHeadLSTM

# ---------------- ----------
# Config
# ---------------- ----------

WINDOW = 60
HORIZON = 1

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # we don't actually use val here, but we keep splits consistent

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

LAMBDA_SPARSITY = 1e-4      # penalty for blocks ON
LAMBDA_ENTROPY = 5e-3       # entropy bonus
ALPHA_CLS = 0.2             # weight for direction accuracy in reward

RET_THRESHOLD = 0.003       # for "valid day" accuracy / PnL (kept here for possible logging)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "AAPL_regime.csv"
LSTM_CKPT = ROOT / "models" / "twohead_best.pt"
POLICY_DIR = ROOT / "models"


# ---------------- ----------
# Helpers
# ---------------- ----------

def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_windows_with_regime(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build sliding-window tensors X, y_ret, y_dir, regime_id from AAPL_regime.csv
    """
    df = pd.read_csv(csv_path)

    # sort by date if present
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # assume these exist
    assert "direction" in df.columns
    assert "return" in df.columns
    assert "regime" in df.columns

    feature_cols = [c for c in df.columns if c not in ("date", "direction", "return", "regime")]

    n = len(df)
    m = n - WINDOW - HORIZON + 1
    assert m > 0, "Not enough rows for chosen WINDOW/HORIZON"

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
    n_train = int(m * TRAIN_RATIO)
    n_val = int(m * (TRAIN_RATIO + VAL_RATIO))
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_val]
    idx_test = idx[n_val:]
    return idx_train, idx_val, idx_test


def build_feature_blocks(feature_cols: List[str]) -> Tuple[List[str], List[List[int]]]:
    """
    Map your feature columns into 5 blocks:
    - price      (e.g., close/open/high/low)
    - volume     (e.g., volume)
    - volatility (e.g., realized vol, ATR, VIX, etc.)
    - technical  (MA, MACD, RSI, etc.)
    - regime     (one-hot regime columns)
    Adjust the lists below to match AAPL_regime.csv EXACTLY.
    """
    # ---- EDIT THESE LISTS IF NEEDED ----
    price_cols = ["close", "open", "high", "low"]
    volume_cols = ["volume"]
    regime_one_hot_cols = ["regime_0", "regime_1", "regime_2", "regime_3"]

    # heuristic guesses for volatility / technical: adjust if your names differ
    vol_cols = [c for c in feature_cols if ("vol" in c.lower() or "vix" in c.lower()) and c not in volume_cols]
    tech_cols = [
        c
        for c in feature_cols
        if c not in price_cols + volume_cols + vol_cols + regime_one_hot_cols
    ]

    block_names = ["price", "volume", "volatility", "technical", "regime"]
    blocks: List[List[int]] = []

    for cols in [price_cols, volume_cols, vol_cols, tech_cols, regime_one_hot_cols]:
        idxs = [feature_cols.index(c) for c in cols if c in feature_cols]
        blocks.append(idxs)

    # sanity
    for name, idxs in zip(block_names, blocks):
        if len(idxs) == 0:
            print(f"[WARN] Block '{name}' is empty. Check build_feature_blocks().")

    return block_names, blocks


class BlockPolicy(nn.Module):
    """
    Simple Bernoulli policy over blocks:
      - parameter: logits per block
      - forward() returns probs (sigmoid(logits))
    """

    def __init__(self, num_blocks: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_blocks))

    def forward(self) -> torch.Tensor:
        probs = torch.sigmoid(self.logits)  # [B]
        return probs


def make_block_mask(z: torch.Tensor, block_indices: List[List[int]], T: int, F: int) -> torch.Tensor:
    """
    z: [B, num_blocks] 0/1
    returns mask [B, T, F] in {0,1}
    """
    B, num_blocks = z.shape
    device = z.device
    mask = torch.ones(B, T, F, device=device)
    for b_idx, cols in enumerate(block_indices):
        if not cols:
            continue
        cols_t = torch.tensor(cols, device=device)
        gate = z[:, b_idx].view(B, 1, 1)  # [B,1,1]
        mask[:, :, cols_t] = mask[:, :, cols_t] * gate
    return mask


def train_policy_for_regime(
    reg_id: int,
    X_train: np.ndarray,
    yret_train: np.ndarray,
    ydir_train: np.ndarray,
    reg_train: np.ndarray,
    block_indices: List[List[int]],
    lstm: TwoHeadLSTM,
    device: torch.device,
):
    mask_reg = reg_train == reg_id
    n_reg = int(mask_reg.sum())
    print(f"\n[REGIME {reg_id}] Training on {n_reg} train samples")
    if n_reg < BATCH_SIZE:
        print(f"[REGIME {reg_id}] Too few samples (< batch size), skipping.")
        return None

    X_r = torch.tensor(X_train[mask_reg], dtype=torch.float32)
    yr_r = torch.tensor(yret_train[mask_reg], dtype=torch.float32)
    yd_r = torch.tensor(ydir_train[mask_reg], dtype=torch.float32)

    dataset = TensorDataset(X_r, yr_r, yd_r)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    policy = BlockPolicy(num_blocks=len(block_indices)).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    lstm.eval()
    for p in lstm.parameters():
        p.requires_grad_(False)

    for epoch in range(1, EPOCHS + 1):
        policy.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_frac_on = 0.0
        n_batches = 0

        for xb, yretb, ydirb in loader:
            xb = xb.to(device)
            yretb = yretb.to(device)
            ydirb = ydirb.to(device)

            B, T, F_dim = xb.shape

            with torch.no_grad():
                dir_full, ret_full = lstm(xb)
                ret_loss_full = F.smooth_l1_loss(ret_full, yretb, reduction="none")
                bce_full = F.binary_cross_entropy_with_logits(
                    dir_full, ydirb, reduction="none"
                )
                acc_full = ((dir_full > 0).float() == ydirb).float()

            # sample gates
            probs = policy()  # [num_blocks]
            probs_expanded = probs.unsqueeze(0).expand(B, -1)
            dist = torch.distributions.Bernoulli(probs_expanded)
            z = dist.sample()  # [B, num_blocks]
            logprob = dist.log_prob(z).sum(dim=1)  # [B]

            frac_on = z.mean(dim=1)  # per-sample

            mask = make_block_mask(z, block_indices, T, F_dim)
            x_gated = xb * mask

            with torch.no_grad():
                dir_g, ret_g = lstm(x_gated)
                ret_loss_g = F.smooth_l1_loss(ret_g, yretb, reduction="none")
                bce_g = F.binary_cross_entropy_with_logits(
                    dir_g, ydirb, reduction="none"
                )
                acc_g = ((dir_g > 0).float() == ydirb).float()

            imp_ret = ret_loss_full - ret_loss_g
            imp_cls = acc_g - acc_full  # accuracy improvement

            reward = imp_ret + ALPHA_CLS * imp_cls
            reward = reward - LAMBDA_SPARSITY * frac_on

            # Group-relative baseline (per batch)
            advantage = reward - reward.mean()

            # entropy bonus (same for all samples)
            entropy = -(
                probs * torch.log(probs + 1e-8)
                + (1.0 - probs) * torch.log(1.0 - probs + 1e-8)
            ).mean()

            loss_policy = -(advantage.detach() * logprob).mean() - LAMBDA_ENTROPY * entropy

            optimizer.zero_grad()
            loss_policy.backward()
            optimizer.step()

            epoch_loss += loss_policy.item()
            epoch_reward += reward.mean().item()
            epoch_frac_on += frac_on.mean().item()
            n_batches += 1

        if n_batches == 0:
            print(f"[REGIME {reg_id}] No batches? skipping.")
            return None

        print(
            f"[REGIME {reg_id}] Epoch {epoch:02d}/{EPOCHS} "
            f"| Loss: {epoch_loss / n_batches:.4f} "
            f"| AvgReward: {epoch_reward / n_batches:.4f} "
            f"| AvgFracOn: {epoch_frac_on / n_batches:.3f}"
        )

    return policy.cpu()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading data from {DATA_PATH}")
    X, y_ret, y_dir, regimes, feature_cols = build_windows_with_regime(DATA_PATH)

    idx_train, idx_val, idx_test = time_split_indices(len(X))
    X_train, yret_train, ydir_train, reg_train = (
        X[idx_train],
        y_ret[idx_train],
        y_dir[idx_train],
        regimes[idx_train],
    )

    # build blocks from feature names
    block_names, block_indices = build_feature_blocks(feature_cols)
    print(f"[INFO] Feature blocks: {block_names}")
    print("[INFO] Block sizes:", [len(b) for b in block_indices])

    # load baseline LSTM
    input_dim = X.shape[-1]
    lstm = TwoHeadLSTM(input_dim=input_dim).to(device)
    print(f"[INFO] Loading baseline LSTM from {LSTM_CKPT}")
    state = torch.load(LSTM_CKPT, map_location=device)
    lstm.load_state_dict(state)
    lstm.to(device)

    POLICY_DIR.mkdir(parents=True, exist_ok=True)

    unique_regimes = sorted(int(r) for r in np.unique(reg_train))
    for reg_id in unique_regimes:
        policy = train_policy_for_regime(
            reg_id,
            X_train,
            yret_train,
            ydir_train,
            reg_train,
            block_indices,
            lstm,
            device,
        )
        if policy is None:
            continue

        out_path = POLICY_DIR / f"block_policy_regime{reg_id}.pt"
        torch.save(
            {
                "logits": policy.logits.detach().numpy(),
                "block_names": block_names,
                "block_indices": block_indices,
                "regime_id": reg_id,
            },
            out_path,
        )
        print(f"[REGIME {reg_id}] Saved policy to {out_path}")


if __name__ == "__main__":
    main()
