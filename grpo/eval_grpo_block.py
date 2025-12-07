import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.loaders.dataset_builder import DatasetBuilder
from training.data_split import split_data
from models.two_head_lstm import TwoHeadLSTM
from grpo.block_policy import BlockPolicy
from config.feature_blocks import build_block_index_map

# ----------------- paths & config -----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

BATCH_SIZE = 128
DEADBAND = 0.0007        # same as training / earlier eval
SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------- metrics helpers -----------------

def compute_basic_metrics(y_true, y_pred, dir_true, dir_logit):
    """
    y_true, y_pred: [N] returns
    dir_true:       [N] (0/1)
    dir_logit:      [N] logits for "up"
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = 0.0

    # direction accuracy on "valid" days (when |return| > deadband)
    probs = 1 / (1 + np.exp(-dir_logit))
    preds = (probs >= 0.5).astype(np.float32)
    mask_valid = (np.abs(y_true) > DEADBAND).astype(np.float32)

    acc = ((preds == dir_true.astype(np.float32)) * mask_valid).sum()
    acc = acc / (mask_valid.sum() + 1e-8)

    return mae, rmse, corr, acc


def compute_pnl_and_sharpe(y_true, y_pred):
    """
    Simple daily PnL: sign(pred_return) * actual_return on valid days.
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mask = (np.abs(y_true) > DEADBAND).astype(np.float64)

    signal = np.sign(y_pred)
    pnl = signal * y_true * mask
    valid_days = mask.sum()

    if valid_days < 1:
        return 0.0, 0.0, 0.0

    mean_pnl = pnl.sum() / valid_days
    std_pnl = pnl.std() + 1e-12
    sharpe = mean_pnl / std_pnl
    return mean_pnl, sharpe, valid_days


# ----------------- main eval -----------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO block gating policy on AAPL_regime."
    )
    parser.add_argument(
        "--policy",
        choices=["reg", "cls"],
        default="reg",
        help="Which block policy to use: "
             "'reg' = old reward (returns only), "
             "'cls' = direction-aware reward.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # ---- load data ----
    csv_path = DATA_DIR / "AAPL_regime.csv"
    print(f"[INFO] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # feature columns used as LSTM inputs
    feature_cols = [c for c in df.columns if c not in ("return", "direction", "regime")]
    num_features = len(feature_cols)

    builder = DatasetBuilder(lookback=60)
    X, y_ret, y_dir = builder.create_sequences(df)

    # regime label for the prediction day (same indexing as y_ret)
    regimes_all = df["regime"].values[builder.lookback:]

    # chronological split (same helper as training)
    (X_train, yr_train, yd_train), \
    (X_val, yr_val, yd_val), \
    (X_test, yr_test, yd_test) = split_data(X, y_ret, y_dir)

    # split regimes using the same boundaries
    n_train = len(X_train)
    n_val = len(X_val)
    reg_train = regimes_all[:n_train]
    reg_val = regimes_all[n_train:n_train + n_val]
    reg_test = regimes_all[n_train + n_val:]

    # sanity
    assert len(X_test) == len(reg_test), "Test regime split mismatch"

    # tensors (only test needed for metrics)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    yr_test_t = torch.tensor(yr_test, dtype=torch.float32)
    yd_test_t = torch.tensor(yd_test, dtype=torch.float32)

    test_ds = TensorDataset(X_test_t, yr_test_t, yd_test_t)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Test samples: {len(test_ds)}, regimes coverage: "
          f"{np.unique(reg_test, return_counts=True)}")

    # ---- feature blocks ----
    block_indices = build_block_index_map(feature_cols)
    block_names = list(block_indices.keys())
    num_blocks = len(block_names)
    print(f"[INFO] Feature blocks: {block_names}")

    # ---- load baseline LSTM ----
    lstm = TwoHeadLSTM(input_dim=num_features).to(device)
    ckpt_path = MODELS_DIR / "twohead_best.pt"
    print(f"[INFO] Loading baseline LSTM from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    lstm.load_state_dict(state)
    lstm.eval()
    for p in lstm.parameters():
        p.requires_grad_(False)

    # ---- load chosen GRPO policy ----
    policy_file_map = {
        "reg": "block_policy.pt",
        "cls": "block_policy_cls.pt",
    }
    policy_path = MODELS_DIR / policy_file_map[args.policy]
    print(f"[INFO] Loading GRPO block policy variant='{args.policy}' "
          f"from {policy_path}")

    policy = BlockPolicy(
        input_dim=num_features * 2,
        num_blocks=num_blocks,
        hidden_dim=64,
    ).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    # ----------------- baseline (all features ON) -----------------
    all_true_ret = []
    all_pred_ret_base = []
    all_dir_true = []
    all_dir_logit_base = []

    with torch.no_grad():
        for xb, yb_ret, yb_dir in test_loader:
            xb = xb.to(device)
            yb_ret = yb_ret.to(device).squeeze()
            yb_dir = yb_dir.to(device).squeeze()

            dir_logit, ret_pred = lstm(xb)
            dir_logit = dir_logit.squeeze(-1)
            ret_pred = ret_pred.squeeze(-1)

            all_true_ret.append(yb_ret.cpu().numpy())
            all_pred_ret_base.append(ret_pred.cpu().numpy())
            all_dir_true.append(yb_dir.cpu().numpy())
            all_dir_logit_base.append(dir_logit.cpu().numpy())

    all_true_ret = np.concatenate(all_true_ret)
    all_pred_ret_base = np.concatenate(all_pred_ret_base)
    all_dir_true = np.concatenate(all_dir_true)
    all_dir_logit_base = np.concatenate(all_dir_logit_base)

    mae_b, rmse_b, corr_b, acc_b = compute_basic_metrics(
        all_true_ret, all_pred_ret_base, all_dir_true, all_dir_logit_base
    )
    mean_pnl_b, sharpe_b, valid_days_b = compute_pnl_and_sharpe(
        all_true_ret, all_pred_ret_base
    )

    print("\n[BASELINE] All features ON")
    print(f"MAE   : {mae_b:0.6f}")
    print(f"RMSE  : {rmse_b:0.6f}")
    print(f"Corr  : {corr_b:0.4f}")
    print(f"Acc(valid days): {acc_b:0.3f}")
    print(f"Mean PnL per day: {mean_pnl_b:0.6f}")
    print(f"Sharpe (unscaled): {sharpe_b:0.3f}")

    # ----------------- GRPO-gated blocks -----------------
    all_pred_ret_gate = []
    all_dir_logit_gate = []

    all_gate_probs = []   # [N, num_blocks]
    all_gate_hard = []    # [N, num_blocks]

    with torch.no_grad():
        for xb, yb_ret, yb_dir in test_loader:
            xb = xb.to(device)

            # baseline again for consistency (not strictly needed for metrics)
            dir_logit_b, ret_b = lstm(xb)
            dir_logit_b = dir_logit_b.squeeze(-1)
            ret_b = ret_b.squeeze(-1)

            # policy probabilities for blocks
            logits = policy(xb)                # [B, num_blocks]
            probs = torch.sigmoid(logits)      # [B, num_blocks]
            z = (probs > 0.5).float()          # deterministic hard gating

            # build feature mask from blocks
            B, T, F = xb.shape
            mask = torch.zeros(B, F, device=xb.device)

            for bi, blk_name in enumerate(block_names):
                idxs = block_indices[blk_name]
                if not idxs:
                    continue
                mask[:, idxs] = z[:, bi].unsqueeze(-1)

            mask = mask.unsqueeze(1)           # [B,1,F]
            x_gated = xb * mask

            dir_logit_g, ret_g = lstm(x_gated)
            dir_logit_g = dir_logit_g.squeeze(-1)
            ret_g = ret_g.squeeze(-1)

            all_pred_ret_gate.append(ret_g.cpu().numpy())
            all_dir_logit_gate.append(dir_logit_g.cpu().numpy())
            all_gate_probs.append(probs.cpu().numpy())
            all_gate_hard.append(z.cpu().numpy())

    all_pred_ret_gate = np.concatenate(all_pred_ret_gate)
    all_dir_logit_gate = np.concatenate(all_dir_logit_gate)
    all_gate_probs = np.concatenate(all_gate_probs, axis=0)
    all_gate_hard = np.concatenate(all_gate_hard, axis=0)

    mae_g, rmse_g, corr_g, acc_g = compute_basic_metrics(
        all_true_ret, all_pred_ret_gate, all_dir_true, all_dir_logit_gate
    )
    mean_pnl_g, sharpe_g, valid_days_g = compute_pnl_and_sharpe(
        all_true_ret, all_pred_ret_gate
    )

    avg_blocks_on = all_gate_hard.mean()

    print(f"\n[GRPO BLOCK] Gated feature blocks (policy='{args.policy}')")
    print(f"MAE   : {mae_g:0.6f}")
    print(f"RMSE  : {rmse_g:0.6f}")
    print(f"Corr  : {corr_g:0.4f}")
    print(f"Acc(valid days): {acc_g:0.3f}")
    print(f"Mean PnL per day: {mean_pnl_g:0.6f}")
    print(f"Sharpe (unscaled): {sharpe_g:0.3f}")
    print(f"Avg blocks ON: {avg_blocks_on:0.3f}")

    # ----------------- block usage stats -----------------

    print("\n[BLOCK USAGE] Avg gate probability per block (test set):")
    for i, name in enumerate(block_names):
        print(f"  {name:9s}: {all_gate_probs[:, i].mean():0.3f}")

    # usage by regime
    reg_test_arr = np.asarray(reg_test)
    regimes_unique = np.unique(reg_test_arr)

    print("\n[BLOCK USAGE BY REGIME] Avg gate prob per block:")
    header = "Regime r (N) : " + "  ".join(f"{n:9s}" for n in block_names)
    print(header)
    for r in regimes_unique:
        mask_r = reg_test_arr == r
        cnt_r = mask_r.sum()
        if cnt_r == 0:
            continue
        probs_r = all_gate_probs[mask_r]
        means_r = probs_r.mean(axis=0)
        cols = "  ".join(f"{m:0.3f}" for m in means_r)
        print(f"Regime {int(r)} (N={cnt_r}): {cols}")


if __name__ == "__main__":
    main()
