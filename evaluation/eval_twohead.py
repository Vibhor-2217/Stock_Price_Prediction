# evaluation/eval_twohead.py

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

from data.loaders.dataset_builder import DatasetBuilder
from training.data_split import split_data
from training.dataset import PriceDataset
from models.two_head_lstm import TwoHeadLSTM


DEADBAND = 0.0007  # 7 bps, same as train_twohead

# -------------------------------------------------
# Config: choose which variant to evaluate
# -------------------------------------------------
USE_REGIME = True  # <-- set True for AAPL_regime, False for baseline AAPL_processed

# -------------------------------------------------
# Paths relative to repo root
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    csv_name = "AAPL_regime.csv" if USE_REGIME else "AAPL_processed.csv"
    ckpt_name = "twohead_best.pt"  # both trainers save to this name for now

    csv_path = DATA_DIR / csv_name
    ckpt_path = MODELS_DIR / ckpt_name

    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    print("[INFO] Building sequences...")
    builder = DatasetBuilder(lookback=60)
    X, y_return, y_dir = builder.create_sequences(df)

    # Split data in same way as training
    (_, _, _), (_, _, _), (X_test, y_test_ret, y_test_dir) = \
        split_data(X, y_return, y_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[-1]

    print(f"[INFO] Creating model with input_dim={input_dim}")
    model = TwoHeadLSTM(input_dim=input_dim).to(device)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)  # warning is safe here
    model.load_state_dict(state)
    model.eval()

    test_dataset = PriceDataset(X_test, y_test_ret, y_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=64)

    all_true_ret = []
    all_pred_ret = []
    all_true_dir = []
    all_pred_dir = []
    all_mask = []

    with torch.no_grad():
        for batch_x, batch_ret, _batch_dir in test_loader:
            batch_x = batch_x.to(device)
            batch_ret = batch_ret.to(device).squeeze()

            dir_logit, ret_pred = model(batch_x)

            dir_target = (batch_ret > 0).float()
            mask = (batch_ret.abs() > DEADBAND).float()

            probs = torch.sigmoid(dir_logit)
            preds = (probs > 0.5).float()

            all_true_ret.append(batch_ret.cpu().numpy())
            all_pred_ret.append(ret_pred.cpu().numpy())
            all_true_dir.append(dir_target.cpu().numpy())
            all_pred_dir.append(preds.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

    true_ret = np.concatenate(all_true_ret)
    pred_ret = np.concatenate(all_pred_ret)
    true_dir = np.concatenate(all_true_dir)
    pred_dir = np.concatenate(all_pred_dir)
    mask = np.concatenate(all_mask)

    # --- Regression metrics ---
    diff = pred_ret - true_ret
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    corr = np.corrcoef(true_ret, pred_ret)[0, 1]

    # --- Classification metrics (valid days only) ---
    valid_idx = mask > 0.5
    valid_true = true_dir[valid_idx]
    valid_pred = pred_dir[valid_idx]

    if valid_true.size > 0:
        acc = (valid_true == valid_pred).mean()
    else:
        acc = float("nan")

    print("\n[TEST RESULTS]")
    print(f"Samples: {len(true_ret)}, valid days: {valid_true.size}")
    print(f"Return MAE : {mae:.6f}")
    print(f"Return RMSE: {rmse:.6f}")
    print(f"Return Corr: {corr:.4f}")
    print(f"Direction Acc (valid days): {acc:.3f}")


if __name__ == "__main__":
    main()
