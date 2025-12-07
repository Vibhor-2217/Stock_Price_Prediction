# grpo/eval_grpo_selector.py

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.loaders.dataset_builder import DatasetBuilder
from training.data_split import split_data
from models.two_head_lstm import TwoHeadLSTM
from grpo.feature_policy import FeaturePolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEADBAND = 0.0007  # same as training


def eval_with_optional_policy(model, X, y_ret, device, policy=None, deterministic=True):
    """
    Evaluate model on (X, y_ret).
    If policy is not None, apply a feature mask from the policy before each forward pass.
    Returns: (mae, rmse, corr, acc, frac_features_on)
    """
    model.eval()
    if policy is not None:
        policy.eval()

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_ret, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=64)

    all_true_ret = []
    all_pred_ret = []
    all_true_dir = []
    all_pred_dir = []
    all_valid_mask = []
    all_frac_on = []  # to track average number of features used

    with torch.no_grad():
        for batch_x, batch_ret in loader:
            batch_x = batch_x.to(device)          # [B, T, F]
            batch_ret = batch_ret.to(device).squeeze()  # [B]

            if policy is not None:
                logits = policy(batch_x)          # [B, F]
                probs = torch.sigmoid(logits)

                if deterministic:
                    mask = (probs > 0.3).float()  # [B, F]
                else:
                    bern = torch.distributions.Bernoulli(probs)
                    mask = bern.sample()

                mask_exp = mask.unsqueeze(1)      # [B, 1, F]
                batch_x = batch_x * mask_exp      # apply mask
                all_frac_on.append(mask.mean(dim=-1).cpu().numpy())  # per-sample frac_on

            # forward through LSTM
            dir_logit, ret_pred = model(batch_x)

            dir_target = (batch_ret > 0).float()
            valid_mask = (batch_ret.abs() > DEADBAND).float()

            probs_dir = torch.sigmoid(dir_logit)
            preds_dir = (probs_dir > 0.5).float()

            all_true_ret.append(batch_ret.cpu().numpy())
            all_pred_ret.append(ret_pred.cpu().numpy())
            all_true_dir.append(dir_target.cpu().numpy())
            all_pred_dir.append(preds_dir.cpu().numpy())
            all_valid_mask.append(valid_mask.cpu().numpy())

    true_ret = np.concatenate(all_true_ret)
    pred_ret = np.concatenate(all_pred_ret)
    true_dir = np.concatenate(all_true_dir)
    pred_dir = np.concatenate(all_pred_dir)
    valid_mask = np.concatenate(all_valid_mask)

    # --- regression metrics over all test points ---
    diff = pred_ret - true_ret
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    if np.std(pred_ret) < 1e-9 or np.std(true_ret) < 1e-9:
        corr = 0.0
    else:
        corr = np.corrcoef(true_ret, pred_ret)[0, 1]

    # --- classification metrics on valid days only ---
    valid_idx = valid_mask > 0.5
    valid_true_dir = true_dir[valid_idx]
    valid_pred_dir = pred_dir[valid_idx]

    if valid_true_dir.size > 0:
        acc = (valid_true_dir == valid_pred_dir).mean()
    else:
        acc = float("nan")

    # frac of features ON (only for masked eval)
    if policy is not None and len(all_frac_on) > 0:
        frac_on = np.concatenate(all_frac_on).mean()
    else:
        frac_on = 1.0

    return mae, rmse, corr, acc, frac_on


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # ----- load data & sequences -----
    csv_path = DATA_DIR / "AAPL_regime.csv"
    print(f"[INFO] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    builder = DatasetBuilder(lookback=60)
    X, y_ret, y_dir = builder.create_sequences(df)

    # get TEST split only
    (_, _, _), (_, _, _), (X_test, y_test_ret, y_test_dir) = split_data(X, y_ret, y_dir)

    num_features = X.shape[-1]
    print(f"[INFO] Test set size: {len(X_test)}, features: {num_features}")

    # ----- load frozen LSTM -----
    lstm = TwoHeadLSTM(input_dim=num_features).to(device)
    ckpt_path = MODELS_DIR / "twohead_best.pt"
    print(f"[INFO] Loading LSTM checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    lstm.load_state_dict(state)

    # ----- load feature policy -----
    policy = FeaturePolicy(input_dim=num_features, hidden_dim=64).to(device)
    policy_ckpt = MODELS_DIR / "feature_policy.pt"
    print(f"[INFO] Loading feature policy from {policy_ckpt}")
    policy.load_state_dict(torch.load(policy_ckpt, map_location=device))

    # ----- EVALUATE: baseline -----
    print("\n[BASELINE] All features ON")
    base_mae, base_rmse, base_corr, base_acc, _ = eval_with_optional_policy(
        lstm, X_test, y_test_ret, device, policy=None
    )
    print(f"MAE   : {base_mae:.6f}")
    print(f"RMSE  : {base_rmse:.6f}")
    print(f"Corr  : {base_corr:.4f}")
    print(f"Acc(valid days): {base_acc:.3f}")

    # ----- EVALUATE: GRPO-masked -----
    print("\n[GRPO] Features masked by policy (probs > 0.5)")
    grpo_mae, grpo_rmse, grpo_corr, grpo_acc, frac_on = eval_with_optional_policy(
        lstm, X_test, y_test_ret, device, policy=policy, deterministic=True
    )
    print(f"MAE   : {grpo_mae:.6f}")
    print(f"RMSE  : {grpo_rmse:.6f}")
    print(f"Corr  : {grpo_corr:.4f}")
    print(f"Acc(valid days): {grpo_acc:.3f}")
    print(f"Avg fraction of features ON: {frac_on:.3f}")


if __name__ == "__main__":
    main()
