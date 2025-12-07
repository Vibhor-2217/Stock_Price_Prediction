from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from data.loaders.dataset_builder import DatasetBuilder
from training.data_split import split_data
from models.two_head_lstm import TwoHeadLSTM
from grpo.feature_policy import FeaturePolicy

# ---------- paths & seed ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- per-sample loss helper ----------

def compute_twohead_loss_per_sample(
    model: TwoHeadLSTM,
    x: torch.Tensor,
    y_return: torch.Tensor,
    deadband: float = 0.0007,
    lambda_reg: float = 2.0,
    lambda_cls: float = 0.7,
):
    """
    Returns per-sample total loss and components.

    total_loss_i = lambda_reg * reg_loss_i + lambda_cls * cls_loss_i
    where reg/cls losses are zeroed out for |return| <= deadband.
    """
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")        # [B]
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")   # [B]

    dir_logit, ret_pred = model(x)                         # [B], [B]

    y_return = y_return.squeeze()                          # [B]
    dir_target = (y_return > 0).float()                    # [B]
    valid_mask = (y_return.abs() > deadband).float()       # [B]

    reg_losses = reg_loss_fn(ret_pred, y_return)           # [B]
    cls_losses = cls_loss_fn(dir_logit, dir_target)        # [B]

    # zero out invalid days
    reg_losses = reg_losses * valid_mask
    cls_losses = cls_losses * valid_mask

    total_losses = lambda_reg * reg_losses + lambda_cls * cls_losses  # [B]

    return total_losses, reg_losses, cls_losses, valid_mask


# ---------- GRPO training ----------

def train_grpo_selector(
    csv_name: str = "AAPL_regime.csv",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_sparse: float = 0.03,   # weaker sparsity than before
    entropy_coef: float = 0.03,    # stronger entropy (more exploration)
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # --- load data ---
    csv_path = DATA_DIR / csv_name
    print(f"[INFO] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    builder = DatasetBuilder(lookback=60)
    X, y_ret, y_dir = builder.create_sequences(df)

    (X_train, yr_train, _), (_, _, _), (_, _, _) = split_data(X, y_ret, y_dir)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    yr_train_t = torch.tensor(yr_train, dtype=torch.float32)
    train_ds = TensorDataset(X_train_t, yr_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    num_features = X.shape[-1]
    print(f"[INFO] Training GRPO selector on {len(train_ds)} samples, F={num_features}")

    # --- load frozen LSTM baseline ---
    lstm = TwoHeadLSTM(input_dim=num_features).to(device)
    ckpt_path = MODELS_DIR / "twohead_best.pt"
    print(f"[INFO] Loading baseline LSTM from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    lstm.load_state_dict(state)
    lstm.eval()
    for p in lstm.parameters():
        p.requires_grad = False

    # --- policy network (trainable) ---
    policy = FeaturePolicy(input_dim=num_features, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        policy.train()

        epoch_policy_loss = 0.0
        epoch_reward = 0.0
        epoch_sparsity = 0.0
        num_batches = 0

        for batch_x, batch_ret in train_loader:
            batch_x = batch_x.to(device)          # [B, T, F]
            batch_ret = batch_ret.to(device)      # [B]

            B, T, F = batch_x.shape

            # ---- baseline per-sample loss (no mask) ----
            with torch.no_grad():
                base_total, _, _, valid_mask = compute_twohead_loss_per_sample(
                    lstm, batch_x, batch_ret
                )  # [B]

            # ---- sample mask from policy ----
            logits = policy(batch_x)                      # [B, F]
            probs = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
            bern = torch.distributions.Bernoulli(probs)
            mask = bern.sample()                          # [B, F]
            log_prob = bern.log_prob(mask).sum(dim=-1)    # [B] (sum over features)

            # apply mask across time dimension
            mask_exp = mask.unsqueeze(1)                  # [B, 1, F]
            x_masked = batch_x * mask_exp                 # [B, T, F]

            # ---- per-sample loss with masked features ----
            with torch.no_grad():
                masked_total, _, _, _ = compute_twohead_loss_per_sample(
                    lstm, x_masked, batch_ret
                )  # [B]

            # ---- reward per sample ----
            # improvement: positive if masked loss < baseline loss
            improvement = base_total - masked_total       # [B]

            # sparsity: fraction of features ON for each sample
            frac_on = mask.mean(dim=-1)                   # [B]

            reward = improvement - lambda_sparse * frac_on   # [B]

            # zero reward on invalid days (no training signal where we had no label)
            reward = reward * valid_mask

            # GRPO: group baseline = mean reward over batch
            baseline = reward.mean().detach()
            advantage = reward - baseline                 # [B]

            # ---- policy loss (REINFORCE + entropy) ----
            policy_loss = - (advantage.detach() * log_prob).mean()

            entropy = - (
                probs * probs.log() +
                (1 - probs) * (1 - probs).log()
            ).sum(dim=-1)                                # [B]
            policy_loss -= entropy_coef * entropy.mean()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_reward += reward.mean().item()
            epoch_sparsity += frac_on.mean().item()
            num_batches += 1

        avg_ploss = epoch_policy_loss / max(num_batches, 1)
        avg_reward = epoch_reward / max(num_batches, 1)
        avg_sparsity = epoch_sparsity / max(num_batches, 1)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"PolicyLoss: {avg_ploss:.4f} | "
            f"AvgReward: {avg_reward:.4f} | "
            f"AvgFracFeaturesOn: {avg_sparsity:.3f}"
        )

    # save policy
    out_path = MODELS_DIR / "feature_policy.pt"
    torch.save(policy.state_dict(), out_path)
    print(f"[DONE] Saved feature policy to {out_path}")


if __name__ == "__main__":
    train_grpo_selector()
