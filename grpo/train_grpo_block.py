# grpo/train_grpo_block.py

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

# ----------------- config -----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3

# sparsity: penalise fraction of blocks ON
LAMBDA_SPARSITY = 1e-3  # try 2e-4, 5e-4, 1e-3

# entropy: penalise high entropy (push probs away from 0.5)
LAMBDA_ENTROPY = 1e-3  # try 1e-3, 5e-3

SEED = 42
DEADBAND = 0.0007  # same as eval for "valid" days


# ----------------- utils -----------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_feature_block_mask_sample(
        z: torch.Tensor,
        block_indices: dict,
        num_features: int,
) -> torch.Tensor:
    """
    Build feature mask from sampled block gates z (0/1).

    z: [B, num_blocks]  with 0/1 per block
    returns mask: [B, 1, F]
    """
    B, num_blocks = z.shape
    device = z.device
    mask = torch.zeros(B, num_features, device = device)

    block_names = list(block_indices.keys())
    for b_idx, b_name in enumerate(block_names):
        feat_idxs = block_indices[b_name]
        if not feat_idxs:
            continue
        mask[:, feat_idxs] = z[:, b_idx].unsqueeze(-1)

    return mask.unsqueeze(1)  # [B, 1, F]


# ----------------- main training -----------------

def train_grpo_block():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # ---- load data ----
    csv_path = DATA_DIR / "AAPL_regime.csv"
    print(f"[INFO] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns = ["date"])

    # build sequences (X: [N,T,F], y_ret: [N], y_dir: [N])
    builder = DatasetBuilder(lookback = 60)
    X, y_ret, y_dir = builder.create_sequences(df)

    # feature columns used by LSTM (same convention as eval_twohead.py)
    feature_cols = [
        c for c in df.columns
        if c not in ("return", "direction", "regime")
    ]
    num_features = X.shape[-1]

    # ---- split ----
    (X_train, yr_train, yd_train), \
        (X_val, yr_val, yd_val), \
        (X_test, yr_test, yd_test) = split_data(X, y_ret, y_dir)

    # torch tensors
    X_train_t = torch.tensor(X_train, dtype = torch.float32)
    yr_train_t = torch.tensor(yr_train, dtype = torch.float32)
    X_val_t = torch.tensor(X_val, dtype = torch.float32)
    yr_val_t = torch.tensor(yr_val, dtype = torch.float32)

    train_ds = TensorDataset(X_train_t, yr_train_t)
    val_ds = TensorDataset(X_val_t, yr_val_t)

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = False)

    print(f"[INFO] GRPO-block on {len(train_ds)} train samples, F={num_features}")

    # ---- feature blocks ----
    block_indices = build_block_index_map(feature_cols)
    block_names = list(block_indices.keys())
    num_blocks = len(block_names)
    print(f"[INFO] Feature blocks: {block_names}")
    print(f"[INFO] Num blocks: {num_blocks}")

    # ---- load frozen LSTM ----
    lstm = TwoHeadLSTM(input_dim = num_features).to(device)
    ckpt_path = MODELS_DIR / "twohead_best.pt"
    print(f"[INFO] Loading LSTM from {ckpt_path}")
    state = torch.load(ckpt_path, map_location = device)
    lstm.load_state_dict(state)
    lstm.eval()
    for p in lstm.parameters():
        p.requires_grad_(False)

    # ---- policy network ----
    policy = BlockPolicy(input_dim = num_features * 2,
                         num_blocks = num_blocks,
                         hidden_dim = 64).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr = LR)

    # ----------------- training loop -----------------
    for epoch in range(1, EPOCHS + 1):
        policy.train()
        train_loss_sum = 0.0
        reward_sum = 0.0
        frac_on_sum = 0.0
        entropy_sum = 0.0
        n_batches = 0

        for batch_x, batch_ret in train_loader:
            batch_x = batch_x.to(device)
            batch_ret = batch_ret.to(device).squeeze()

            optimizer.zero_grad()

            # policy outputs logits -> probs
            logits = policy(batch_x)  # [B, num_blocks]
            probs = torch.sigmoid(logits)  # [B, num_blocks]

            # sample block gates
            z = torch.bernoulli(probs)  # [B, num_blocks]

            # build feature mask & apply
            feat_mask = build_feature_block_mask_sample(
                    z, block_indices, num_features = batch_x.shape[-1]
            )  # [B,1,F]
            x_masked = batch_x * feat_mask  # [B,T,F]

            # evaluate LSTM under gating (no grad through LSTM)
            with torch.no_grad():
                dir_logit, ret_pred = lstm(x_masked)
            ret_pred = ret_pred.squeeze()

            # reward: negative squared error (higher is better)
            mse = (ret_pred - batch_ret) ** 2
            reward = -mse  # [B]

            # mask out very small-return days (like eval)
            valid_mask = (batch_ret.abs() > DEADBAND).float()
            reward = reward * valid_mask

            # GRPO-style advantage: reward - mean(reward in batch)
            batch_mean = reward.mean()
            advantage = reward - batch_mean  # [B]

            # logprob of sampled z under Bernoulli(probs)
            logp = z * torch.log(probs + 1e-8) + \
                   (1 - z) * torch.log(1 - probs + 1e-8)  # [B,num_blocks]
            logp_sum = logp.sum(dim = 1)  # [B]

            # policy gradient loss (maximize advantage*logprob)
            pg_loss = -(advantage.detach() * logp_sum).mean()

            # sparsity: penalise fraction of blocks ON
            frac_on = z.mean()

            # entropy: penalise high entropy (push probs towards 0/1)
            entropy = - (probs * torch.log(probs + 1e-8) +
                         (1 - probs) * torch.log(1 - probs + 1e-8))
            entropy = entropy.mean()

            loss = pg_loss + LAMBDA_SPARSITY * frac_on + LAMBDA_ENTROPY * entropy
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            reward_sum += reward.mean().item()
            frac_on_sum += frac_on.item()
            entropy_sum += entropy.item()
            n_batches += 1

        # ---- epoch logs ----
        avg_loss = train_loss_sum / n_batches
        avg_reward = reward_sum / n_batches
        avg_frac_on = frac_on_sum / n_batches
        avg_entropy = entropy_sum / n_batches

        print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"PolicyLoss: {avg_loss:.4f} | "
                f"AvgReward: {avg_reward:.4f} | "
                f"AvgBlocksOn: {avg_frac_on:.3f} | "
                f"Entropy: {avg_entropy:.3f}"
        )

        # (optional) quick validation diagnostics: how many blocks ON on val set
        policy.eval()
        with torch.no_grad():
            all_z_val = []
            for vx, vret in val_loader:
                vx = vx.to(device)
                logits_val = policy(vx)
                probs_val = torch.sigmoid(logits_val)
                z_val = (probs_val > 0.5).float()
                all_z_val.append(z_val.cpu().numpy())
            all_z_val = np.concatenate(all_z_val, axis = 0)
            avg_blocks_on_val = all_z_val.mean()
        print(f"          [VAL] Avg blocks ON (p>0.5): {avg_blocks_on_val:.3f}")

    # ---- save policy ----
    out_path = MODELS_DIR / "block_policy.pt"
    torch.save(policy.state_dict(), out_path)
    print(f"[DONE] Saved block policy to {out_path}")
    print(f"      (λ_sparse={LAMBDA_SPARSITY}, λ_entropy={LAMBDA_ENTROPY})")


if __name__ == "__main__":
    train_grpo_block()
