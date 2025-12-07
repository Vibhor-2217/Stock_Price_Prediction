# training/train_twohead.py

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.two_head_lstm import TwoHeadLSTM
from training.dataset import PriceDataset
from training.data_split import split_data


# ---------- project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- seeding ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- training ----------
def train_twohead(
    X,
    y_return,
    y_dir,          # kept for API compatibility, not directly used
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    deadband: float = 0.0007,   # 7 bps
    lambda_reg: float = 2.0,
    lambda_cls: float = 0.7,
):

    # make run deterministic
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Split data (same as baseline) ----
    (X_train, yr_train, yd_train), (X_val, yr_val, yd_val), _ = split_data(
        X, y_return, y_dir
    )

    train_dataset = PriceDataset(X_train, yr_train, yd_train)
    val_dataset = PriceDataset(X_val, yr_val, yd_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = X.shape[-1]
    model = TwoHeadLSTM(input_dim=input_dim).to(device)

    # Losses
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")        # per-sample Huber
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")   # per-sample BCE

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    ckpt_path = MODELS_DIR / "twohead_best.pt"

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0
        num_train_batches = 0

        for batch_x, batch_ret, _batch_dir in train_loader:
            batch_x = batch_x.to(device)
            batch_ret = batch_ret.to(device).squeeze()  # [B]

            optimizer.zero_grad()

            dir_logit, ret_pred = model(batch_x)  # both [B]

            # Direction target & mask derived from returns
            dir_target = (batch_ret > 0).float()         # 1 if up day
            mask = (batch_ret.abs() > deadband).float()  # ignore tiny moves

            # Regression loss (Huber) with mask
            reg_losses = reg_loss_fn(ret_pred, batch_ret)  # [B]
            reg_loss = (reg_losses * mask).sum() / (mask.sum() + 1e-8)

            # Classification loss with mask
            cls_losses = cls_loss_fn(dir_logit, dir_target)  # [B]
            cls_loss = (cls_losses * mask).sum() / (mask.sum() + 1e-8)

            loss = lambda_reg * reg_loss + lambda_cls * cls_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_loss / max(num_train_batches, 1)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        total_correct = 0
        total_valid = 0

        with torch.no_grad():
            for batch_x, batch_ret, _batch_dir in val_loader:
                batch_x = batch_x.to(device)
                batch_ret = batch_ret.to(device).squeeze()

                dir_logit, ret_pred = model(batch_x)

                dir_target = (batch_ret > 0).float()
                mask = (batch_ret.abs() > deadband).float()

                reg_losses = reg_loss_fn(ret_pred, batch_ret)
                reg_loss = (reg_losses * mask).sum() / (mask.sum() + 1e-8)

                cls_losses = cls_loss_fn(dir_logit, dir_target)
                cls_loss = (cls_losses * mask).sum() / (mask.sum() + 1e-8)

                loss = lambda_reg * reg_loss + lambda_cls * cls_loss
                val_loss += loss.item()
                num_val_batches += 1

                # accuracy on valid subset
                probs = torch.sigmoid(dir_logit)
                preds = (probs > 0.5).float()
                correct = ((preds == dir_target) * mask.bool()).sum().item()
                valid = mask.sum().item()

                total_correct += correct
                total_valid += valid

        avg_val_loss = val_loss / max(num_val_batches, 1)
        val_acc = (total_correct / total_valid) if total_valid > 0 else 0.0

        # checkpoint on best val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc(valid days): {val_acc:.3f}"
        )

    return model
