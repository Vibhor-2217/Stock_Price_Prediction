"""
evaluation/confusion_matrix.py

Run from project root:

    python -m evaluation.confusion_matrix
or
    python evaluation/confusion_matrix.py

Outputs:
  - prints accuracy & confusion matrix
  - saves:
      plots/confusion_matrix.png
      plots/returns_pred_vs_true.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Local imports from your project ---
from data.loaders.dataset_builder import DatasetBuilder
from training.data_split import split_data
from training.dataset import PriceDataset
from models.two_head_lstm import TwoHeadLSTM


# ------------------------ helpers ------------------------ #

def get_project_root() -> Path:
    """Infer project root as the parent of this file's parent."""
    return Path(__file__).resolve().parents[1]


def ensure_plots_dir(root: Path) -> Path:
    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def build_test_loader(df_path: Path, window_size: int = 60, batch_size: int = 64):
    """
    Build DataLoader for the test split using the same
    DatasetBuilder + split_data pipeline as training.
    """
    df = pd.read_csv(df_path)

    builder = DatasetBuilder(lookback=window_size)
    X, y_price, y_dir = builder.create_sequences(df)

    (X_train, yp_train, yd_train), (_, _, _), (X_test, yp_test, yd_test) = split_data(
        X, y_price, y_dir
    )

    test_ds = PriceDataset(X_test, yp_test, yd_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return test_loader, yp_test, yd_test

def load_model(ckpt_path, input_dim: int):
    """
    Load the trained two-head LSTM with the SAME architecture used in training.
    The checkpoint twohead_best.pt was trained with hidden_size=128, num_layers=2.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: match training hyperparameters
    model = TwoHeadLSTM(
        input_dim=input_dim,
        hidden_dim=128,   # <- was 64 in the new default; we override
        num_layers=2,
        dropout=0.1        # or whatever you used in train_twohead.py
    )

    # You can keep this as-is; the FutureWarning is just informational
    state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ------------------------ main eval ------------------------ #

def main():
    root = get_project_root()
    plots_dir = ensure_plots_dir(root)

    # Paths
    df_path = root / "data" / "processed" / "AAPL_regime.csv"
    ckpt_path = root / "models" / "twohead_best.pt"

    if not df_path.exists():
        raise FileNotFoundError(f"Processed data not found at {df_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Build test loader (also gives us raw test targets for plotting)
    test_loader, y_reg_test, y_dir_test = build_test_loader(df_path)

    # Infer feature dimension from one batch
    sample_batch = next(iter(test_loader))[0]  # X, y_reg, y_dir
    _, window_size, input_dim = sample_batch.shape

    model, device = load_model(ckpt_path, input_dim=input_dim)

    all_dir_true = []
    all_dir_pred = []
    all_reg_true = []
    all_reg_pred = []

    with torch.no_grad():
        for X_batch, y_reg_batch, y_dir_batch in test_loader:
            X_batch = X_batch.to(device)

            # Two-head LSTM forward: returns (ret_pred, dir_logit)
            ret_pred, dir_logit = model(X_batch)

            # --- classification outputs ---
            dir_prob = torch.sigmoid(dir_logit).cpu().numpy()
            dir_hat = (dir_prob > 0.5).astype(int)

            all_dir_true.append(y_dir_batch.numpy())
            all_dir_pred.append(dir_hat)

            # --- regression outputs ---
            all_reg_true.append(y_reg_batch.numpy())
            all_reg_pred.append(ret_pred.cpu().numpy())

    y_true = np.concatenate(all_dir_true)
    y_pred = np.concatenate(all_dir_pred).reshape(-1)

    y_reg_true = np.concatenate(all_reg_true)
    y_reg_pred = np.concatenate(all_reg_pred)

    # ------------------- confusion matrix ------------------- #
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = (y_true == y_pred).mean()

    print("[CONFUSION MATRIX] Direction (0 = down, 1 = up)")
    print(cm)
    print(f"\nDirection accuracy (test): {acc:.3f}")

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down", "Up"])
    ax.set_yticklabels(["Down", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Direction Confusion Matrix")

    # Annotate counts
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = plots_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved confusion matrix to: {cm_path}")

    # ------------------- regression graph ------------------- #
    # Simple time-series plot: true vs predicted regression target
    idx = np.arange(len(y_reg_true))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(idx, y_reg_true, label="True", linewidth=1)
    ax2.plot(idx, y_reg_pred, label="Predicted", linewidth=1, alpha=0.8)

    ax2.set_title("Regression target: true vs predicted (test set)")
    ax2.set_xlabel("Test sample index")
    ax2.set_ylabel("Target value")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    ts_path = plots_dir / "returns_pred_vs_true.png"
    fig2.savefig(ts_path, dpi=150)
    plt.close(fig2)

    print(f"Saved regression plot to: {ts_path}")


if __name__ == "__main__":
    main()
