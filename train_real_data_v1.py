import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

# --------------------------
# IMPORT MODEL COMPONENTS
# --------------------------
from models.two_head_lstm_v2 import TwoHeadLSTM_V2
from models.state_encoder_v1 import StateEncoderV1
from models.auxiliary_heads_v1 import AuxiliaryHeadsV1
from grpo.grpo_trainer_v1 import train_one_batch, build_model


# ==========================================================
#               REAL MARKET DATASET LOADER
# ==========================================================

class MarketDataset(Dataset):
    def __init__(self, features_df, regime_df, seq_len=50):

        self.features = features_df.values
        self.regimes = regime_df["regime"].values.astype(int)  # <-- FIX
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len - 1

    def __getitem__(self, idx):

        X = self.features[idx : idx + self.seq_len]

        next_ret = float(self.features[idx + self.seq_len, 0])

        # scalar regime value
        next_regime = int(self.regimes[idx + self.seq_len])

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor([next_ret], dtype=torch.float32),
            torch.tensor(next_regime, dtype=torch.long),
        )

# ==========================================================
#               MAIN TRAINING FUNCTION
# ==========================================================

def train_real_data():

    # -----------------------------
    # Load CSV files
    # -----------------------------
    df = pd.read_csv("data/processed/AAPL_processed.csv")
    df_regime = pd.read_csv("data/processed/AAPL_regime.csv")

    df = df.select_dtypes(include = [np.number])
    leak_cols = ["return", "log_return", "direction"]
    df = df.drop(columns = [c for c in leak_cols if c in df.columns])

    df_regime = df_regime.select_dtypes(include = [np.number])
    df_regime = df_regime[["regime"]]

    print("Loaded dataset:")
    print(df.head())

    # -----------------------------
    # Dataset settings
    # -----------------------------
    seq_len = 50
    batch_size = 32

    dataset = MarketDataset(df, df_regime, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -----------------------------
    # Build the model components
    # -----------------------------
    input_dim = df.shape[1]     # number of columns in processed data
    hidden_dim = 64
    state_dim = 128

    print("Building model...")

    model, encoder, grpo, aux, optimizer = build_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        state_dim=state_dim
    )

    print("Model built successfully!")
    print("----------------------------------")

    # -----------------------------
    # Training loop
    # -----------------------------
    steps = 2000
    step_count = 0

    for epoch in range(3):
        for batch_x, next_ret, regime in loader:

            # Convert to correct shape [seq_len, batch, input_dim]
            batch_x = batch_x.permute(1, 0, 2)

            loss = train_one_batch(
                model=model,
                encoder=encoder,
                grpo=grpo,
                aux=aux,
                optimizer=optimizer,
                batch_x=batch_x,
                true_next_ret=next_ret,
                true_regime=regime
            )

            step_count += 1

            if step_count % 100 == 0:
                print(f"Step {step_count} | Loss = {loss:.4f}")

            if step_count >= steps:
                break

    print("----------------------------------")
    print("Training finished!")
    print("Saving models...")

    torch.save(model.state_dict(), "model_twohead_lstm_v2.pt")
    torch.save(encoder.state_dict(), "state_encoder_v1.pt")
    torch.save(grpo.state_dict(), "grpo_block_v1.pt")
    torch.save(aux.state_dict(), "aux_heads_v1.pt")

    print("All models saved successfully!")


# ==========================================================
#                        RUN
# ==========================================================

if __name__ == "__main__":
    train_real_data()
