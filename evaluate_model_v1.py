import torch
import pandas as pd
import numpy as np

from models.two_head_lstm_v2 import TwoHeadLSTM_V2
from models.state_encoder_v1 import StateEncoderV1
from models.auxiliary_heads_v1 import AuxiliaryHeadsV1
from grpo.grpo_trainer_v1 import SimpleGRPO   # GRPO block only

# ===============================================================
#                  LOAD AND PREPARE DATA
# ===============================================================

def load_data(seq_len=50):

    df = pd.read_csv("data/processed/AAPL_processed.csv")
    df_regime = pd.read_csv("data/processed/AAPL_regime.csv")

    # Keep only numeric columns
    df = df.select_dtypes(include = [np.number])
    leak_cols = ["return", "log_return", "direction"]
    df = df.drop(columns = [c for c in leak_cols if c in df.columns])

    df_regime = df_regime.select_dtypes(include = [np.number])
    df_regime = df_regime[["regime"]]

    features = df.values
    regimes = df_regime["regime"].values.astype(int)

    X_seq = []
    ret_target = []
    regime_target = []

    for i in range(len(features) - seq_len - 1):
        X_seq.append(features[i : i + seq_len])
        ret_target.append(features[i + seq_len, 0])       # next return
        regime_target.append(regimes[i + seq_len])        # next regime label

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    ret_target = torch.tensor(np.array(ret_target), dtype=torch.float32).unsqueeze(1)
    regime_target = torch.tensor(np.array(regime_target), dtype=torch.long)

    return X_seq, ret_target, regime_target


# ===============================================================
#               LOAD MODELS FROM SAVED WEIGHTS
# ===============================================================

def load_models(input_dim, hidden_dim=64, state_dim=128, num_regimes=4):

    model = TwoHeadLSTM_V2(input_dim, hidden_dim)
    encoder = StateEncoderV1(hidden_dim, state_dim)
    grpo = SimpleGRPO(state_dim)
    aux = AuxiliaryHeadsV1(state_dim, num_regimes=num_regimes)

    # Load the saved state dicts
    model.load_state_dict(torch.load("model_twohead_lstm_v2.pt"))
    encoder.load_state_dict(torch.load("state_encoder_v1.pt"))
    grpo.load_state_dict(torch.load("grpo_block_v1.pt"))
    aux.load_state_dict(torch.load("aux_heads_v1.pt"))

    model.eval()
    encoder.eval()
    grpo.eval()
    aux.eval()

    return model, encoder, grpo, aux


# ===============================================================
#                RUN EVALUATION ON THE WHOLE SET
# ===============================================================

def evaluate():

    seq_len = 50

    print("Loading data...")
    X_seq, next_ret_true, next_regime_true = load_data(seq_len=seq_len)
    print(f"Evaluation samples: {len(X_seq)}")

    input_dim = X_seq.shape[-1]

    print("Loading models...")
    model, encoder, grpo, aux = load_models(input_dim)

    print("Running evaluation...")

    preds_ret = []
    preds_regime = []
    preds_direction = []

    with torch.no_grad():
        for i in range(len(X_seq)):
            x = X_seq[i]                  # shape [seq_len, features]
            x = x.unsqueeze(1)            # → [seq_len, batch=1, features]

            # LSTM
            H1, H2 = model(x)

            # Encoder
            z = encoder(H1, H2)           # [1, state_dim]

            # Auxiliary predictions
            next_ret_pred, regime_logits = aux(z)

            pred_ret = next_ret_pred.item()
            pred_regime = torch.argmax(regime_logits, dim=1).item()

            # Convert return → direction
            pred_direction = 1 if pred_ret > 0 else 0

            preds_ret.append(pred_ret)
            preds_regime.append(pred_regime)
            preds_direction.append(pred_direction)

    # Convert to tensors for metrics
    preds_ret = torch.tensor(preds_ret)
    preds_regime = torch.tensor(preds_regime)
    preds_direction = torch.tensor(preds_direction)

    true_direction = (next_ret_true.squeeze() > 0).long()

    # ==========================================================
    #                     METRICS
    # ==========================================================

    mse_next_return = torch.mean((preds_ret - next_ret_true.squeeze())**2).item()
    regime_accuracy = (preds_regime == next_regime_true).float().mean().item()
    direction_accuracy = (preds_direction == true_direction).float().mean().item()

    print("\n============== EVALUATION RESULTS ==============")
    print(f"Next Return MSE          : {mse_next_return:.6f}")
    print(f"Regime Classification Acc: {regime_accuracy*100:.2f}%")
    print(f"Direction Accuracy       : {direction_accuracy*100:.2f}%")
    print("================================================\n")



# ===============================================================
#                        RUN
# ===============================================================

if __name__ == "__main__":
    evaluate()
