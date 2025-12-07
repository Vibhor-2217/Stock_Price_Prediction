import torch

# -----------------------------------------------------------
# Correct imports based on your directory structure
# -----------------------------------------------------------
from models.two_head_lstm_v2 import TwoHeadLSTM_V2
from models.state_encoder_v1 import StateEncoderV1
from models.auxiliary_heads_v1 import AuxiliaryHeadsV1
from grpo.grpo_trainer_v1 import train_one_batch, build_model


def main():

    # -----------------------------
    # ARCHITECTURE CONFIG
    # -----------------------------
    input_dim = 10          # Replace with your feature count
    hidden_dim = 32         # LSTM hidden size per head
    state_dim = 64          # Embedding size fed to GRPO

    print("Building model components...")

    # Build model (TwoHeadLSTM + StateEncoder + GRPO + AuxHeads)
    model, encoder, grpo, aux, optimizer = build_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        state_dim=state_dim
    )

    print("Model successfully built!")
    print("-----------------------------------")

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    steps = 200

    for step in range(steps):

        # -------------------------------------------------
        # DUMMY DATA FOR TESTING — replace with YOUR DATA
        # -------------------------------------------------
        batch_size = 16
        seq_len = 50

        batch_x = torch.randn(seq_len, batch_size, input_dim)
        next_return = torch.randn(batch_size, 1)
        regime_labels = torch.randint(0, 3, (batch_size,))

        # Train one batch
        loss = train_one_batch(
            model=model,
            encoder=encoder,
            grpo=grpo,
            aux=aux,
            optimizer=optimizer,
            batch_x=batch_x,
            true_next_ret=next_return,
            true_regime=regime_labels
        )

        if step % 20 == 0:
            print(f"Step {step}/{steps} | Loss = {loss:.4f}")

    print("-----------------------------------")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
