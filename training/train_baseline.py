import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.lstm_baseline import LSTMBaseline
from training.dataset import PriceDataset
from training.data_split import split_data
print(">> USING UPDATED train_baseline.py <<")
def train_baseline(X, y_return, y_dir, epochs=15, batch_size=32, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Split data
    (X_train, yr_train, yd_train), (X_val, yr_val, yd_val), _ = split_data(X, y_return, y_dir)

    # Datasets
    train_dataset = PriceDataset(X_train, yr_train, yd_train)
    val_dataset   = PriceDataset(X_val, yr_val, yd_val)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    input_dim = X.shape[-1]
    model = LSTMBaseline(input_dim).to(device)

    # Losses
    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------------- TRAIN LOOP -------------------
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_train_batches = 0

        for batch_x, batch_return, batch_dir in train_loader:
            batch_x = batch_x.to(device)
            batch_return = batch_return.to(device).squeeze()
            batch_dir = batch_dir.to(device)

            optimizer.zero_grad()
            dir_logits, return_pred = model(batch_x)
            return_pred = return_pred.squeeze()

            loss1 = loss_cls(dir_logits, batch_dir)
            loss2 = loss_reg(return_pred, batch_return)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_train_batches += 1

        # validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_x, batch_return, batch_dir in val_loader:
                batch_x = batch_x.to(device)
                batch_return = batch_return.to(device).squeeze()
                batch_dir = batch_dir.to(device)

                dir_logits, return_pred = model(batch_x)
                return_pred = return_pred.squeeze()

                loss1 = loss_cls(dir_logits, batch_dir)
                loss2 = loss_reg(return_pred, batch_return)

                val_loss += (loss1 + loss2).item()
                num_val_batches += 1

        avg_train_loss = total_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )


    return model
