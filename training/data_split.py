import torch

def split_data(X, y_price, y_dir, train_ratio=0.7, val_ratio=0.15):
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    X_train, y_price_train, y_dir_train = X[:train_end], y_price[:train_end], y_dir[:train_end]
    X_val,   y_price_val,   y_dir_val   = X[train_end:val_end], y_price[train_end:val_end], y_dir[train_end:val_end]
    X_test,  y_price_test,  y_dir_test  = X[val_end:], y_price[val_end:], y_dir[val_end:]

    return (
        (X_train, y_price_train, y_dir_train),
        (X_val, y_price_val, y_dir_val),
        (X_test, y_price_test, y_dir_test)
    )
