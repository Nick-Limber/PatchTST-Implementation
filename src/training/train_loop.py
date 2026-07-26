import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import os
import sys
from src.data_layer.synthetic import generate_dataset
from src.training.dataset import SlidingWindowDataset
from src.training.losses import POINT_LOSSES
from src.models.patchtst.model import PatchTST

df = generate_dataset(n_series=10, n_days=250, seed=10, start_date="2022-01-01")

def train_validate_split(df: pd.DataFrame, train_percent: float = 0.8):
    train_frames = []
    val_frames = []

    for series_id in df["series_id"].unique():
        series_df = (
            df[df["series_id"] == series_id]
            .sort_values("timestamp")
        )

        split_idx = int(len(series_df) * train_percent)
        train_frames.append(series_df.iloc[:split_idx])
        val_frames.append(series_df.iloc[split_idx:])
        
    train_df = pd.concat(train_frames, ignore_index=True)
    val_df   = pd.concat(val_frames,   ignore_index=True)

    return train_df, val_df


def create_data_loaders(
    train_df, val_df, context_len=28, horizon=7,
    train_stride=1, batch_size=32,
):
    train_dataset = SlidingWindowDataset(train_df, context_len, horizon, stride=train_stride)
    val_dataset   = SlidingWindowDataset(val_df,   context_len, horizon, stride=horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_fn, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_x, batch_y, _ in loader:
        optimizer.zero_grad()
        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def validate(model, loader, loss_fn):

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def train(
    model: nn.Module,
    df: pd.DataFrame,
    config: dict,
    checkpoint_path: str,
    loss_fn=None,
) -> dict:
 
    training_cfg = config["training"]
    model_cfg    = config["model"]
    data_cfg     = config.get("data", {})
 
    max_epochs   = training_cfg["epochs"]
    lr           = training_cfg["learning_rate"]
    batch_size   = training_cfg["batch_size"]
    patience     = training_cfg["early_stopping_patience"]
    grad_clip    = training_cfg.get("grad_clip", 1.0)
    context_len  = model_cfg["context_len"]
    horizon      = model_cfg["horizon"]
    train_pct    = data_cfg.get("train_percent", 0.8)
 
    if loss_fn is None:
        loss_name = training_cfg.get("loss", "mse")
        loss_fn   = POINT_LOSSES[loss_name]
 
    train_df, val_df = train_validate_split(df, train_percent=train_pct)
 
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, context_len, horizon, batch_size
    )
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      
        factor=0.5,      
        patience=3,
    )
 
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
 
    # Make sure the checkpoint directory exists before the first save.
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
 
    print(f"Starting training: {max_epochs} epochs, lr={lr}, loss={loss_fn.__name__}")
    print(f"Train windows: {len(train_loader.dataset)}  "
          f"Val windows: {len(val_loader.dataset)}")
    print("-" * 60)
 
    for epoch in range(1, max_epochs + 1):
 
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, grad_clip
        )
 
        val_loss = validate(model, val_loader, loss_fn)
 
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
 
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            improved_marker = " ← best"
        else:
            epochs_without_improvement += 1
            improved_marker = ""
 
        print(
            f"Epoch {epoch:03d}/{max_epochs}  "
            f"train={train_loss:.4f}  "
            f"val={val_loss:.4f}  "
            f"lr={current_lr:.6f}"
            f"{improved_marker}"
        )
 
        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch} -- "
                f"val loss did not improve for {patience} consecutive epochs."
            )
            break
 
    print("-" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best weights saved to: {checkpoint_path}")
 
    return history
 
 
 
class LinearBaseline(nn.Module):

    def __init__(self, context_len: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(context_len, horizon)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
 
if __name__ == "__main__":

    config = {
        "model": {
            "context_len": 96,
            "horizon": 28,
            "patch_len": 16,
            "stride": 8,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "d_feedforward": 256,
            "dropout": 0.1,
        },
        "data": {
            "train_percent": 0.8,
        },
        "training": {
            "epochs": 150,
            "learning_rate": 0.00001,
            "batch_size": 32,
            "early_stopping_patience": 10,
            "grad_clip": 1.0,
            "loss": "mse",
        },
    }
 
    df = generate_dataset(n_series=10, n_days=1000, seed=0, noise_std=2.0)
 
    model = PatchTST(
            context_len=config["model"]["context_len"],
            patch_len=config["model"]["patch_len"],
            stride=config["model"]["stride"],
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["n_heads"],
            n_layers=config["model"]["n_layers"],
            d_feedforward=config["model"]["d_feedforward"],
            dropout=config["model"]["dropout"],
            horizon=config["model"]["horizon"]
    )
 
    history = train(
        model=model,
        df=df,
        config=config,
        checkpoint_path="experiments/runs/patchtst_baseline/best.pt",
    )
 
    print("\nLoss curve (first and last 3 epoch):")
    for i, (tr, va) in enumerate(
        zip(history["train_loss"], history["val_loss "]), start=1
    ):
        if i <= 3 or i > len(history["train_loss"]) - 3:
            print(f"  epoch {i:02d}: train={tr:.4f}  val={va:.4f}")
 










