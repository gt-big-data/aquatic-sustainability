"""
2-layer MLP baseline for coral bleaching severity prediction.

Input:  flattened features from final week values (N, F)
Output: bleaching severity class labels

Evaluation output matches bleaching_lstm.py style:
  - Classification report
  - Confusion matrix
"""

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)


def resolve_default_data_path() -> Path:
    candidates = [
        Path("flatten.npz"),
        Path("flattened.npz"),
        Path("datasets/flatten.npz"),
        Path("datasets/flattened.npz"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("flattened.npz")


def load_state_dict_compat(model: nn.Module, checkpoint_path: Path) -> None:
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)


class FlattenedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = HIDDEN_DIM_1,
        hidden_dim_2: int = HIDDEN_DIM_2,
        dropout: float = DROPOUT,
        n_classes: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim_2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_val_norm, X_test_norm, mean, std


def get_class_weights(y: np.ndarray, n_classes: int):
    counts = Counter(y.tolist())
    total = sum(counts.values())
    weights = {c: total / (len(counts) * n) for c, n in counts.items()}
    return torch.FloatTensor([weights.get(i, 1.0) for i in range(n_classes)]).to(DEVICE)


def class_names_for(n_classes: int):
    if n_classes == 4:
        return ["None (0%)", "Low (1-10%)", "Moderate (10-50%)", "Severe (>50%)"]
    if n_classes == 3:
        return ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
    return [f"Class {i}" for i in range(n_classes)]


def train(
    data_path,
    output_dir,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    hidden_dim_1=HIDDEN_DIM_1,
    hidden_dim_2=HIDDEN_DIM_2,
    dropout=DROPOUT,
    num_workers=0,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path).resolve()
    best_model_path = output_dir / "best_mlp_model.pt"
    results_path = output_dir / "mlp_results.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("=" * 70)
    print("Loading data")
    print("=" * 70)
    print(f"  Data path: {data_path}")
    print(f"  Output dir: {output_dir}")

    data = np.load(data_path, allow_pickle=True)
    if "X" not in data.files or "y" not in data.files:
        raise KeyError("Input .npz must contain arrays: X and y")

    X = data["X"]
    y = data["y"]
    if X.ndim != 2:
        raise ValueError(f"Expected flattened X with shape (N, F), got {X.shape}")

    n_classes = int(y.max()) + 1
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  Inferred classes: {n_classes}")
    print(f"  Device: {DEVICE}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.176,
        stratify=y_trainval,
        random_state=SEED,
    )

    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Train class dist: {dict(Counter(y_train.tolist()))}")

    X_train, X_val, X_test, feat_mean, feat_std = normalize_features(X_train, X_val, X_test)

    train_ds = FlattenedDataset(X_train, y_train)
    val_ds = FlattenedDataset(X_val, y_val)
    test_ds = FlattenedDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MLPBaseline(
        input_dim=X.shape[1],
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
        n_classes=n_classes,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,} ({trainable_params:,} trainable)")

    class_weights = get_class_weights(y_train, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, labels in train_loader:
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s{marker}"
            )

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 70)
    print("Evaluation on test set")
    print("=" * 70)

    load_state_dict_compat(model, best_model_path)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    class_names = class_names_for(n_classes)
    label_order = list(range(n_classes))

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=label_order,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=label_order)
    print(f"{'':>15} | " + " | ".join(f"{n:>12}" for n in class_names))
    print("-" * (18 + 15 * len(class_names)))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>15} | " + " | ".join(f"{v:>12d}" for v in row))

    np.savez(
        results_path,
        predictions=all_preds,
        labels=all_labels,
        probabilities=all_probs,
        history_train_loss=history["train_loss"],
        history_val_loss=history["val_loss"],
        history_val_acc=history["val_acc"],
        feat_mean=feat_mean,
        feat_std=feat_std,
        class_names=np.array(class_names, dtype=object),
    )
    print(f"\n  Results saved to {results_path}")
    print(f"  Best model saved to {best_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2-layer MLP baseline classifier.")
    parser.add_argument(
        "--data-path",
        default=str(resolve_default_data_path()),
        help="Path to flattened .npz containing arrays X and y.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for best_mlp_model.pt and mlp_results.npz.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--hidden-dim-1", type=int, default=HIDDEN_DIM_1)
    parser.add_argument("--hidden-dim-2", type=int, default=HIDDEN_DIM_2)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) - 1),
        help="DataLoader workers. Defaults to max(0, SLURM_CPUS_PER_TASK-1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        dropout=args.dropout,
        num_workers=args.num_workers,
    )
