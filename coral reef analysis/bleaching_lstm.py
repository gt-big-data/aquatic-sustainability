"""
LSTM model for coral bleaching severity prediction.

Input:  sequence windows of thermal stress features (shape inferred from data)
Output: ordinal bleaching severity classes (class count inferred from data)

Architecture:
  - Unidirectional LSTM (forward-only temporal encoding)
  - 2 LSTM layers with dropout
  - Attention mechanism (learns which weeks matter most)
  - Static features (lat, lon) concatenated before classification head
  - Ordinal-aware loss option
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import time
import os
import argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15  # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 52
N_FEATURES = 7
N_CLASSES = 4
HIDDEN_SIZE = 128
N_LAYERS = 2
DROPOUT = 0.3

np.random.seed(SEED)
torch.manual_seed(SEED)


def resolve_default_data_path():
    """Prefer local sequences.npz, then project datasets/sequences.npz."""
    cwd_candidate = Path("sequences.npz")
    script_candidate = Path(__file__).resolve().parent / "datasets" / "sequences.npz"
    if cwd_candidate.exists():
        return cwd_candidate
    if script_candidate.exists():
        return script_candidate
    return cwd_candidate


def load_state_dict_compat(model, checkpoint_path):
    """Load checkpoint across PyTorch versions that may not support weights_only."""
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)


def class_names_for(n_classes):
    """Return display names for common bleaching label schemes."""
    if n_classes == 4:
        return ["None (0%)", "Low (1-10%)", "Moderate (10-50%)", "Severe (>50%)"]
    if n_classes == 3:
        return ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
    return [f"Class {i}" for i in range(n_classes)]

# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class BleachingDataset(Dataset):
    def __init__(self, X, y, meta):
        # Normalize features per-feature across the dataset
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
        # Static features: lat, lon (normalized)
        self.static = torch.FloatTensor(meta[:, :2])  # lat, lon
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.static[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────
# ATTENTION LAYER
# ──────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """Learns which timesteps are most important for prediction."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        scores = self.attn(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)            # (batch, seq_len)
        context = torch.bmm(
            weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_size)
        return context, weights


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
class BleachingLSTM(nn.Module):
    def __init__(
        self,
        n_features=N_FEATURES,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        n_classes=N_CLASSES,
        n_static=2,
        dropout=DROPOUT,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Unidirectional LSTM (forward-only)
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Attention over LSTM outputs
        self.attention = TemporalAttention(hidden_size)
        
        # Classification head
        # Combines: attention context + last hidden state + static features
        head_input_size = hidden_size + hidden_size + n_static
        
        self.classifier = nn.Sequential(
            nn.Linear(head_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, n_classes),
        )
    
    def forward(self, x_seq, x_static):
        # x_seq: (batch, seq_len, n_features)
        # x_static: (batch, 2)
        
        # Project input features
        x = self.input_proj(x_seq)  # (batch, seq_len, hidden//2)
        
        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)
        
        # Attention-weighted context
        attn_context, attn_weights = self.attention(lstm_out)  # (batch, hidden)
        
        # Last timestep hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        
        # Combine everything
        combined = torch.cat([attn_context, last_hidden, x_static], dim=1)
        
        logits = self.classifier(combined)  # (batch, n_classes)
        return logits, attn_weights


# ──────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────
def get_class_weights(y, n_classes):
    """Inverse frequency weighting for imbalanced classes."""
    counts = Counter(y.tolist())
    total = sum(counts.values())
    weights = {c: total / (len(counts) * n) for c, n in counts.items()}
    return torch.FloatTensor([weights.get(i, 1.0) for i in range(n_classes)]).to(DEVICE)


def get_weighted_sampler(y):
    """WeightedRandomSampler for balanced batches."""
    counts = Counter(y.tolist())
    class_weights = {c: 1.0 / n for c, n in counts.items()}
    sample_weights = [class_weights[label] for label in y.tolist()]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def normalize_features(X_train, X_val, X_test):
    """Per-feature normalization using training set statistics."""
    # X shape: (N, seq_len, n_features)
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std


def normalize_static(meta_train, meta_val, meta_test):
    """Normalize lat/lon."""
    mean = meta_train[:, :2].mean(axis=0)
    std = meta_train[:, :2].std(axis=0)
    std[std == 0] = 1
    
    meta_train_norm = meta_train.copy()
    meta_val_norm = meta_val.copy()
    meta_test_norm = meta_test.copy()
    
    meta_train_norm[:, :2] = (meta_train[:, :2] - mean) / std
    meta_val_norm[:, :2] = (meta_val[:, :2] - mean) / std
    meta_test_norm[:, :2] = (meta_test[:, :2] - mean) / std
    
    return meta_train_norm, meta_val_norm, meta_test_norm, mean, std

# ──────────────────────────────────────────────────────────────
# FOCAL LOSS
# ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor

    def forward(self, logits, targets):
        # Compute p_t from unweighted log-probs; class weighting is applied once
        # as alpha_t to avoid distorting p_t (and over-amplifying imbalance).
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
        else:
            alpha_t = 1.0

        focal_loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt
        return focal_loss.mean()


# ──────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────
def train(
    data_path,
    output_dir,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    loss_name="ce",
    focal_gamma=2.0,
    num_workers=0,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path).resolve()
    best_model_path = output_dir / "best_model.pt"
    results_path = output_dir / "results.npz"
    stats_path = output_dir / "normalization_stats.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    print("=" * 70)
    print("Loading data")
    print("=" * 70)
    print(f"  Data path: {data_path}")
    print(f"  Output dir: {output_dir}")
    data = np.load(data_path)
    X, y, meta = data["X"], data["y"], data["meta"]
    label_values = np.unique(y)
    bleach_bins = data["bleach_bins"] if "bleach_bins" in data.files else None
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, seq_len, n_features), got {X.shape}")
    if meta.ndim != 2 or meta.shape[1] < 2:
        raise ValueError(f"Expected meta to have at least 2 columns (lat/lon), got {meta.shape}")

    seq_len = int(X.shape[1])
    n_features = int(X.shape[2])
    n_classes = int(np.max(y)) + 1
    print(f"  X: {X.shape}, y: {y.shape}, meta: {meta.shape}")
    print(f"  Label values: {label_values.tolist()}")
    if bleach_bins is not None:
        print(f"  Bleach bins: {np.asarray(bleach_bins).tolist()}")
    print(f"  Inferred seq_len={seq_len}, n_features={n_features}, n_classes={n_classes}")
    print(f"  Device: {DEVICE}")
    
    # Stratified split: 70/15/15
    X_trainval, X_test, y_trainval, y_test, m_trainval, m_test = train_test_split(
        X, y, meta, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
        X_trainval, y_trainval, m_trainval, test_size=0.176, stratify=y_trainval, random_state=SEED
    )  # 0.176 of 0.85 ≈ 0.15 of total
    
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Train class dist: {dict(Counter(y_train.tolist()))}")
    
    # Normalize
    X_train, X_val, X_test, feat_mean, feat_std = normalize_features(X_train, X_val, X_test)
    m_train, m_val, m_test, static_mean, static_std = normalize_static(m_train, m_val, m_test)
    feat_mean = feat_mean.astype(np.float32, copy=False)
    feat_std = feat_std.astype(np.float32, copy=False)
    static_mean = static_mean.astype(np.float32, copy=False)
    static_std = static_std.astype(np.float32, copy=False)
    
    # Datasets and loaders
    train_ds = BleachingDataset(X_train, y_train, m_train)
    val_ds = BleachingDataset(X_val, y_val, m_val)
    test_ds = BleachingDataset(X_test, y_test, m_test)
    
    # sampler = get_weighted_sampler(y_train)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model
    model = BleachingLSTM(n_features=n_features, n_classes=n_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Loss and optimizer
    class_weights = get_class_weights(y_train, n_classes=n_classes)
    print(f"  Class weights: {class_weights.detach().cpu().numpy().tolist()}")
    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        raise ValueError(f"Unsupported loss_name: {loss_name}")
    print(f"  Loss: {loss_name} (focal_gamma={focal_gamma})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    best_val_loss = float("inf")
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x_seq, x_static, labels in train_loader:
            x_seq = x_seq.to(DEVICE)
            x_static = x_static.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(x_seq, x_static)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        scheduler.step()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_seq, x_static, labels in val_loader:
                x_seq = x_seq.to(DEVICE)
                x_static = x_static.to(DEVICE)
                labels = labels.to(DEVICE)
                
                logits, _ = model(x_seq, x_static)
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
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
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
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s{marker}"
            )
        
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break
    
    # ──────────────────────────────────────────────────────────
    # EVALUATION
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluation on test set")
    print("=" * 70)
    
    load_state_dict_compat(model, best_model_path)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_attn = []
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for x_seq, x_static, labels in test_loader:
            x_seq = x_seq.to(DEVICE)
            x_static = x_static.to(DEVICE)
            
            logits, attn_weights = model(x_seq, x_static)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_attn.append(attn_weights.cpu().numpy())
            all_probs.append(probs)
            all_logits.append(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_attn = np.concatenate(all_attn, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
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
    print(f"{'':>15} | " + " | ".join(f"{n:>8}" for n in class_names))
    print("-" * 70)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>15} | " + " | ".join(f"{v:>8d}" for v in row))
    
    # Attention analysis: which weeks matter most?
    print("\nAttention Analysis (mean attention weight by week):")
    mean_attn = all_attn.mean(axis=0)
    top_k = min(10, seq_len)
    top_weeks = np.argsort(mean_attn)[::-1][:top_k]
    print(f"  Top {top_k} most attended weeks (0=oldest, {seq_len-1}=most recent):")
    for w in top_weeks:
        print(f"    Week {w:2d} (t-{seq_len-w:2d} weeks before event): {mean_attn[w]:.4f}")
    
    # Save everything
    np.savez(
        results_path,
        predictions=all_preds,
        labels=all_labels,
        attention_weights=all_attn,
        probabilities=all_probs,
        logits=all_logits,
        class_names=np.array(class_names, dtype=object),
        history_train_loss=history["train_loss"],
        history_val_loss=history["val_loss"],
        history_val_acc=history["val_acc"],
        feat_mean=feat_mean,
        feat_std=feat_std,
        static_mean=static_mean,
        static_std=static_std,
    )
    np.savez_compressed(
        stats_path,
        feat_mean=feat_mean,
        feat_std=feat_std,
        static_mean=static_mean,
        static_std=static_std,
    )
    print(f"\n  Results saved to {results_path}")
    print(f"  Best model saved to {best_model_path}")
    print(f"  Normalization stats saved to {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM coral bleaching classifier.")
    parser.add_argument(
        "--data-path",
        default=str(resolve_default_data_path()),
        help="Path to sequences.npz containing arrays X, y, meta.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for best_model.pt and results.npz.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default="focal",
        help="Training loss (default: focal).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss (default: 2.0).",
    )
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
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        num_workers=args.num_workers,
    )
