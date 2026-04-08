"""
NeBULA Dataset - EMGNet — Subject-Independent Classification
-------------------------------------------------------------

 Cross-subject EMG classification using a 3-block CNN baseline.
 Execution-phase windows only (100ms–1100ms post-onset).
 3 consecutive windows are concatenated per trial to provide
 temporal context across the active movement period.

"""


import os
import json
import argparse
import random
import copy
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report




# CONFIG
# ----------------------------------------------

@dataclass
class Config:
    data_dir: str = "../preprocessed"
    results_dir: str = "./results/emg_net"
    x_name: str = "X_emg_win.npy"
    y_name: str = "y_win.npy"
    s_name: str = "subject_ids_win.npy"
    t_name: str = "trial_ids_win.npy"

    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.30
    patience: int = 15
    seed: int = 42

    n_channels: int = 11
    n_classes: int = 3


# All possible window start positions from epoch.py
# (epoch length = 500, window size = 80, step = 40)
WINDOW_STARTS = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], dtype=np.int64)

# Keep only execution-phase windows (approx +100ms to +1100ms post-onset).
# EDA showed task-discriminative amplitude differences are concentrated here.
KEEP_WINDOW_STARTS = {120, 160, 200, 240, 280, 320}



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# DATA
# ----------------------------------------------

class EMGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 11, 240)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# MODEL
# ----------------------------------------------

class EMGNet(nn.Module):
    """
    EMGNet — 3-block 1D CNN for subject-independent EMG classification.

    Three convolutional blocks with progressively smaller kernels:
      Block 1 (kernel=9, 45ms): detects broad activation bursts
      Block 2 (kernel=7, 35ms): captures activation envelope shape
      Block 3 (kernel=5, 25ms): fine-grained temporal features

    AdaptiveAvgPool fixes the output to 5 timesteps regardless of
    input length — required for MPS (Apple GPU) compatibility.

    Input:  (B, 11, 240)  — 11 EMG channels, 240 samples (3 x 80)
    Output: (B, 3)
    """

    def __init__(self, n_channels: int = 11, n_classes: int = 3, dropout: float = 0.30):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: broad temporal features
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),   # 240 -> 120
            nn.Dropout(dropout),

            # Block 2: mid-range temporal features
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),   # 120 -> 60
            nn.Dropout(dropout),

            # Block 3: fine-grained temporal features
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(5),       # 60 -> 5 (MPS-safe fixed output)
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 128 * 5 = 640
            nn.Linear(128 * 5, 64),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# DATA LOADING
# ----------------------------------------------

def recover_window_starts(n_windows: int) -> np.ndarray:
    """
    Recover window start position for each row in the windowed array.
    Assumes epoch.py appended windows in a fixed repeating order per trial.
    """
    pos = np.arange(n_windows) % len(WINDOW_STARTS)
    return WINDOW_STARTS[pos]


def subject_split(subjects):
    """
    70/15/15 subject-level split.
    np.random.seed is set before this call so the split is deterministic.
    """
    subjects = np.unique(subjects)
    np.random.shuffle(subjects)

    train = subjects[:25]
    val   = subjects[25:30]
    test  = subjects[30:]

    return set(train), set(val), set(test)


def load_data(cfg: Config):
    X = np.load(os.path.join(cfg.data_dir, cfg.x_name)).astype(np.float32)
    y = np.load(os.path.join(cfg.data_dir, cfg.y_name)).astype(np.int64)
    s = np.load(os.path.join(cfg.data_dir, cfg.s_name)).astype(np.int64)
    trial_ids = np.load(os.path.join(cfg.data_dir, cfg.t_name)).astype(np.int64)

    # Convert labels from 1-indexed to 0-indexed if necessary
    if y.min() == 1:
        y = y - 1

    win_starts = recover_window_starts(len(X))

    X_seq, y_seq, s_seq = [], [], []

    # Build 3-window sequences trial-by-trial
    # Each (subject, trial) pair produces len(exec_windows) - 2 sequences
    unique_keys = sorted(set(zip(s.tolist(), trial_ids.tolist())))

    for sid, tid in unique_keys:
        mask = (s == sid) & (trial_ids == tid)

        X_t  = X[mask]
        y_t  = y[mask]
        ws_t = win_starts[mask]

        # Sort windows chronologically within the trial
        order = np.argsort(ws_t)
        X_t  = X_t[order]
        y_t  = y_t[order]
        ws_t = ws_t[order]

        # Keep only execution-phase windows
        keep = np.isin(ws_t, list(KEEP_WINDOW_STARTS))
        X_t  = X_t[keep]
        y_t  = y_t[keep]

        if len(X_t) < 3:
            continue

        # Slide a 3-window sequence over the execution windows
        for i in range(len(X_t) - 2):
            # Concatenate along time axis: (11,80) x3 -> (11,240)
            X_seq.append(np.concatenate([X_t[i], X_t[i + 1], X_t[i + 2]], axis=1))
            y_seq.append(y_t[i + 2])   # label comes from the latest window
            s_seq.append(sid)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.int64),
        np.array(s_seq, dtype=np.int64),
    )


def make_loaders(X, y, s, batch_size: int):
    train_s, val_s, test_s = subject_split(s)

    train_idx = np.array([sid in train_s for sid in s])
    val_idx   = np.array([sid in val_s   for sid in s])
    test_idx  = np.array([sid in test_s  for sid in s])

    train_loader = DataLoader(EMGDataset(X[train_idx], y[train_idx]),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(EMGDataset(X[val_idx],   y[val_idx]),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(EMGDataset(X[test_idx],  y[test_idx]),
                              batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_s, val_s, test_s


# TRAIN
# ----------------------------------------------

def compute_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to handle minor class imbalance."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss   = criterion(logits, yb)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

    mean_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return mean_loss, acc, f1, y_true, y_pred


def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = get_device()

    X, y, s = load_data(cfg)
    train_loader, val_loader, test_loader, train_s, val_s, test_s = make_loaders(
        X, y, s, cfg.batch_size
    )

    model = EMGNet(
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 65)
    print("  EMGNet — Subject-Independent NeBULA Classification")
    print("-" * 65)
    print(f"  Device         : {device}")
    print(f"  Input shape    : {X.shape}")
    print(f"  Window starts  : {sorted(list(KEEP_WINDOW_STARTS))}")
    print(f"  Train subjects : {sorted([int(x) for x in train_s])}")
    print(f"  Val subjects   : {sorted([int(x) for x in val_s])}")
    print(f"  Test subjects  : {sorted([int(x) for x in test_s])}")
    print(f"  Train samples  : {len(train_loader.dataset)}")
    print(f"  Val samples    : {len(val_loader.dataset)}")
    print(f"  Test samples   : {len(test_loader.dataset)}")
    print(f"  Parameters     : {n_params:,}")
    print()

    weights   = compute_class_weights(train_loader.dataset.y.numpy(), cfg.n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_f1  = -1.0
    best_val_loss = float("inf")
    best_state   = None
    best_epoch   = 0
    no_improve   = 0
    history      = []

    for epoch in range(1, cfg.epochs + 1):
        tl, ta, tf, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va, vf, _, _ = run_epoch(model, val_loader,   criterion, device)
        scheduler.step(vf)

        history.append({
            "epoch": epoch,
            "train_loss": tl, "train_acc": ta, "train_f1": tf,
            "val_loss":   vl, "val_acc":   va, "val_f1":   vf,
        })

        print(
            f"  Epoch {epoch:03d} | "
            f"train loss={tl:.4f} acc={ta:.3f} f1={tf:.3f} | "
            f"val loss={vl:.4f} acc={va:.3f} f1={vf:.3f}"
        )

        improved = (vf > best_val_f1) or (vf == best_val_f1 and vl < best_val_loss)

        if improved:
            best_val_f1   = vf
            best_val_loss = vl
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"\n  Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, device
    )
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print("\n" + "-" * 65)
    print("  TEST RESULTS")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Accuracy      : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  F1 macro      : {test_f1:.4f}")
    print("  Confusion matrix:")
    print(cm)
    print("-" * 65)

    prefix = os.path.join(cfg.results_dir, "emg_net")
    torch.save(model.state_dict(), prefix + ".pt")
    np.save(prefix + "_history.npy", np.array(history, dtype=object))

    summary = {
        "model": "EMGNet",
        "experiment": "subject_independent",
        "config": asdict(cfg),
        "device": str(device),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "train_subjects": sorted([int(x) for x in train_s]),
        "val_subjects":   sorted([int(x) for x in val_s]),
        "test_subjects":  sorted([int(x) for x in test_s]),
        "n_train_samples": int(len(train_loader.dataset)),
        "n_val_samples":   int(len(val_loader.dataset)),
        "n_test_samples":  int(len(test_loader.dataset)),
        "kept_window_starts": sorted([int(x) for x in KEEP_WINDOW_STARTS]),
        "sequence_length": 3,
        "classification_report": report,
    }

    with open(prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Model   → {prefix}.pt")
    print(f"  History → {prefix}_history.npy")
    print(f"  Summary → {prefix}_summary.json")


# CLI
# ----------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EMGNet subject-independent NeBULA classification")
    parser.add_argument("--data",     default="../preprocessed")
    parser.add_argument("--results",  default="./results/emg_net")
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--wd",       type=float, default=1e-4)
    parser.add_argument("--dropout",  type=float, default=0.30)
    parser.add_argument("--patience", type=int,   default=15)
    parser.add_argument("--seed",     type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_dir=args.data,
        results_dir=args.results,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()