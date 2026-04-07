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


# ============================================================
# EEGNet — Subject-Independent Classification (NeBULA)
# ------------------------------------------------------------
# Cross-subject EEG classification using EEGNet (Lawhern et al., 2018).
# Onset-centred windows are selected based on EEG diagnostics.
# ============================================================


# ================= CONFIG =================

@dataclass
class Config:
    data_dir: str = "../preprocessed"
    results_dir: str = "./results/eeg_net"
    x_name: str = "X_eeg_win.npy"
    y_name: str = "y_win.npy"
    s_name: str = "subject_ids_win.npy"
    t_name: str = "trial_ids_win.npy"

    batch_size: int = 64
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.25
    patience: int = 40
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42

    n_channels: int = 15
    n_samples: int = 80
    n_classes: int = 3


# All possible window start positions from epoch.py
# (epoch length = 500, window size = 80, step = 40)
WINDOW_STARTS = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], dtype=np.int64)

# Keep only windows around movement onset (-300ms, -100ms, +100ms)
# Selected based on EEG diagnostic analysis
KEEP_WINDOW_STARTS = {40, 80, 120}
# ==========================================


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


# ================= DATA =================

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # EEGNet expects (batch, 1, channels, samples)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================= MODEL =================

class EEGNet(nn.Module):
    """
    EEGNet — compact convolutional network for EEG classification.
    Based on Lawhern et al. (2018), Journal of Neural Engineering.

    Block 1: Temporal Conv2d (1 x 32) + Depthwise spatial Conv2d (15 x 1).
             Temporal and spatial learning are kept separate.
    Block 2: Depthwise + pointwise separable convolution.
             Reduces parameters while maintaining representational power.

    Input:  (B, 1, 15, 80)
    Output: (B, 3)
    """

    def __init__(
        self,
        n_channels=15,
        win_size=80,
        n_classes=3,
        F1=16,
        D=2,
        F2=32,
        dropout=0.25
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            # Temporal filter: learns when patterns occur
            nn.Conv2d(1, F1, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(F1),

            # Depthwise spatial filter: learns which channels activate together
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),   # 80 -> 20
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 8), padding=(0, 4),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),   # 20 -> 5
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, win_size)
            flat_size = self.block2(self.block1(dummy)).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


# ================= DATA LOADING =================

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

    if y.min() == 1:
        y = y - 1

    win_starts = recover_window_starts(len(X))
    keep_mask = np.isin(win_starts, list(KEEP_WINDOW_STARTS))

    X = X[keep_mask]
    y = y[keep_mask]
    s = s[keep_mask]
    trial_ids = trial_ids[keep_mask]
    win_starts = win_starts[keep_mask]

    return X, y, s, trial_ids, win_starts


def make_loaders(X, y, s, batch_size: int):
    train_s, val_s, test_s = subject_split(s)

    train_idx = np.array([sid in train_s for sid in s])
    val_idx   = np.array([sid in val_s for sid in s])
    test_idx  = np.array([sid in test_s for sid in s])

    train_loader = DataLoader(EEGDataset(X[train_idx], y[train_idx]),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(EEGDataset(X[val_idx], y[val_idx]),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(EEGDataset(X[test_idx], y[test_idx]),
                              batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_s, val_s, test_s


# ================= TRAIN =================

def run_epoch(model, loader, criterion, optimizer=None, device="cpu", grad_clip=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds_all, trues_all = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(xb)
            loss = criterion(out, yb)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        total_loss += loss.item() * len(yb)
        preds_all.extend(out.argmax(1).detach().cpu().tolist())
        trues_all.extend(yb.detach().cpu().tolist())

    mean_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(trues_all, preds_all)
    f1 = f1_score(trues_all, preds_all, average="macro", zero_division=0)

    return mean_loss, acc, f1, trues_all, preds_all


def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = get_device()

    X, y, s, trial_ids, win_starts = load_data(cfg)

    print("=" * 72)
    print("  EEGNet — Subject-Independent NeBULA Classification")
    print("=" * 72)
    print(f"  Device         : {device}")
    print(f"  Input shape    : {X.shape}")
    print(f"  Window starts  : {sorted(list(KEEP_WINDOW_STARTS))}")
    print(f"  Subjects       : {sorted(np.unique(s).tolist())}")

    train_loader, val_loader, test_loader, train_s, val_s, test_s = make_loaders(
        X, y, s, cfg.batch_size
    )

    print(f"  Train subjects : {sorted([int(x) for x in train_s])}")
    print(f"  Val subjects   : {sorted([int(x) for x in val_s])}")
    print(f"  Test subjects  : {sorted([int(x) for x in test_s])}")

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        ys = loader.dataset.y.numpy()
        print(
            f"  {name:5s}: {len(ys):5d} windows  "
            f"(Task0={(ys == 0).sum()}, Task1={(ys == 1).sum()}, Task2={(ys == 2).sum()})"
        )

    model = EEGNet(
        n_channels=cfg.n_channels,
        win_size=cfg.n_samples,
        n_classes=cfg.n_classes,
        F1=16,
        D=2,
        F2=32,
        dropout=cfg.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters     : {n_params:,}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tl, ta, tf, _, _ = run_epoch(model, train_loader, criterion,
                                     optimizer=optimizer, device=device,
                                     grad_clip=cfg.grad_clip)
        vl, va, vf, _, _ = run_epoch(model, val_loader, criterion,
                                     optimizer=None, device=device,
                                     grad_clip=cfg.grad_clip)
        scheduler.step(vf)

        history.append({
            "epoch": epoch,
            "train_loss": tl, "train_acc": ta, "train_f1": tf,
            "val_loss": vl,   "val_acc": va,   "val_f1": vf,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"  Epoch {epoch:03d} | "
            f"train loss={tl:.4f} acc={ta:.3f} f1={tf:.3f} | "
            f"val loss={vl:.4f} acc={va:.3f} f1={vf:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        improved = (vf > best_val_f1) or (vf == best_val_f1 and vl < best_val_loss)

        if improved:
            best_val_f1 = vf
            best_val_loss = vl
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"\n  Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    model.load_state_dict(best_weights)
    model.eval()

    test_loss, test_acc, test_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, optimizer=None, device=device, grad_clip=cfg.grad_clip
    )

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print("\n" + "=" * 72)
    print("  TEST RESULTS")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Accuracy      : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  F1 macro      : {test_f1:.4f}")
    print("  Confusion matrix:")
    print(cm)
    print("=" * 72)

    prefix = os.path.join(cfg.results_dir, "eeg_net")
    torch.save(model.state_dict(), prefix + ".pt")
    np.save(prefix + "_history.npy", np.array(history, dtype=object))

    summary = {
        "model": "EEGNet",
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
        "confusion_matrix": [[int(v) for v in row] for row in cm.tolist()],
        "train_subjects": [int(x) for x in sorted(list(train_s))],
        "val_subjects":   [int(x) for x in sorted(list(val_s))],
        "test_subjects":  [int(x) for x in sorted(list(test_s))],
        "n_train_samples": int(len(train_loader.dataset)),
        "n_val_samples":   int(len(val_loader.dataset)),
        "n_test_samples":  int(len(test_loader.dataset)),
        "kept_window_starts": sorted([int(x) for x in KEEP_WINDOW_STARTS]),
        "classification_report": report,
    }

    with open(prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Model   → {prefix}.pt")
    print(f"  History → {prefix}_history.npy")
    print(f"  Summary → {prefix}_summary.json")


# ================= CLI =================

def parse_args():
    parser = argparse.ArgumentParser(description="EEGNet subject-independent NeBULA classification")
    parser.add_argument("--data",     default="../preprocessed")
    parser.add_argument("--results",  default="./results/eeg_net")
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--dropout",  type=float, default=0.25)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--patience", type=int,   default=40)
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
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()