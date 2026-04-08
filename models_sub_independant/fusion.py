"""
NeBULA Dataset - EEG-EMG Fusion — Subject-Independent Classification
-------------------------------------------------------------

EMG-anchored gated fusion model.

Architecture:
  - EMG branch: same CNN+BiLSTM+attention as the standalone EMG model.
                Produces the primary classification logits.
  - EEG branch: small lightweight CNN. Processes pre-movement onset
                windows only (-300ms to +100ms). Produces a 32-dim embedding.
  - Gate:       sigmoid scalar learned from both modalities.
                Initialised near zero (bias=-1.5) so EEG starts muted.
  - Output:     EMG_logits + gate * EEG_logits

"""

import os
import json
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# CONFIG
# ----------------------------------------------

DATA_DIR    = "../preprocessed"
RESULTS_DIR = "./results/fusion"

BATCH_SIZE = 64
EPOCHS     = 160
LR         = 5e-4
DROPOUT    = 0.30
PATIENCE   = 30
SEED       = 42
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05

N_EEG_CHANNELS = 15
N_EMG_CHANNELS = 11
N_CLASSES      = 3

# All possible window start positions from epoch.py
WINDOW_STARTS = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], dtype=np.int64)

# EMG: execution-phase windows (+100ms to +1100ms post-onset)
EMG_KEEP_WINDOW_STARTS = {120, 160, 200, 240, 280, 320}

# EEG: pre-movement and onset windows (-300ms, -100ms, +100ms)
# These are the windows most likely to contain any cortical preparation signal
EEG_CONTEXT_STARTS = [40, 80, 120]



def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def recover_window_starts(n_windows: int) -> np.ndarray:
    """
    Recover window start position for each row in the windowed array.
    Assumes epoch.py appended windows in a fixed repeating order per trial.
    """
    pos = np.arange(n_windows) % len(WINDOW_STARTS)
    return WINDOW_STARTS[pos]


def compute_emg_features(X: np.ndarray) -> np.ndarray:
    """
    Handcrafted time-domain features per channel.

    X shape: (N, C, T)
    Returns: (N, 4*C)

    Features: RMS, mean activation, std, waveform length.
    RMS is the most discriminative (Task1 > Task2 > Task3 consistently).
    """
    rms  = np.sqrt(np.mean(X ** 2, axis=2))
    mean = np.mean(X, axis=2)
    std  = np.std(X, axis=2)
    wl   = np.sum(np.abs(np.diff(X, axis=2)), axis=2)
    return np.concatenate([rms, mean, std, wl], axis=1).astype(np.float32)


#  DATA
# ----------------------------------------------

class FusionDataset(Dataset):
    def __init__(self, X_emg_raw, X_emg_feat, X_eeg_raw, y):
        self.X_emg_raw  = torch.tensor(X_emg_raw,  dtype=torch.float32)
        self.X_emg_feat = torch.tensor(X_emg_feat, dtype=torch.float32)
        self.X_eeg_raw  = torch.tensor(X_eeg_raw,  dtype=torch.float32)
        self.y          = torch.tensor(y,           dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_emg_raw[idx],
            self.X_emg_feat[idx],
            self.X_eeg_raw[idx],
            self.y[idx],
        )


def load_raw_arrays():
    # Load all windowed arrays from disk.
    X_eeg = np.load(os.path.join(DATA_DIR, "X_eeg_win.npy")).astype(np.float32)
    X_emg = np.load(os.path.join(DATA_DIR, "X_emg_win.npy")).astype(np.float32)
    y     = np.load(os.path.join(DATA_DIR, "y_win.npy")).astype(np.int64)
    s     = np.load(os.path.join(DATA_DIR, "subject_ids_win.npy")).astype(np.int64)
    trial_ids = np.load(os.path.join(DATA_DIR, "trial_ids_win.npy")).astype(np.int64)

    if y.min() == 1:
        y = y - 1

    win_starts = recover_window_starts(len(y))
    return X_eeg, X_emg, y, s, trial_ids, win_starts


def create_fusion_samples():
    """
    Build paired EEG-EMG samples trial-by-trial.

    For each (subject, trial):
      EEG context : concatenate onset windows [40, 80, 120] -> (15, 240)
                    Same EEG window is paired with each EMG sequence.

      EMG sequence: slide a 3-window window over execution windows
                    [120, 160, 200, 240, 280, 320] -> 4 sequences per trial,
                    each (11, 240).

    Output shapes:
      X_emg_raw  (N, 11, 240)
      X_emg_feat (N, 44)      — handcrafted features on the 240-sample EMG
      X_eeg_raw  (N, 15, 240)
      y          (N,)
      s          (N,)
    """
    X_eeg, X_emg, y, s, trial_ids, win_starts = load_raw_arrays()

    X_emg_raw_list  = []
    X_emg_feat_list = []
    X_eeg_raw_list  = []
    y_list = []
    s_list = []

    unique_keys = sorted(set(zip(s.tolist(), trial_ids.tolist())))

    for sid, tid in unique_keys:
        mask = (s == sid) & (trial_ids == tid)

        Xe = X_eeg[mask]
        Xm = X_emg[mask]
        yt = y[mask]
        ws = win_starts[mask]

        # Sort windows chronologically within the trial
        order = np.argsort(ws)
        Xe = Xe[order]
        Xm = Xm[order]
        yt = yt[order]
        ws = ws[order]

        # All windows have the same label — take the first
        label = yt[0]

        # ── EEG onset context ──
        # Concatenate the three pre/onset windows into a single (15, 240) array
        eeg_keep = np.isin(ws, EEG_CONTEXT_STARTS)
        Xe_ctx   = Xe[eeg_keep]
        ws_eeg   = ws[eeg_keep]

        # Skip trial if any onset window is missing
        if len(Xe_ctx) != len(EEG_CONTEXT_STARTS):
            continue

        eeg_order = np.argsort(ws_eeg)
        Xe_ctx    = Xe_ctx[eeg_order]
        eeg_raw   = np.concatenate([Xe_ctx[i] for i in range(len(Xe_ctx))], axis=1)  # (15, 240)

        # ── EMG execution sequence ──
        emg_keep = np.isin(ws, list(EMG_KEEP_WINDOW_STARTS))
        Xm_exec  = Xm[emg_keep]
        ws_emg   = ws[emg_keep]

        if len(Xm_exec) < 3:
            continue

        emg_order = np.argsort(ws_emg)
        Xm_exec   = Xm_exec[emg_order]

        # Slide a 3-window sequence over execution windows (4 sequences per trial)
        for i in range(len(Xm_exec) - 2):
            emg_raw  = np.concatenate(
                [Xm_exec[i], Xm_exec[i + 1], Xm_exec[i + 2]], axis=1
            )  # (11, 240)

            emg_feat = compute_emg_features(emg_raw[None, ...])[0]  # (44,)

            X_emg_raw_list.append(emg_raw)
            X_emg_feat_list.append(emg_feat)
            X_eeg_raw_list.append(eeg_raw)   # same EEG context for all EMG sequences
            y_list.append(label)
            s_list.append(sid)

    return (
        np.array(X_emg_raw_list,  dtype=np.float32),
        np.array(X_emg_feat_list, dtype=np.float32),
        np.array(X_eeg_raw_list,  dtype=np.float32),
        np.array(y_list,          dtype=np.int64),
        np.array(s_list,          dtype=np.int64),
    )


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


def make_loaders(X_emg_raw, X_emg_feat, X_eeg_raw, y, s):
    train_s, val_s, test_s = subject_split(s)

    def get_loader(subset, shuffle):
        mask = np.array([sid in subset for sid in s])
        ds   = FusionDataset(
            X_emg_raw[mask], X_emg_feat[mask], X_eeg_raw[mask], y[mask]
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        get_loader(train_s, True),
        get_loader(val_s,   False),
        get_loader(test_s,  False),
        train_s, val_s, test_s,
    )


# MODEL
# ----------------------------------------------

class AttentionPool(nn.Module):
    """Soft attention over the LSTM timestep dimension."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)


class EMGRawBranch(nn.Module):
    """
    CNN + BiLSTM branch for raw EMG signal.
    Architecture mirrors the standalone EMG CNN-LSTM model.

    Input:  (B, 11, 240)
    Output: (B, 128)
    """
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(N_EMG_CHANNELS, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),        # 240 -> 120

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),        # 120 -> 60

            nn.Dropout(DROPOUT),
        )

        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64,
            batch_first=True, bidirectional=True,
        )

        self.attn = AttentionPool(128)

    def forward(self, x):
        x = self.cnn(x)           # (B, 64, 60)
        x = x.transpose(1, 2)     # (B, 60, 64)
        x, _ = self.lstm(x)       # (B, 60, 128)
        return self.attn(x)       # (B, 128)


class EMGFeatBranch(nn.Module):
    """
    MLP branch for handcrafted EMG features.

    Input:  (B, 44)
    Output: (B, 32)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.mlp(x)


class EEGAuxBranch(nn.Module):
    """
    Lightweight EEG onset-context branch.
    Kept small intentionally — EEG should not overpower EMG.
    Uses EEGNet-style spatial + temporal convolutions.

    Input:  (B, 15, 240)
    Output: (B, 32)
    """
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # Temporal filter across time
            nn.Conv2d(1, 16, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),

            # Spatial filter across all 15 EEG channels
            nn.Conv2d(16, 16, kernel_size=(N_EEG_CHANNELS, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),   # 240 -> 120
            nn.Dropout(DROPOUT),

            # Second temporal block
            nn.Conv2d(16, 24, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),   # 120 -> 60
            nn.Dropout(DROPOUT),
        )

        # Project flattened features to 32-dim embedding
        self.proj = nn.Sequential(
            nn.Linear(24 * 60, 32),
            nn.ELU(),
            nn.Dropout(DROPOUT),
        )

    class EEGAuxBranch(nn.Module):
        """
        Lightweight EEG onset-context branch with temporal attention.

        Input:  (B, 15, 240)
        Output: (B, 32)
        """

        def __init__(self):
            super().__init__()

            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 7), padding=(0, 3), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),

                nn.Conv2d(16, 16, kernel_size=(N_EEG_CHANNELS, 1), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 2)),  # 240 -> 120
                nn.Dropout(DROPOUT),

                nn.Conv2d(16, 24, kernel_size=(1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(24),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 2)),  # 120 -> 60
                nn.Dropout(DROPOUT),
            )

            # attention over EEG time steps
            self.attn = AttentionPool(24)

            self.proj = nn.Sequential(
                nn.Linear(24, 32),
                nn.ELU(),
                nn.Dropout(DROPOUT),
            )

        def forward(self, x):
            x = x.unsqueeze(1)  # (B, 1, 15, 240)
            x = self.cnn(x)  # (B, 24, 1, 60)
            x = x.squeeze(2)  # (B, 24, 60)
            x = x.transpose(1, 2)  # (B, 60, 24)
            x = self.attn(x)  # (B, 24)
            return self.proj(x)  # (B, 32)

    def forward(self, x):
        x = x.unsqueeze(1)    # (B, 1, 15, 240)
        x = self.cnn(x)       # (B, 24, 1, 60)
        x = x.flatten(1)      # (B, 24*60 = 1440)
        return self.proj(x)   # (B, 32)


class EMGAnchoredFusion(nn.Module):
    """
    EMG-anchored gated fusion model.

    Forward pass:
      1. EMG raw branch + feature branch -> EMG logits (primary)
      2. EEG auxiliary branch -> EEG logits (correction)
      3. Gate (sigmoid) computed from both modalities
      4. Output = EMG_logits + gate * EEG_logits

    If EEG carries no information, the gate learns to stay near 0
    and the model degrades gracefully to pure EMG classification.
    """
    def __init__(self, emg_feat_dim: int):
        super().__init__()

        self.emg_raw  = EMGRawBranch()
        self.emg_feat = EMGFeatBranch(emg_feat_dim)

        # Primary classifier: EMG only
        self.emg_head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

        self.eeg_aux  = EEGAuxBranch()

        # Small EEG correction head
        self.eeg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, N_CLASSES),
        )

        # Gate: scalar in [0,1] controlling EEG contribution
        # Input: EMG features (160-dim) + EEG embedding (32-dim)
        self.gate = nn.Sequential(
            nn.Linear(128 + 32 + 32, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Initialise gate bias so EEG starts near-zero influence
        # Sigmoid(-1.5) ≈ 0.18 — EEG begins mostly muted
        nn.init.constant_(self.gate[-2].bias, -1.5)

    def forward(self, x_emg_raw, x_emg_feat, x_eeg_raw):
        z_emg_raw  = self.emg_raw(x_emg_raw)        # (B, 128)
        z_emg_feat = self.emg_feat(x_emg_feat)      # (B, 32)
        z_emg      = torch.cat([z_emg_raw, z_emg_feat], dim=1)

        emg_logits = self.emg_head(z_emg)           # (B, 3) — primary prediction

        z_eeg      = self.eeg_aux(x_eeg_raw)        # (B, 32)
        eeg_logits = self.eeg_head(z_eeg)           # (B, 3) — correction signal

        # Gate combines EMG and EEG features to decide EEG contribution
        g = self.gate(torch.cat([z_emg, z_eeg], dim=1))  # (B, 1)

        # Residual fusion: EMG dominates, EEG adds a small learned correction
        return emg_logits + g * eeg_logits


# TRAIN
# ----------------------------------------------

def get_class_weights(y_train: np.ndarray, device):
    """Square-root inverse-frequency weights to soften class imbalance correction."""
    counts  = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    weights = counts.sum() / (N_CLASSES * counts)
    weights = weights ** 0.5   # soften: full inverse would over-correct
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds, trues = [], []

    for xb_emg_raw, xb_emg_feat, xb_eeg_raw, yb in loader:
        xb_emg_raw  = xb_emg_raw.to(device)
        xb_emg_feat = xb_emg_feat.to(device)
        xb_eeg_raw  = xb_eeg_raw.to(device)
        yb          = yb.to(device)

        if is_train:
            # Mild EMG augmentation: Gaussian noise + per-channel amplitude jitter
            noise_std   = 0.01 * xb_emg_raw.std().detach()
            xb_emg_raw  = xb_emg_raw + noise_std * torch.randn_like(xb_emg_raw)

            scale      = torch.empty(xb_emg_raw.size(0), xb_emg_raw.size(1), 1,
                                     device=xb_emg_raw.device).uniform_(0.98, 1.02)
            xb_emg_raw = xb_emg_raw * scale

            optimizer.zero_grad()

        out  = model(xb_emg_raw, xb_emg_feat, xb_eeg_raw)
        loss = criterion(out, yb)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(yb)
        preds.extend(out.argmax(1).detach().cpu().numpy())
        trues.extend(yb.detach().cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    return mean_loss, acc, f1, trues, preds


def train():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    X_emg_raw, X_emg_feat, X_eeg_raw, y, s = create_fusion_samples()
    train_loader, val_loader, test_loader, train_s, val_s, test_s = make_loaders(
        X_emg_raw, X_emg_feat, X_eeg_raw, y, s
    )

    model    = EMGAnchoredFusion(emg_feat_dim=X_emg_feat.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 65)
    print("  EEG-EMG Fusion — Subject-Independent NeBULA Classification")
    print("  EMG-anchored with learned EEG gate")
    print("-" * 65)
    print(f"  Device         : {device}")
    print(f"  EMG raw input  : {X_emg_raw.shape}")
    print(f"  EMG feat input : {X_emg_feat.shape}")
    print(f"  EEG raw input  : {X_eeg_raw.shape}")
    print(f"  EMG windows    : {sorted(list(EMG_KEEP_WINDOW_STARTS))}")
    print(f"  EEG windows    : {EEG_CONTEXT_STARTS}")
    print(f"  Train subjects : {sorted([int(x) for x in train_s])}")
    print(f"  Val subjects   : {sorted([int(x) for x in val_s])}")
    print(f"  Test subjects  : {sorted([int(x) for x in test_s])}")
    print(f"  Parameters     : {n_params:,}")
    print()

    def split_stats(loader, name):
        ys = loader.dataset.y.numpy()
        print(f"  {name:5s}: {len(ys):5d} samples  "
              f"(Task0={(ys==0).sum()}, "
              f"Task1={(ys==1).sum()}, "
              f"Task2={(ys==2).sum()})")

    split_stats(train_loader, "train")
    split_stats(val_loader,   "val")
    split_stats(test_loader,  "test")
    print()

    y_train   = train_loader.dataset.y.numpy()
    weights   = get_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    best_state    = None
    best_epoch    = 0
    best_val_f1   = -1.0
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = []

    for epoch in range(1, EPOCHS + 1):
        tl, ta, tf, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, vf, _, _ = run_epoch(model, val_loader,   criterion, None,      device)

        scheduler.step(vf)

        history.append({
            "epoch": epoch,
            "train_loss": tl, "train_acc": ta, "train_f1": tf,
            "val_loss":   vl, "val_acc":   va, "val_f1":   vf,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"  Epoch {epoch:03d} | "
            f"train loss={tl:.4f} acc={ta:.3f} f1={tf:.3f} | "
            f"val loss={vl:.4f} acc={va:.3f} f1={vf:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        improved = (vf > best_val_f1) or (vf == best_val_f1 and vl < best_val_loss)

        if improved:
            best_val_f1   = vf
            best_val_loss = vl
            best_epoch    = epoch
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    model.load_state_dict(best_state)

    _, test_acc, test_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, None, device
    )

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print("\n" + "-" * 56)
    print("  TEST RESULTS")
    print(f"  Accuracy   : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  F1 macro   : {test_f1:.4f}")
    print(f"  Per class  : "
          f"Task1={report['0']['f1-score']:.3f}  "
          f"Task2={report['1']['f1-score']:.3f}  "
          f"Task3={report['2']['f1-score']:.3f}")
    print(f"  Precision  : "
          f"Task1={report['0']['precision']:.3f}  "
          f"Task2={report['1']['precision']:.3f}  "
          f"Task3={report['2']['precision']:.3f}")
    print(f"  Recall     : "
          f"Task1={report['0']['recall']:.3f}  "
          f"Task2={report['1']['recall']:.3f}  "
          f"Task3={report['2']['recall']:.3f}")
    print(f"  Best epoch : {best_epoch}")
    print("  Confusion matrix:")
    print(f"    {cm[0].tolist()}  ← Task 1 predicted as")
    print(f"    {cm[1].tolist()}  ← Task 2 predicted as")
    print(f"    {cm[2].tolist()}  ← Task 3 predicted as")
    print("-" * 65)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "fusion.pt"))
    np.save(os.path.join(RESULTS_DIR, "fusion_history.npy"), np.array(history, dtype=object))

    summary = {
        "model": "EEG_EMG_FUSION",
        "experiment": "subject_independent",
        "device": str(device),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "train_subjects": sorted([int(x) for x in train_s]),
        "val_subjects":   sorted([int(x) for x in val_s]),
        "test_subjects":  sorted([int(x) for x in test_s]),
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "emg_window_starts": sorted([int(x) for x in EMG_KEEP_WINDOW_STARTS]),
            "eeg_context_starts": EEG_CONTEXT_STARTS,
        },
    }

    with open(os.path.join(RESULTS_DIR, "fusion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Model   → {RESULTS_DIR}/fusion.pt")
    print(f"  History → {RESULTS_DIR}/fusion_history.npy")
    print(f"  Summary → {RESULTS_DIR}/fusion_summary.json")


if __name__ == "__main__":
    train()