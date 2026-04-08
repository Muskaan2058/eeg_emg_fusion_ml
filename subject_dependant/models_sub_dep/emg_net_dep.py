"""
NeBULA Dataset - EMGNet — Subject-Dependent EMG classifier
-------------------------------------------------------------
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



#  Model
# ----------------------------------------------

class EMGNet(nn.Module):
    """
    Three-block 1D CNN for multi-channel EMG.

    Input: (batch, 11, 80) — 11 muscles × 80 timepoints.

    Conv1d is natural for EMG: 11 muscles treated as parallel time series.
    No spatial/temporal separation needed unlike EEGNet.

    Block 1 (kernel=9, 45ms): fast activation onset/offset.
    Block 2 (kernel=7, 35ms): envelope shape and timing.
    Block 3 + AdaptiveAvgPool(5): compact summary. MPS-safe.
    """

    def __init__(self, n_channels=11, n_samples=80, n_classes=3, dropout=0.25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),        # 80 → 40
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),        # 40 → 20
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(5),            # 20 → 5  (MPS-safe)
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5, 64),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))



#  Data helpers
# ----------------------------------------------

def load_data(data_dir):
    """Load windowed EMG arrays from epoch.py output."""
    X = np.load(os.path.join(data_dir, 'X_emg_win.npy'))         # (N, 11, 80)
    y = np.load(os.path.join(data_dir, 'y_win.npy'))              # (N,)
    s = np.load(os.path.join(data_dir, 'subject_ids_win.npy'))    # (N,)
    y = y - 1   # 1/2/3 → 0/1/2
    return X, y, s


def make_loaders(X, y, batch_size=16):
    """Chronological 70/15/15 split."""
    n  = len(y)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    splits = {
        'train': (X[:t1],   y[:t1]),
        'val':   (X[t1:t2], y[t1:t2]),
        'test':  (X[t2:],   y[t2:]),
    }
    loaders = {}
    for name, (Xs, ys) in splits.items():
        # Conv1d expects (batch, channels, samples) — no unsqueeze needed
        Xt = torch.FloatTensor(Xs)
        yt = torch.LongTensor(ys)
        loaders[name] = DataLoader(TensorDataset(Xt, yt),
                                   batch_size=batch_size,
                                   shuffle=(name == 'train'))
    return loaders



#  Training
# ----------------------------------------------

def train_one_subject(X_s, y_s, cfg):
    loaders = make_loaders(X_s, y_s, batch_size=cfg['batch_size'])

    model     = EMGNet(n_channels=11, n_samples=80, n_classes=3,
                       dropout=cfg['dropout'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=cfg['epochs'], eta_min=1e-6)

    best_val_loss = float('inf')
    best_weights  = None
    no_improve    = 0

    for epoch in range(cfg['epochs']):
        model.train()
        for Xb, yb in loaders['train']:
            optimiser.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in loaders['val']:
                val_losses.append(criterion(model(Xb), yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= cfg['patience']:
                break

    model.load_state_dict(best_weights)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loaders['test']:
            preds.extend(model(Xb).argmax(1).tolist())
            trues.extend(yb.tolist())

    return {
        'accuracy':      accuracy_score(trues, preds),
        'f1_macro':      f1_score(trues, preds, average='macro'),
        'confusion':     confusion_matrix(trues, preds).tolist(),
        'n_train':       len(loaders['train'].dataset),
        'n_test':        len(loaders['test'].dataset),
        'stopped_epoch': cfg['epochs'] - no_improve,
    }



#  Main
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     default='../../preprocessed')
    parser.add_argument('--results',  default='../results_dep')
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--epochs',   type=int,   default=200)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--dropout',  type=float, default=0.25)
    parser.add_argument('--batch',    type=int,   default=16)
    parser.add_argument('--patience', type=int,   default=40)
    args = parser.parse_args()

    # Save into results/emgnet_sub_dep/
    out_dir = os.path.join(args.results, 'emgnet_sub_dep')
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(epochs=args.epochs, lr=args.lr, dropout=args.dropout,
               batch_size=args.batch, patience=args.patience)

    print("-" * 60)
    print("  EMGNet — Subject-Dependent NeBULA Classification")
    print("-" * 60)

    X, y, sids = load_data(args.data)
    subjects   = args.subjects if args.subjects else sorted(np.unique(sids).tolist())
    print(f"  Subjects : {subjects}")
    print(f"  Config   : epochs={cfg['epochs']}, lr={cfg['lr']}, "
          f"dropout={cfg['dropout']}, patience={cfg['patience']}")
    print()

    all_results = []

    for subj in subjects:
        mask = sids == subj
        X_s, y_s = X[mask], y[mask]

        if len(np.unique(y_s)) < 3:
            print(f"  sub-{subj:02d}  SKIP — fewer than 3 classes")
            continue

        metrics = train_one_subject(X_s, y_s, cfg)
        all_results.append({'subject': int(subj), **metrics})

        print(f"  sub-{subj:02d}  "
              f"acc={metrics['accuracy']:.3f}  "
              f"f1={metrics['f1_macro']:.3f}  "
              f"(train={metrics['n_train']}win, test={metrics['n_test']}win, "
              f"stopped@ep{metrics['stopped_epoch']})")

    if all_results:
        accs = [r['accuracy'] for r in all_results]
        f1s  = [r['f1_macro']  for r in all_results]
        print()
        print("-" * 60)
        print(f"  Mean accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Mean F1       : {np.mean(f1s):.3f}  ± {np.std(f1s):.3f}")
        print(f"  Best subject  : sub-{all_results[int(np.argmax(accs))]['subject']:02d}  "
              f"({max(accs):.3f})")
        print(f"  Worst subject : sub-{all_results[int(np.argmin(accs))]['subject']:02d}  "
              f"({min(accs):.3f})")
        print("-" * 60)

        out_path = os.path.join(out_dir, 'emgnet_sub_dep_results.npy')
        np.save(out_path, all_results)
        print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()