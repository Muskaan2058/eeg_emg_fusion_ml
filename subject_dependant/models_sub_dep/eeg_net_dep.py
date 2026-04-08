"""
NeBULA Dataset - EEGNet — Subject-Dependent EEG classifier
-------------------------------------------------------------
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet  adapted for 80-sample windows.

    Block 1 — temporal + depthwise spatial:
      Conv2d (1×32): learns WHEN patterns occur (160ms kernel).
      Conv2d (15×1) depthwise: learns WHICH channels co-activate.
      AvgPool (1,4): 80 → 20.

    Block 2 — separable conv:
      Depthwise (1×16) + pointwise (1×1).
      AvgPool (1,8): 20 → 2.

    Classifier: Flatten → Linear(3).
    """

    def __init__(self, n_channels=15, n_samples=80, n_classes=3,
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),   # 80 → 20
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),   # 20 → 2
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy     = torch.zeros(1, 1, n_channels, n_samples)
            flat_size = self.block2(self.block1(dummy)).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))



#  Data helpers
# ----------------------------------------------

def load_data(data_dir):
    """Load windowed EEG arrays from epoch.py output."""
    X = np.load(os.path.join(data_dir, 'X_eeg_win.npy'))         # (N, 15, 80)
    y = np.load(os.path.join(data_dir, 'y_win.npy'))              # (N,)
    s = np.load(os.path.join(data_dir, 'subject_ids_win.npy'))    # (N,)
    y = y - 1   # 1/2/3 → 0/1/2
    return X, y, s


def make_loaders(X, y, batch_size=16):
    """
    Chronological 70/15/15 split.
    NOT random — EEG patterns drift across a session so we must
    keep temporal order: train on early trials, test on later ones.
    """
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
        # unsqueeze(1): (N,15,80) → (N,1,15,80) — Conv2d needs channel dim
        Xt = torch.FloatTensor(Xs).unsqueeze(1)
        yt = torch.LongTensor(ys)
        loaders[name] = DataLoader(TensorDataset(Xt, yt),
                                   batch_size=batch_size,
                                   shuffle=(name == 'train'))
    return loaders



#  Training
# ----------------------------------------------

def train_one_subject(X_s, y_s, cfg):
    loaders = make_loaders(X_s, y_s, batch_size=cfg['batch_size'])

    model     = EEGNet(n_channels=15, n_samples=80, n_classes=3,
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
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
    parser.add_argument('--dropout',  type=float, default=0.5)
    parser.add_argument('--batch',    type=int,   default=16)
    parser.add_argument('--patience', type=int,   default=40)
    args = parser.parse_args()

    # Save into results/eegnet_sub_dep/
    out_dir = os.path.join(args.results, 'eegnet_sub_dep')
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(epochs=args.epochs, lr=args.lr, dropout=args.dropout,
               batch_size=args.batch, patience=args.patience)

    print("-" * 60)
    print("  EEGNet — Subject-Dependent NeBULA Classification")
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

        out_path = os.path.join(out_dir, 'eegnet_sub_dep_results.npy')
        np.save(out_path, all_results)
        print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()