"""
Training and Evaluation Script
================================
Trains and evaluates the DGCNN and GAT architectures on the quark/gluon jet
classification task and produces a side-by-side performance comparison.

Usage
-----
    # Train a single model
    python src/train.py --model dgcnn --train data/train.h5 --test data/test.h5
    python src/train.py --model gat   --train data/train.h5 --test data/test.h5

    # Train both models and generate comparison plots
    python src/train.py --compare --train data/train.h5 --test data/test.h5

    # Quick smoke test on a small subset
    python src/train.py --compare --train data/train.h5 --test data/test.h5 \
        --max-jets 5000 --epochs 5
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import QGJetDataset
from dgcnn import DGCNN
from gat_net import GATJetClassifier


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, in_channels: int, device: torch.device) -> nn.Module:
    """Instantiate a model by name and move it to the target device."""
    if name == 'dgcnn':
        return DGCNN(in_channels=in_channels, k=16, dropout=0.5).to(device)
    elif name == 'gat':
        return GATJetClassifier(in_channels=in_channels, heads=8, dropout=0.4).to(device)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: dgcnn, gat")


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, criterion, device):
    """Run a single training epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        logits = model(batch)
        # y is stored as (N, 1); squeeze to (N,) for cross-entropy
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()
        # Gradient clipping avoids occasional divergence from sparse attention
        # gradients in the GAT when learning rates are not yet warmed up
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate a model on a DataLoader and return a results dictionary.

    Returns
    -------
    dict with keys: loss, accuracy, auc, y_true, y_score
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_true, all_score = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.squeeze()
        loss = criterion(logits, y)
        total_loss += loss.item() * batch.num_graphs

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_true.extend(y.cpu().numpy())
        all_score.extend(probs)

    y_true = np.array(all_true)
    y_score = np.array(all_score)
    y_pred = (y_score >= 0.5).astype(int)

    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_score),
        'y_true': y_true,
        'y_score': y_score,
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(model, name, train_loader, val_loader, args, device):
    """
    Train a model with cosine-annealing learning rate schedule and early
    stopping on validation AUC.

    Returns the best validation AUC achieved and the model state at that point.
    """
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    print(f"\n{'=' * 60}")
    print(f"  Training {name.upper()}")
    print(f"{'=' * 60}")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Device     : {device}")
    print()

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser,
                                     criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs}  |  "
                  f"train loss: {train_loss:.4f}  |  "
                  f"val loss: {val_metrics['loss']:.4f}  |  "
                  f"val AUC: {val_metrics['auc']:.4f}  |  "
                  f"{elapsed:.0f}s elapsed")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            break

    print(f"\n  Best validation AUC: {best_auc:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = os.path.join(args.models_dir, f"{name}_best.pt")
    torch.save({'model_state': best_state, 'args': vars(args)}, ckpt_path)
    print(f"  Checkpoint saved -> {ckpt_path}")

    return best_auc, history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(histories: dict, output_path: str = "training_history.png"):
    """Plot training loss and validation AUC curves for all models."""
    n_models = len(histories)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colours = {'dgcnn': 'steelblue', 'gat': 'tomato'}

    for name, hist in histories.items():
        colour = colours.get(name, 'grey')
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], label=f"{name.upper()} train",
                     color=colour, linestyle='--', alpha=0.7)
        axes[0].plot(epochs, hist['val_loss'], label=f"{name.upper()} val",
                     color=colour)
        axes[1].plot(epochs, hist['val_auc'], label=name.upper(), color=colour)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Validation ROC AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTraining history plot saved -> {output_path}")


def plot_roc_curves(results: dict, output_path: str = "roc_comparison.png"):
    """Plot ROC curves for all models on the test set."""
    fig, ax = plt.subplots(figsize=(7, 6))

    colours = {'dgcnn': 'steelblue', 'gat': 'tomato'}

    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_score'])
        auc = metrics['auc']
        colour = colours.get(name, 'grey')
        ax.plot(fpr, tpr, label=f"{name.upper()}  (AUC = {auc:.4f})",
                color=colour, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    ax.set_xlabel("False Positive Rate (gluon misidentification)")
    ax.set_ylabel("True Positive Rate (quark efficiency)")
    ax.set_title("ROC Curve — Quark vs. Gluon Jet Classification")
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC comparison plot saved  -> {output_path}")


def print_results_table(results: dict):
    """Print a formatted summary table of test-set metrics."""
    print(f"\n{'=' * 55}")
    print(f"  Test Set Results")
    print(f"{'=' * 55}")
    print(f"  {'Model':<10}  {'Accuracy':>10}  {'ROC AUC':>10}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for name, metrics in results.items():
        print(f"  {name.upper():<10}  {metrics['accuracy']:>10.4f}  "
              f"{metrics['auc']:>10.4f}")
    print(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GNN models for quark/gluon jet classification"
    )
    parser.add_argument('--model', choices=['dgcnn', 'gat'], default='dgcnn',
                        help="Model to train (ignored when --compare is set)")
    parser.add_argument('--compare', action='store_true',
                        help="Train both models and produce comparison output")
    parser.add_argument('--train', default='data/train.h5',
                        help="Path to training HDF5 file")
    parser.add_argument('--test', default='data/test.h5',
                        help="Path to test HDF5 file")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument('--k', type=int, default=16,
                        help="k-NN connectivity for graph construction")
    parser.add_argument('--max-jets', type=int, default=None,
                        help="Limit dataset size for debugging")
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--num-workers', type=int, default=0,
                        help="DataLoader worker processes")
    parser.add_argument('--models-dir', type=str, default='models',
                        help="Directory where model checkpoints are saved")
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help="Directory where figure outputs are saved")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\nBuilding training and validation graphs…")
    t0 = time.time()
    train_ds = QGJetDataset(args.train, split='train',
                            val_fraction=args.val_fraction,
                            max_jets=args.max_jets, k=args.k)
    val_ds = QGJetDataset(args.train, split='val',
                          val_fraction=args.val_fraction,
                          max_jets=args.max_jets, k=args.k)
    print(f"  Train: {len(train_ds):,} jets  |  Val: {len(val_ds):,} jets  "
          f"({time.time() - t0:.1f}s)")

    print("Building test graphs…")
    t0 = time.time()
    test_ds = QGJetDataset(args.test, split='test',
                           max_jets=args.max_jets, k=args.k)
    print(f"  Test : {len(test_ds):,} jets  ({time.time() - t0:.1f}s)")

    # Node feature dimensionality is inferred from the first graph
    in_channels = train_ds[0].x.size(1)

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    # ------------------------------------------------------------------
    # Determine which models to run
    # ------------------------------------------------------------------
    model_names = ['dgcnn', 'gat'] if args.compare else [args.model]

    histories = {}
    test_results = {}

    for name in model_names:
        model = build_model(name, in_channels, device)
        _, history = train_model(model, name, train_loader, val_loader, args, device)
        histories[name] = history

        print(f"\nEvaluating {name.upper()} on test set…")
        metrics = evaluate(model, test_loader, device)
        test_results[name] = metrics
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  ROC AUC  : {metrics['auc']:.4f}")

    # ------------------------------------------------------------------
    # Summary and plots
    # ------------------------------------------------------------------
    if len(model_names) > 1:
        print_results_table(test_results)
        plot_roc_curves(test_results, os.path.join(args.figures_dir, "roc_comparison.png"))
    else:
        name = model_names[0]
        fpr, tpr, _ = roc_curve(test_results[name]['y_true'],
                                 test_results[name]['y_score'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='steelblue',
                label=f"AUC = {test_results[name]['auc']:.4f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {name.upper()}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        roc_path = os.path.join(args.figures_dir, f"roc_{name}.png")
        fig.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"ROC curve saved -> {roc_path}")

    plot_training_history(histories, os.path.join(args.figures_dir, "training_history.png"))


if __name__ == "__main__":
    main()
