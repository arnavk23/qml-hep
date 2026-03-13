"""
Task VI: Quantum Representation Learning with Contrastive Loss

Implements:
1) MNIST loading
2) Trainable quantum state preparation from an image
3) Two-image SWAP test circuit producing fidelity
4) Contrastive training: maximize fidelity for same class, minimize otherwise
"""

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------------
# MNIST pair dataset
# -------------------------------
class MNISTPairDataset(Dataset):
    """Samples random pairs of MNIST images and emits (img_a, img_b, same_label)."""

    def __init__(self, root: str, train: bool, max_samples: int | None = None):
        transform = transforms.ToTensor()
        base = datasets.MNIST(root=root, train=train, download=True, transform=transform)

        if max_samples is not None:
            indices = list(range(min(max_samples, len(base))))
            self.images = base.data[indices].float() / 255.0
            self.labels = base.targets[indices]
        else:
            self.images = base.data.float() / 255.0
            self.labels = base.targets

        self.class_to_indices = {}
        for idx, label in enumerate(self.labels.tolist()):
            self.class_to_indices.setdefault(label, []).append(idx)
        self.available_classes = sorted(self.class_to_indices.keys())

        self.num_items = len(self.images)

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        label_a = int(self.labels[idx].item())
        img_a = self.images[idx]

        same = bool(random.getrandbits(1))
        if same:
            idx_b = random.choice(self.class_to_indices[label_a])
            label_target = 1.0
        else:
            other_candidates = [c for c in self.available_classes if c != label_a]
            if not other_candidates:
                idx_b = random.choice(self.class_to_indices[label_a])
                label_target = 1.0
                img_b = self.images[idx_b]
                return img_a.unsqueeze(0), img_b.unsqueeze(0), torch.tensor(label_target, dtype=torch.float32)

            other_label = random.choice(other_candidates)
            idx_b = random.choice(self.class_to_indices[other_label])
            label_target = 0.0

        img_b = self.images[idx_b]
        return img_a.unsqueeze(0), img_b.unsqueeze(0), torch.tensor(label_target, dtype=torch.float32)


# -------------------------------
# Quantum representation model
# -------------------------------
@dataclass
class QuantumConfig:
    n_qubits: int = 4
    n_layers: int = 2


class QuantumContrastiveModel(nn.Module):
    """
    Shared trainable state-preparation + SWAP-test fidelity model.

    For each image:
      - Downsample to (2^n_qubits) features
      - Normalize to unit vector
      - Amplitude embedding + trainable variational block

    SWAP test outputs fidelity estimate F in [0,1].
    """

    def __init__(self, config: QuantumConfig):
        super().__init__()
        self.config = config
        self.n_features = 2 ** config.n_qubits

        self.pre_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_features),
        )

        total_wires = 2 * config.n_qubits + 1
        self.ancilla = 0
        self.wires_a = list(range(1, 1 + config.n_qubits))
        self.wires_b = list(range(1 + config.n_qubits, 1 + 2 * config.n_qubits))

        self.dev = qml.device("default.qubit", wires=total_wires)

        init = 0.05 * torch.randn(config.n_layers, config.n_qubits, 3)
        self.theta = nn.Parameter(init)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def swap_test_qnode(vec_a, vec_b, theta):
            self._prepare_quantum_state(vec_a, theta, self.wires_a)
            self._prepare_quantum_state(vec_b, theta, self.wires_b)

            qml.Hadamard(wires=self.ancilla)
            for wa, wb in zip(self.wires_a, self.wires_b):
                qml.CSWAP(wires=[self.ancilla, wa, wb])
            qml.Hadamard(wires=self.ancilla)

            return qml.probs(wires=self.ancilla)

        self.qnode = swap_test_qnode

    def _prepare_quantum_state(self, vec, theta, wires):
        qml.AmplitudeEmbedding(features=vec, wires=wires, normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights=theta, wires=wires)

    def _encode_image(self, x):
        vec = self.pre_net(x)
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        vec = torch.tanh(vec)
        vec = F.normalize(vec, p=2, dim=-1, eps=1e-8)
        return vec

    def forward(self, img_a, img_b):
        va = self._encode_image(img_a)
        vb = self._encode_image(img_b)
        theta_safe = torch.nan_to_num(self.theta, nan=0.0, posinf=1.0, neginf=-1.0)

        fidelities = []
        for i in range(va.shape[0]):
            probs = self.qnode(va[i], vb[i], theta_safe)
            p0 = probs[0]
            fidelity = torch.clamp(2.0 * p0 - 1.0, 0.0, 1.0)
            fidelities.append(fidelity)

        return torch.stack(fidelities)


# -------------------------------
# Contrastive loss
# -------------------------------
def contrastive_fidelity_loss(fidelity, same_label):
    """
    same_label=1 -> maximize fidelity; same_label=0 -> minimize fidelity.
    """
    return torch.mean(same_label * (1.0 - fidelity) ** 2 + (1.0 - same_label) * fidelity ** 2)


# -------------------------------
# Train / evaluate
# -------------------------------
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_same = 0.0
    total_diff = 0.0
    n_same = 0
    n_diff = 0

    for img_a, img_b, same_label in loader:
        if is_train:
            optimizer.zero_grad()

        fidelity = model(img_a, img_b)
        loss = contrastive_fidelity_loss(fidelity, same_label)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                for param in model.parameters():
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)

        total_loss += loss.item() * img_a.size(0)

        same_mask = same_label > 0.5
        diff_mask = ~same_mask
        if same_mask.any():
            total_same += fidelity[same_mask].mean().item()
            n_same += 1
        if diff_mask.any():
            total_diff += fidelity[diff_mask].mean().item()
            n_diff += 1

    mean_loss = total_loss / len(loader.dataset)
    mean_same_f = total_same / max(n_same, 1)
    mean_diff_f = total_diff / max(n_diff, 1)
    return mean_loss, mean_same_f, mean_diff_f


def save_curves(train_hist, val_hist, out_path):
    epochs = range(1, len(train_hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, train_hist["loss"], label="Train loss")
    ax[0].plot(epochs, val_hist["loss"], label="Val loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Contrastive loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(epochs, train_hist["same_f"], label="Train same-class fidelity")
    ax[1].plot(epochs, train_hist["diff_f"], label="Train diff-class fidelity")
    ax[1].plot(epochs, val_hist["same_f"], "--", label="Val same-class fidelity")
    ax[1].plot(epochs, val_hist["diff_f"], "--", label="Val diff-class fidelity")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Fidelity")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quantum contrastive representation learning on MNIST")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-val", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = MNISTPairDataset(root=args.data_dir, train=True, max_samples=args.max_train)
    val_ds = MNISTPairDataset(root=args.data_dir, train=False, max_samples=args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = QuantumContrastiveModel(QuantumConfig(n_qubits=args.n_qubits, n_layers=args.n_layers))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_hist = {"loss": [], "same_f": [], "diff_f": []}
    val_hist = {"loss": [], "same_f": [], "diff_f": []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_same, tr_diff = run_epoch(model, train_loader, optimizer)
        va_loss, va_same, va_diff = run_epoch(model, val_loader, optimizer=None)

        train_hist["loss"].append(tr_loss)
        train_hist["same_f"].append(tr_same)
        train_hist["diff_f"].append(tr_diff)

        val_hist["loss"].append(va_loss)
        val_hist["same_f"].append(va_same)
        val_hist["diff_f"].append(va_diff)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={tr_loss:.4f}, sameF={tr_same:.4f}, diffF={tr_diff:.4f} | "
            f"val loss={va_loss:.4f}, sameF={va_same:.4f}, diffF={va_diff:.4f}"
        )

    curves_path = os.path.join(args.out_dir, "training_curves.png")
    save_curves(train_hist, val_hist, curves_path)

    ckpt_path = os.path.join(args.out_dir, "quantum_contrastive_mnist.pt")
    torch.save(
        {
            "theta": model.theta.detach().cpu(),
            "pre_net": model.pre_net.state_dict(),
            "config": vars(args),
            "train_hist": train_hist,
            "val_hist": val_hist,
        },
        ckpt_path,
    )

    print("\nSaved:")
    print(f"- curves: {curves_path}")
    print(f"- checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
