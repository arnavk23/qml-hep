"""
Quark/Gluon Jet Dataset
========================
Loads the ParticleNet quark/gluon jet dataset (Zenodo record 3164691) and
converts each jet from a zero-padded particle array into a graph-structured
PyTorch Geometric Data object suitable for GNN-based classification.

Graph construction rationale
------------------------------
A jet is a collimated spray of particles produced by a hard-scattered quark
or gluon. Representing it as a set of independent, unconnected particles
ignores the fact that particles produced nearby in (η, φ) space share a
common origin — either from the same QCD splitting or from soft wide-angle
radiation. Angular proximity in (Δη, Δφ) therefore serves as a physically
motivated prior for inter-particle correlations.

Construction steps applied to each jet:
  1. Remove zero-padded entries (particles with pT < 1e-6 GeV).
  2. Convert raw Cartesian 4-momenta (px, py, pz, E) to cylindrical
     coordinates (pT, η, φ, E) and compute jet-centred relative quantities
     (Δη, Δφ) using the pT-weighted centroid.
  3. Build a directed k = 16 nearest-neighbour graph in (Δη, Δφ) space.
     The choice k = 16 matches the original ParticleNet hyperparameter and
     provides enough neighbourhood context for multi-scale feature learning
     while avoiding over-smoothing in deeper GNN layers.
  4. Assemble a six-dimensional node feature vector per particle:
         [Δη, Δφ, log(pT), log(E), log(pT / ΣpT), log(E / ΣE)]
     Logarithmic scaling of energy quantities compresses the dynamic range
     (which can span several orders of magnitude) and empirically improves
     both gradient flow and final model accuracy.

Expected HDF5 file schema
--------------------------
  X : float32  (N_jets, N_particles, 4)   —  [px, py, pz, E] per particle
  y : int32    (N_jets,)                  —  0 → gluon,  1 → quark

Download:
  https://zenodo.org/records/3164691
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def _to_cylindrical(px, py, pz, E):
    """Convert Cartesian 4-momenta to (pT, pseudorapidity η, azimuth φ, E)."""
    pT = np.sqrt(px ** 2 + py ** 2)
    p_mag = np.sqrt(px ** 2 + py ** 2 + pz ** 2) + 1e-9
    eta = np.arctanh(np.clip(pz / p_mag, -1 + 1e-7, 1 - 1e-7))
    phi = np.arctan2(py, px)
    return pT, eta, phi, E


def _looks_like_particlenet_pt_eta_phi_pid(raw: np.ndarray) -> bool:
    """
    Heuristic detector for the common ParticleNet tensor schema:
        [pT, η, φ, pid]

    The 4th channel (`pid`) is typically integer-valued with magnitudes around
    standard PDG IDs (e.g. 22, 130, 211, 2212).  This makes it easy to
    distinguish from Cartesian energy, which is continuous.
    """
    ch3 = raw[:, 3]
    finite = np.isfinite(ch3)
    if finite.sum() == 0:
        return False
    vals = ch3[finite]
    frac_part = np.abs(vals - np.round(vals))
    near_integer_ratio = np.mean(frac_part < 1e-6)
    return near_integer_ratio > 0.90 and np.max(np.abs(vals)) > 100


def _extract_kinematics(raw: np.ndarray):
    """
    Return (pT, η, φ, E) from either supported schema:

    1) Cartesian 4-momenta: [px, py, pz, E]
    2) ParticleNet features: [pT, η, φ, pid]

    For schema (2), E is approximated using the massless relation
    E ≈ pT * cosh(η), which is sufficient for stable feature construction.
    """
    if _looks_like_particlenet_pt_eta_phi_pid(raw):
        pT = np.clip(raw[:, 0], 0.0, None)
        eta = raw[:, 1]
        phi = _wrap_phi(raw[:, 2])
        E = pT * np.cosh(np.clip(eta, -8, 8))
        return pT, eta, phi, E

    px, py, pz, E = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    return _to_cylindrical(px, py, pz, E)


def _wrap_phi(dphi):
    """Wrap an angular difference to the interval (−π, π]."""
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def _node_features(particles: np.ndarray) -> np.ndarray:
    """
    Compute six-dimensional node features from raw Cartesian 4-momenta.

    The jet centroid is defined as the pT-weighted mean direction in (η, φ)
    space, projected back through the unit circle to handle the φ wrap-around.
    Log energy quantities are rescaled relative to the jet-level totals so
    that the features are invariant to the absolute jet energy scale.

    Parameters
    ----------
    particles : ndarray of shape (N, 4), columns [px, py, pz, E]

    Returns
    -------
    ndarray of shape (N, 6), columns [Δη, Δφ, log pT, log E, log pT_rel, log E_rel]
    """
    pT, eta, phi, E = _extract_kinematics(particles)

    pT_sum = pT.sum() + 1e-9
    E_sum = E.sum() + 1e-9

    # pT-weighted centroid; use unit-circle mean to handle φ wrap-around
    eta_jet = np.sum(pT * eta) / pT_sum
    phi_jet = np.arctan2(
        np.sum(pT * np.sin(phi)) / pT_sum,
        np.sum(pT * np.cos(phi)) / pT_sum,
    )

    d_eta = eta - eta_jet
    d_phi = _wrap_phi(phi - phi_jet)
    log_pT = np.log(pT + 1e-9)
    log_E = np.log(E + 1e-9)
    log_pT_rel = np.log(pT / pT_sum + 1e-9)
    log_E_rel = np.log(E / E_sum + 1e-9)

    return np.stack([d_eta, d_phi, log_pT, log_E, log_pT_rel, log_E_rel],
                    axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class QGJetDataset(Dataset):
    """
    Converts raw jet data from the Zenodo 3164691 HDF5 files into a list of
    PyTorch Geometric Data objects, each representing a single jet as a
    k-NN graph in angular (Δη, Δφ) space.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file (train.h5 or test.h5).
    split : str
        'train', 'val', or 'test'. Train and val are carved from the same
        file; test should be called on the dedicated test file.
    val_fraction : float
        Fraction of the file's jets reserved for validation.
    max_jets : int, optional
        Cap on the number of jets loaded. Useful for rapid prototyping.
    k : int
        Number of nearest neighbours per node (default 16).
    seed : int
        Random seed for the train/val split.
    """

    N_NODE_FEATURES = 6

    def __init__(self, filepath, split='train', val_fraction=0.15,
                 max_jets=None, k=16, seed=42):
        super().__init__()
        self.k = k
        self.graphs = self._load(filepath, split, val_fraction, max_jets, seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, filepath, split, val_fraction, max_jets, seed):
        with h5py.File(filepath, 'r') as f:
            # Accept both 'X'/'y' and other common naming conventions
            x_key = 'X' if 'X' in f else next(k for k in f if k.lower().startswith('x'))
            y_key = 'y' if 'y' in f else next(k for k in f if k.lower().startswith('y'))
            X = f[x_key][:]
            y = f[y_key][:].astype(np.int64)

        if max_jets is not None:
            X, y = X[:max_jets], y[:max_jets]

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_val = int(len(y) * val_fraction)

        if split == 'val':
            idx = idx[:n_val]
        elif split == 'train':
            idx = idx[n_val:]
        # 'test': use all indices in the dedicated test file

        return [self._to_graph(X[i], int(y[i])) for i in idx]

    def _to_graph(self, raw: np.ndarray, label: int) -> Data:
        """
        Convert a single jet's zero-padded particle array to a PyG Data object.

        Only particles with pT > 1e-6 GeV are retained. For the rare edge case
        where fewer than two valid particles remain, a minimal graph with no
        edges is returned so the DataLoader batch collation remains consistent.
        """
        pT, _, _, _ = _extract_kinematics(raw)
        valid = pT > 1e-6
        particles = raw[valid]

        y_tensor = torch.tensor([label], dtype=torch.long)

        if len(particles) < 2:
            x = torch.zeros((max(1, len(particles)), self.N_NODE_FEATURES),
                            dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=y_tensor)

        # Construct six-dimensional node feature matrix
        node_feats = _node_features(particles)

        # k-NN graph in (Δη, Δφ) space — physically motivated spatial graph
        coords = node_feats[:, :2]
        k = min(self.k, len(particles) - 1)

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
        _, indices = nbrs.kneighbors(coords)

        n = len(particles)
        src = np.repeat(np.arange(n), k)
        dst = indices[:, 1:].flatten()      # column 0 is the self-neighbour

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        x = torch.tensor(node_feats, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, y=y_tensor)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
