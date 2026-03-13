"""
Dynamic Graph CNN (DGCNN) for Quark/Gluon Jet Classification
==============================================================
Architecture based on:
    Wang et al. (2019) "Dynamic Graph CNN for Learning on Point Clouds"
    https://arxiv.org/abs/1801.07829

Core concept: EdgeConv with dynamic graph recomputation
--------------------------------------------------------
Standard graph convolutions operate on a fixed graph that is constructed once
from the raw input coordinates. DGCNN departs from this by recomputing the
k-nearest-neighbour graph after each EdgeConv layer in the current feature
space. The motivation for this in jet physics is the following:

  - Layer 0: the graph is built in angular (Δη, Δφ) space, capturing
    geometrically adjacent particles.
  - Layer 1+: the graph is rebuilt in the learned feature space. Particles
    that are separated angularly may share learned representations (e.g.
    two hard-fragmentation products of the same splitting) and will be
    connected; conversely, particles that are close in angle but otherwise
    dissimilar may no longer be neighbours.

This progressive re-clustering allows the model to discover long-range
correlations in jet substructure that a fixed angular graph would miss.

EdgeConv operation
------------------
For each node i with k-nearest neighbours {j}: the edge feature between
i and j is the concatenation [h_i, h_j − h_i].  The term h_j − h_i
explicitly encodes the relative displacement in feature space, making each
node aware of its local context while remaining invariant to global
translations of the feature vectors.  A shared MLP is applied to each edge
feature and max-pooling aggregates across the neighbourhood:

    h_i' = max_{j ∈ N(i)} MLP([h_i, h_j − h_i])

Multi-scale readout
-------------------
The intermediate features from all three EdgeConv layers are concatenated
before global pooling.  This allows the classifier head to simultaneously
draw on fine-grained local structure (early layers, small k neighbourhoods)
and coarser global correlations (later layers in learned feature space).
Global mean and max pooling are both applied and concatenated, doubling the
representational capacity of the graph-level embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class SharedMLP(nn.Sequential):
    """
    A multi-layer perceptron with BatchNorm and ReLU activations applied
    uniformly to all edges (hence "shared" across edge instances).

    Parameters
    ----------
    channels : list of int
        Sequence of layer widths, e.g. [128, 64, 64].
    """
    def __init__(self, channels):
        layers = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers += [
                nn.Linear(in_c, out_c, bias=False),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
        super().__init__(*layers)


def _knn_graph(x: torch.Tensor, k: int, batch: torch.Tensor) -> torch.Tensor:
    """
    Compute a directed k-nearest-neighbour graph in the current feature space
    using pairwise Euclidean distances.

    This pure-PyTorch implementation processes each graph in the batch
    independently, ensuring that neighbours are never drawn from a different
    jet.  For the typical jet size (20–80 particles) the O(N²) distance matrix
    is inexpensive to compute on modern hardware.

    Parameters
    ----------
    x     : (N_total, F)  — concatenated node features for the whole batch
    k     : int           — number of nearest neighbours (excluding self)
    batch : (N_total,)    — batch assignment vector

    Returns
    -------
    edge_index : (2, E)  — COO edge index (source, destination)
    """
    device = x.device
    src_list, dst_list = [], []

    for bid in batch.unique():
        mask = batch == bid
        x_b = x[mask]
        n = x_b.size(0)
        k_b = min(k, n - 1)
        if k_b == 0:
            continue

        # Squared pairwise distances via the identity ‖a − b‖² = ‖a‖² + ‖b‖² − 2⟨a,b⟩
        dist = torch.cdist(x_b, x_b)
        _, nn_idx = dist.topk(k_b + 1, dim=1, largest=False)
        nn_idx = nn_idx[:, 1:]          # drop the self-loop (distance = 0)

        global_idx = mask.nonzero(as_tuple=True)[0]
        src = global_idx.repeat_interleave(k_b)
        dst = global_idx[nn_idx.reshape(-1)]
        src_list.append(src)
        dst_list.append(dst)

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    return torch.stack([torch.cat(src_list), torch.cat(dst_list)])


# ---------------------------------------------------------------------------
# DGCNN building block
# ---------------------------------------------------------------------------

class DynamicEdgeConv(nn.Module):
    """
    EdgeConv layer with in-forward graph recomputation.

    The k-NN graph is rebuilt at call time from the current node embeddings,
    making it "dynamic" across layers.  This is the distinguishing feature of
    DGCNN compared to architectures with a fixed input graph.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        self.k = k
        # EdgeConv receives the concatenated [h_i || h_j − h_i] edge feature
        self.conv = EdgeConv(
            nn=SharedMLP([2 * in_channels, out_channels, out_channels]),
            aggr='max',
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        edge_index = _knn_graph(x, self.k, batch)
        return self.conv(x, edge_index)


# ---------------------------------------------------------------------------
# Full DGCNN model
# ---------------------------------------------------------------------------

class DGCNN(nn.Module):
    """
    Three-layer Dynamic Graph CNN with multi-scale feature concatenation,
    dual global pooling, and a two-layer classifier head.

    Layer dimensions
    ----------------
        Input projection  :  in_channels → 64
        DynamicEdgeConv 1 :  64  → 64
        DynamicEdgeConv 2 :  64  → 128
        DynamicEdgeConv 3 :  128 → 256
        Global pool       :  concat(mean, max) over [64 + 128 + 256] = 896
        FC 1              :  896 → 512
        FC 2              :  512 → 256
        Output            :  256 → 2  (logits)

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features (6 for this dataset).
    k : int
        Number of nearest neighbours used in each EdgeConv layer.
    dropout : float
        Dropout probability applied in the classifier head.
    """

    def __init__(self, in_channels: int = 6, k: int = 16, dropout: float = 0.5):
        super().__init__()
        self.k = k

        # Initial linear projection to normalise the raw input feature scale
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.ec1 = DynamicEdgeConv(64,  64,  k)
        self.ec2 = DynamicEdgeConv(64,  128, k)
        self.ec3 = DynamicEdgeConv(128, 256, k)

        # (64 + 128 + 256) × 2 channels after mean and max global pooling
        pool_dim = (64 + 128 + 256) * 2

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        x = self.input_proj(x)

        h1 = self.ec1(x, batch)
        h2 = self.ec2(h1, batch)
        h3 = self.ec3(h2, batch)

        # Concatenate feature maps from all three scales before readout.
        # This preserves both fine-grained local structure (h1) and abstract
        # long-range correlations (h3) in the final graph representation.
        h = torch.cat([h1, h2, h3], dim=1)

        # Dual pooling: mean captures the average jet activity; max preserves
        # the dominant substructure signal regardless of multiplicity.
        out = torch.cat([global_mean_pool(h, batch),
                         global_max_pool(h, batch)], dim=1)

        return self.classifier(out)
