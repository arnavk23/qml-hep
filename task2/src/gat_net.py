"""
Graph Attention Network (GAT) for Quark/Gluon Jet Classification
=================================================================
Architecture based on:
    Veličković et al. (2018) "Graph Attention Networks"
    https://arxiv.org/abs/1710.10903

Design philosophy: attention over a fixed angular graph
-------------------------------------------------------
In contrast to the DGCNN, which dynamically rewires the graph in learned
feature space, this model fixes the graph at the structure built from the
physical (Δη, Δφ) angular space and instead learns to weight the contribution
of each neighbour through a trainable attention mechanism.

This is physically motivated for jet classification:

  - The angular graph encodes well-understood QCD geometry: collinear
    splittings create clusters of nearby particles, while soft wide-angle
    radiation is structurally different from hard collinear fragments.
    Preserving this geometry throughout every layer ensures that the model
    operates on physically interpretable neighbourhoods.

  - Not all particles within an angular neighbourhood are equally relevant to
    the substructure signal. Hard collinear daughters of a splitting carry
    most of the discriminating information, while soft underlying-event
    particles add noise. Attention coefficients let the model learn this
    weighting directly from data, without requiring hand-crafted importance
    scores.

  - Multi-head attention spreads representational capacity across independent
    attention subspaces, capturing different aspects of inter-particle
    relationships simultaneously. One head may learn to focus on hard
    splittings, another on soft radiation patterns, etc. Averaging over heads
    at the output layer reduces variance without sacrificing expressiveness.

Attention mechanism
-------------------
For each directed edge (i → j) with node embeddings h_i and h_j, the
attention coefficient is computed as:

    e_ij  = LeakyReLU( a^T [W h_i ∥ W h_j] )
    α_ij  = softmax_j( e_ij )

The updated representation for node i is then:

    h_i'  = σ( Σ_{j ∈ N(i)} α_ij W h_j )

With K parallel attention heads, the outputs are either concatenated
(intermediate layers) or averaged (final layer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


# ---------------------------------------------------------------------------
# GAT-based jet classifier
# ---------------------------------------------------------------------------

class GATJetClassifier(nn.Module):
    """
    Four-layer Graph Attention Network with residual connections, multi-head
    attention, and a two-layer classifier head.

    Layer dimensions (with heads=8, concat=True)
    ---------------------------------------------
        Input projection    :  in_channels → 64           (Linear + BN + ELU)
        GATConv 1           :  64  → 64 × 8  = 512        (8 heads, concat)
        GATConv 2           :  512 → 64 × 8  = 512        (8 heads, concat)
        GATConv 3           :  512 → 64 × 8  = 512        (8 heads, concat)
        GATConv 4           :  512 → 256                  (1 head, mean)
        Global pool         :  concat(mean, max) → 512
        FC 1                :  512 → 256
        FC 2                :  256 → 64
        Output              :  64  → 2  (logits)

    The ELU activation is used throughout instead of ReLU because it provides
    a smooth gradient for negative inputs, which can help convergence in deep
    networks with BatchNorm where pre-activation values may be negative.

    Parameters
    ----------
    in_channels : int
        Input node feature dimensionality (6 for this dataset).
    heads : int
        Number of parallel attention heads in GATConv layers 1–3.
    dropout : float
        Dropout probability applied to attention coefficients and classifier.
    """

    def __init__(self, in_channels: int = 6, heads: int = 8, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        hidden = 64

        # Project raw node features to a uniform embedding before attention
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
        )

        # Intermediate layers concatenate across heads, expanding the width
        self.gat1 = GATConv(hidden,          hidden, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden * heads,  hidden, heads=heads, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden * heads,  hidden, heads=heads, dropout=dropout, concat=True)

        # Final attention layer uses a single head and averages, producing a
        # fixed-width representation independent of the head count
        self.gat4 = GATConv(hidden * heads, 256, heads=1, dropout=dropout, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.bn3 = nn.BatchNorm1d(hidden * heads)
        self.bn4 = nn.BatchNorm1d(256)

        # Classifier receives the concatenated global mean and max pooled features
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)

        # Attention layers with residual connections where dimensionality permits.
        # The residual from the input projection is reused across layers 1–3
        # because they share the same hidden × heads width; this improves
        # gradient flow and mitigates over-smoothing in deep GAT stacks.
        x1 = F.elu(self.bn1(self.gat1(x, edge_index)))
        x2 = F.elu(self.bn2(self.gat2(x1, edge_index))) + x1
        x3 = F.elu(self.bn3(self.gat3(x2, edge_index))) + x2
        x4 = F.elu(self.bn4(self.gat4(x3, edge_index)))

        # Graph-level readout: global mean captures the average particle
        # representation while global max preserves the most prominent signal
        out = torch.cat([global_mean_pool(x4, batch),
                         global_max_pool(x4, batch)], dim=1)

        return self.classifier(out)
