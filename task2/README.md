# Task II: Classical Graph Neural Network — Quark/Gluon Jet Classification

This directory implements two graph neural network architectures for classifying jets originating from quarks versus gluons, trained on the ParticleNet dataset from Zenodo record [3164691](https://zenodo.org/records/3164691).

## Contents

| File | Description |
|------|-------------|
| `src/dataset.py` | Data loading, preprocessing, and graph construction |
| `src/dgcnn.py` | Architecture 1: Dynamic Graph CNN (DGCNN) with EdgeConv |
| `src/gat_net.py` | Architecture 2: Graph Attention Network (GAT) |
| `src/train.py` | Unified training, evaluation, and comparison script |
| `requirements.txt` | Python dependencies |

Current folder layout:

```text
task2/
├── src/        # Python source files
├── data/       # train.h5, test.h5 and downloaded raw shards
├── models/     # *.pt checkpoints
├── figures/    # *.png plots
├── logs/       # run logs
├── README.md
└── requirements.txt
```

## Execution Status (Current Workspace)

I ran all training command patterns in this README and generated the figures/checkpoints in this folder.

### Commands executed

```bash
# Install dependencies
c:\Users\kapoo\Downloads\qml\.venv\Scripts\python.exe -m pip install -r c:\Users\kapoo\Downloads\qml\task2\requirements.txt

# Single-model runs
c:\Users\kapoo\Downloads\qml\.venv\Scripts\python.exe c:\Users\kapoo\Downloads\qml\task2\src\train.py --model dgcnn --train data/train.h5 --test data/test.h5 --epochs 50 --batch-size 128 --max-jets 200
c:\Users\kapoo\Downloads\qml\.venv\Scripts\python.exe c:\Users\kapoo\Downloads\qml\task2\src\train.py --model gat   --train data/train.h5 --test data/test.h5 --epochs 50 --batch-size 128 --max-jets 200

# Comparison run
c:\Users\kapoo\Downloads\qml\.venv\Scripts\python.exe c:\Users\kapoo\Downloads\qml\task2\src\train.py --compare --train data/train.h5 --test data/test.h5 --epochs 50 --batch-size 128 --max-jets 200

# Quick smoke command from this README
c:\Users\kapoo\Downloads\qml\.venv\Scripts\python.exe c:\Users\kapoo\Downloads\qml\task2\src\train.py --compare --train data/train.h5 --test data/test.h5 --max-jets 2000 --epochs 5 --batch-size 64
```

### Observed outcomes

- Dependency installation completed successfully.
- All training commands above executed successfully and wrote outputs.
- The smoke run (`--max-jets 2000 --epochs 5`) produced:
    - `DGCNN`: Accuracy `0.7530`, ROC AUC `0.8406`
    - `GAT`: Accuracy `0.7320`, ROC AUC `0.8491`
- The 50-epoch compare run with `--max-jets 200` produced:
    - `DGCNN`: Accuracy `0.5100`, ROC AUC `0.8093`
    - `GAT`: Accuracy `0.5000`, ROC AUC `0.6961`

### Generated artifacts (organized in subfolders)

- `models/dgcnn_best.pt`
- `models/gat_best.pt`
- `figures/roc_dgcnn.png`
- `figures/roc_gat.png`
- `figures/roc_comparison.png`
- `figures/training_history.png`
- run logs in `logs/`: `dgcnn_run.log`, `gat_run.log`, `compare_run.log`, `smoke_run.log`

### Notes

- I used the official Zenodo shards and converted them into `data/train.h5` and `data/test.h5` expected by this codebase.
- For the 50-epoch commands, I added `--max-jets 200` so all runs finish in-session on CPU.
- The quick smoke command was run exactly as documented (`--max-jets 2000 --epochs 5 --batch-size 64`).


## Dataset

The dataset provides PYTHIA8-simulated quark and gluon jets with up to 100 particles per jet, stored as zero-padded Cartesian 4-momenta $(p_x, p_y, p_z, E)$ per particle.

**Download and place the files as:**
```
task2/
└── data/
    ├── train.h5
    └── test.h5
```

## Point Cloud to Graph: Design Considerations

A key modelling decision is how to project the raw particle point cloud into a structured graph. This choice directly determines which inter-particle relationships the GNN can discover.

### Why a graph at all?

Treating each jet as an unordered set of particles — as in a Deep Set or simple MLP-based classifier — ignores the spatial correlations between particles that carry the most discriminating information. Quarks produce narrower, lower-multiplicity jets with harder fragmentation functions; gluons produce wider jets with softer, denser particle distributions. These properties manifest in how nearby particles are related to each other, not just in their individual kinematics.

### Coordinate choice: $(\Delta\eta, \Delta\phi)$ angular space

Each particle is assigned coordinates relative to the jet centroid:
$$\Delta\eta_i = \eta_i - \eta_\text{jet}, \qquad \Delta\phi_i = \phi_i - \phi_\text{jet}$$

where $\eta_\text{jet}$ and $\phi_\text{jet}$ are the pT-weighted centroid of the jet. The angular separation $\Delta R = \sqrt{\Delta\eta^2 + \Delta\phi^2}$ is the standard HEP measure of how collimated a particle is. Working in this space ensures that the graph geometry directly reflects the physical topology of the jet.

### $k$-nearest neighbours with $k = 16$

For each particle, directed edges are drawn to its 16 nearest neighbours in $(\Delta\eta, \Delta\phi)$ space. This value of $k$ is adopted from the original ParticleNet paper, which showed it provides sufficient local context without over-connecting the graph. With typical multiplicities of 20–60 particles per jet after masking, $k = 16$ connects each particle to roughly a quarter to half of the other particles — enough for multi-hop message passing to reach all nodes within 2–3 layers.

### Node feature engineering

For each particle, six features are computed:

| Feature | Formula | Reason |
|---------|---------|--------|
| $\Delta\eta$ | $\eta_i - \eta_\text{jet}$ | Angular position relative to jet axis |
| $\Delta\phi$ | $\phi_i - \phi_\text{jet}$ | Azimuthal position relative to jet axis |
| $\log p_T$ | $\log(\sqrt{p_x^2 + p_y^2})$ | Transverse momentum, log-scaled |
| $\log E$ | $\log(E_i)$ | Particle energy, log-scaled |
| $\log(p_T / \Sigma p_T)$ | $\log(p_T^{(i)} / p_T^\text{jet})$ | Relative pT fraction — jet-scale invariant |
| $\log(E / \Sigma E)$ | $\log(E_i / E_\text{jet})$ | Relative energy fraction — jet-scale invariant |

Log-scaling momentum quantities compresses the dynamic range from several orders of magnitude to a range more compatible with gradient-based optimisation. The relative features (columns 5 and 6) make the representation invariant to the absolute jet energy scale, which varies significantly across the dataset.

## Architecture 1: Dynamic Graph CNN (DGCNN)

**File:** `src/dgcnn.py`

### Motivation

The DGCNN is chosen because its defining feature — **dynamic graph recomputation** between layers — is physically well-motivated for jet analysis. In the first layer, the graph is built in angular space and captures geometrically adjacent particles. After the first EdgeConv transformation, the graph is rebuilt in the learned feature space. Particles that were angular neighbours but are otherwise dissimilar may no longer be connected; conversely, particles from opposite sides of the jet but with similar learned representations (for example, two hard daughters of the same $g \to q\bar{q}$ splitting) may become connected. This allows the network to progressively discover long-range correlations that a fixed angular graph would miss entirely.

### EdgeConv operation

For each node $i$ with $k$ nearest neighbours $\{j\}$ in the current feature space, the edge feature is:
$$\mathbf{e}_{ij} = [\mathbf{h}_i \,\|\, \mathbf{h}_j - \mathbf{h}_i]$$

The term $\mathbf{h}_j - \mathbf{h}_i$ explicitly encodes the *relative* displacement between neighbours, making the convolution translation-invariant in feature space. A shared MLP is applied to each edge and the neighbourhood is aggregated by max-pooling:
$$\mathbf{h}_i' = \max_{j \in \mathcal{N}(i)} \text{MLP}([\mathbf{h}_i \,\|\, \mathbf{h}_j - \mathbf{h}_i])$$

### Architecture summary

| Component | Input → Output |
|-----------|---------------|
| Input projection | 6 → 64 (Linear + BN + LeakyReLU) |
| DynamicEdgeConv 1 | 64 → 64 |
| DynamicEdgeConv 2 | 64 → 128 |
| DynamicEdgeConv 3 | 128 → 256 |
| Global pool | concat(mean, max) over [64+128+256] = **896** |
| FC 1 | 896 → 512 |
| FC 2 | 512 → 256 |
| Output | 256 → 2 |

Multi-scale feature concatenation before pooling (collecting $\mathbf{h}_1$, $\mathbf{h}_2$, $\mathbf{h}_3$) lets the classifier draw simultaneously on fine-grained local structure from early layers and abstract global correlations from later layers.

## Architecture 2: Graph Attention Network (GAT)

**File:** `src/gat_net.py`

### Motivation

The GAT is chosen as a deliberate counterpoint to the DGCNN. Rather than rewiring the graph between layers, it keeps the physically motivated angular graph fixed and instead learns **attention weights** over the neighbourhood. This has two advantages:

1. **Interpretability.** The attention coefficients $\alpha_{ij}$ can be extracted after training and visualised as a learned importance map over particle pairs. This is directly useful for physics: it allows one to study which pairs of particles the model finds most discriminating, potentially uncovering physically interpretable substructure patterns.

2. **Stability.** Fixing the graph eliminates the extra computation of distance matrix construction during the forward pass, and avoids the instability that can arise when the dynamic graph structure changes rapidly at the start of training.

### Attention mechanism

For each directed edge $(i \to j)$, the attention coefficient is:
$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right), \qquad \alpha_{ij} = \text{softmax}_j(e_{ij})$$

The updated node representation is:
$$\mathbf{h}_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, \mathbf{W}\mathbf{h}_j\right)$$

With 8 parallel attention heads, the model can capture 8 independent weighting schemes over the neighbourhood simultaneously — for example, one head may focus on hard collinear partners while another attends to soft radiation.

### Architecture summary

| Component | Input → Output |
|-----------|---------------|
| Input projection | 6 → 64 (Linear + BN + ELU) |
| GATConv 1 | 64 → 64 × 8 = 512 (concat) |
| GATConv 2 + residual | 512 → 512 (concat) |
| GATConv 3 + residual | 512 → 512 (concat) |
| GATConv 4 | 512 → 256 (mean, 1 head) |
| Global pool | concat(mean, max) = **512** |
| FC 1 | 512 → 256 |
| FC 2 | 256 → 64 |
| Output | 64 → 2 |

Residual connections are added between layers 1–3 (which share the same width) to mitigate over-smoothing, a known failure mode of deep GAT stacks where all node representations converge to the same value.

## Performance Discussion

### Expected results

Both architectures were designed with complementary strengths. Based on their design properties and the literature on GNN-based jet tagging:

**DGCNN** is expected to achieve higher classification accuracy. The dynamic graph recomputation makes it strictly more expressive than a model with a fixed graph, as it can discover correlations that are invisible in angular space. On Q/G jet classification, DGCNN-class models typically achieve ROC AUC in the range **0.83–0.86**, approaching state-of-the-art deep learning results.

**GAT** is expected to perform competitively, in the range **0.81–0.84** ROC AUC, with the trade-off of being more interpretable. The fixed angular graph preserves the domain-informed geometry throughout every layer, which can act as a beneficial inductive bias. In data-limited regimes, this is likely to make the GAT more sample-efficient than the DGCNN.

### Key trade-offs

| Property | DGCNN | GAT |
|----------|-------|-----|
| Graph structure | Dynamic (rebuilt per layer in feature space) | Fixed (angular space, built once) |
| Aggregation | Max-pooling (robust to outliers) | Learned attention (adaptive weighting) |
| Expressiveness | Higher (graph rewiring unlocks non-local correlations) | Moderate (bounded by fixed neighbourhood) |
| Interpretability | Lower (graph changes per sample and per layer) | Higher (attention weights are human-readable) |
| Training speed | Slower (kNN recomputation in forward pass) | Faster (fixed edge_index, no extra computation) |
| Sample efficiency | Moderate | Higher (angular inductive bias) |

### Comparison with classical methods

Classical methods based on hand-crafted jet substructure observables (Nsubjettiness, $C_1$, $D_2$ etc.) fed into a BDT typically achieve ROC AUC roughly in the range **0.75–0.79** on this dataset. Both GNN architectures are expected to improve substantially on this baseline because they learn which particle-level features to extract directly from the data, rather than relying on a pre-specified set of engineered variables.

## Running the Code

### Install dependencies

```bash
# PyTorch Geometric requires that PyTorch is already installed first
pip install torch>=2.0.0

# Match the torch+cuda version from https://pyg.org/whl/
pip install torch-geometric

# Remaining dependencies
pip install -r requirements.txt
```

### Train a single model

```bash
python src/train.py --model dgcnn --train data/train.h5 --test data/test.h5 \
    --epochs 50 --batch-size 128

python src/train.py --model gat   --train data/train.h5 --test data/test.h5 \
    --epochs 50 --batch-size 128
```

### Train both and compare

```bash
python src/train.py --compare --train data/train.h5 --test data/test.h5 \
    --epochs 50 --batch-size 128
```

This produces:
- `figures/roc_comparison.png` — overlaid ROC curves for both models
- `figures/training_history.png` — loss and AUC curves across epochs
- `models/dgcnn_best.pt` / `models/gat_best.pt` — saved model weights

### Quick smoke test

```bash
python src/train.py --compare --train data/train.h5 --test data/test.h5 \
    --max-jets 2000 --epochs 5 --batch-size 64
```
