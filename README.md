# LocalCapsulePoolingNetwork

**A Capsule-Network-Driven Graph Pooling Method for Hierarchical Graph Representation Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.0+-green.svg)](https://pytorch-geometric.readthedocs.io/)

---

## 📌 Overview

**LocalCapsulePoolingNetwork** is a novel graph pooling framework that leverages capsule networks to adaptively group nodes within local neighborhoods into hierarchical clusters. Unlike conventional pooling operators that select nodes independently or rely on rigid aggregation heuristics, this method uses **dynamic routing** to learn cluster assignments — enabling it to preserve fine-grained graph topology even under aggressive compression.

In each pooling layer, the algorithm:
1. Transforms node features via a GCN layer
2. Applies squash non-linearity to obtain capsule vectors
3. Routes nodes to cluster representatives using an iterative attention mechanism
4. Selects top-$k$ clusters via a learned scoring function

The resulting pooled graph maintains connectivity through a differentiable **$S^T A S$** reassembly operation, ensuring gradient flow and structural preservation.

---

## 🔬 Method

### Core Algorithm

Given an input graph $\mathcal{G} = (\mathbf{X}, \mathbf{A})$ with node features $\mathbf{X} \in \mathbb{R}^{N \times F}$ and adjacency matrix $\mathbf{A}$, a single LocalCapsulePooling layer produces a contracted graph $\mathcal{G}' = (\mathbf{X}', \mathbf{A}')$ as follows:

**Step 1 — Feature Transformation**
$$\mathbf{H} = \text{GCN}(\mathbf{X}, \mathbf{A})$$

**Step 2 — Squash Non-Linearity**
$$\mathbf{V}_i = \text{squash}(\mathbf{H}_i) = \frac{\|\mathbf{H}_i\|^2}{1 + \|\mathbf{H}_i\|^2} \cdot \frac{\mathbf{H}_i}{\|\mathbf{H}_i\|}$$

**Step 3 — Dynamic Routing**
For $r$ routing iterations:
$$c_{ij} = \text{softmax}(b_{ij}) = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}$$
$$b_{ij} \leftarrow b_{ij} + \mathbf{v}_j^\top \mathbf{u}_i$$

**Step 4 — Cluster Selection**
$$\text{score}_k = \|\mathbf{c}_k\|, \quad \text{perm} = \text{topk}(\text{score}, \lceil r \cdot N \rceil)$$

**Step 5 — Graph Reassembly**
$$\mathbf{A}' = \mathbf{S}^\top \mathbf{A} \mathbf{S}$$

The pooling ratio $r$ controls the fraction of nodes retained per layer. Even at **10% retention**, the method retains sufficient discriminative information for competitive classification accuracy.

### Architecture

```
Input Graph
    │
    ▼
┌─────────────────────────────┐
│  GCN (feature transform)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Squash + BatchNorm        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Dynamic Routing (r iter)  │◄─── cluster assignment matrix S
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  TopK Cluster Selection    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  S^T A S Reassembly       │
└──────────────┬──────────────┘
               │
               ▼
      Pooled Graph (N' < N)
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Capsule-based pooling** | Leverages vector-valued representations to capture node importance directionality |
| **Local adjacency guidance** | Cluster formation guided by original graph topology, preserving structural locality |
| **Configurable routing** | Number of routing iterations is a hyperparameter (default: 3) |
| **Residual connections** | Skip connections prevent information degradation in deep stacks |
| **Structural preservation** | $S^\top A S$ reassembly maintains graph connectivity after pooling |
| **Early stopping** | Built-in patience-based early stopping to prevent overfitting |
| **Gradient clipping** | Stable training via gradient norm capping |

---

## 📦 Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- PyTorch Geometric ≥ 2.0
- torch-scatter
- torch-sparse
- torch-cluster

### Steps

```bash
# Clone the repository
git clone https://github.com/imSeaton/LocalCapsulePoolingNetwork.git
cd LocalCapsulePoolingNetwork

# Install dependencies (recommended: create a new conda/virtual environment)
pip install -r requirements.txt

# Or install core dependencies manually
pip install torch torch-geometric torch-scatter torch-sparse torch-cluster
```

### Verify Installation

```python
import torch
from LocalCapsulePoolingNetwork import LocalCapsulePoolingNetwork
from data_processing import get_dataset

dataset = get_dataset('PROTEINS')
print(f"Dataset loaded: {dataset}")
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
from LocalCapsulePoolingNetwork import LocalCapsulePoolingNetwork
from data_processing import get_dataset

# Load dataset
dataset = get_dataset('PROTEINS')

# Initialize model
model = LocalCapsulePoolingNetwork(
    dataset=dataset,
    num_layers=3,
    hidden=128,
    ratio=0.5,
    dropout_att=0.5,
    readout_mode='TAR',
    dataset_name='PROTEINS'
)

# Forward pass
data = dataset[0]
out, aux_loss, pooled_x, pooled_edge_index = model(data)
print(f"Output shape: {out.shape}")  # (1, num_classes)
```

### Training via CLI

```bash
# Train on PROTEINS dataset
python main.py -model LocalCapsulePoolingNetwork -data PROTEINS -epoch 100

# Train on MUTAG dataset with custom parameters
python main.py \
    -model LocalCapsulePoolingNetwork \
    -data MUTAG \
    -epoch 200 \
    -ratio 0.3 \
    -hid_dim 256 \
    -lr 0.001 \
    -early_stopping_patience 50

# Train on DD dataset
python main.py -model LocalCapsulePoolingNetwork -data DD -num_layers 4
```

### Run Baselines

```bash
# Compare with other pooling methods
python main.py -model SAGPool -data PROTEINS -ratio 0.5
python main.py -model TopKPool -data PROTEINS -ratio 0.5
python main.py -model ASAP -data PROTEINS -ratio 0.5
python main.py -model DiffPool -data PROTEINS -ratio 0.5
```

---

## 🔧 Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-model` | `LocalCapsulePoolingNetwork` | Model architecture |
| `-data` | `PROTEINS` | Dataset name |
| `-epoch` | `100` | Maximum training epochs |
| `-alpha` | `1e-4` | Auxiliary loss weight (F-norm stability) |
| `-readout_mode` | `TAR` | Readout mode: `TAR` (task-aware) or `Common` |
| `-l2` | `1e-3` | L2 regularization coefficient |
| `-num_layers` | `3` | Number of GCN + pooling layers |
| `-lr_decay_step` | `50` | Learning rate decay interval (epochs) |
| `-lr_decay_factor` | `0.5` | Learning rate decay multiplier |
| `-batch` | `128` | Batch size |
| `-hid_dim` | `128` | Hidden dimension |
| `-dropout_att` | `0.5` | Dropout rate on attention scores |
| `-lr` | `0.001` | Initial learning rate |
| `-ratio` | `0.5` | Pooling ratio (fraction of nodes retained) |
| `-folds` | `10` | Cross-validation folds |
| `-early_stopping_patience` | `50` | Epochs to wait before early stopping |
| `-gpu` | `0` | GPU device ID (`-1` for CPU) |

---

## 📊 Supported Datasets

Datasets are automatically downloaded via `torch_geometric.datasets.TUDataset`.

### Biological / Molecular Graphs

| Dataset | Graphs | Classes | Nodes (avg) | Edges (avg) |
|---------|--------|---------|-------------|-------------|
| `DD` | 1,178 | 2 | 284.3 | 715.7 |
| `MUTAG` | 188 | 2 | 17.9 | 39.6 |
| `NCI1` | 4,110 | 2 | 29.9 | 64.6 |
| `NCI109` | 4,127 | 2 | 29.7 | 64.3 |
| `ENZYMES` | 600 | 6 | 32.6 | 124.3 |
| `PROTEINS` | 1,113 | 2 | 39.1 | 145.6 |
| `FRANKENSTEIN` | 4,337 | 2 | 16.9 | 34.8 |

### Social / Synthetic Graphs

| Dataset | Graphs | Classes | Nodes (avg) | Edges (avg) |
|---------|--------|---------|-------------|-------------|
| `IMDB-BINARY` | 1,000 | 2 | 19.8 | 193.1 |
| `IMDB-MULTI` | 1,500 | 3 | 13.0 | 131.9 |
| `REDDIT-BINARY` | 2,000 | 2 | 429.6 | 497.4 |
| `REDDIT-MULTI` | 4,999 | 5 | 508.5 | 594.9 |
| `COLLAB` | 5,000 | 3 | 74.5 | 2,457.3 |

---

## 📂 Project Structure

```
LocalCapsulePoolingNetwork/
├── constants.py                      # Dataset classification enum
├── LocalCapsulePooling.py            # Core pooling layer (capsule routing)
├── LocalCapsulePoolingNetwork.py     # Full model (GCN + pooling + readout)
├── main.py                           # Training CLI with cross-validation
├── utils.py                         # Squash functions, graph ops, loss functions
├── data_processing.py               # Dataset loading and preprocessing
├── helper.py                        # GPU setup utilities
├── disentangle.py                   # Linear disentanglement module
├── SparseGCNConv.py                 # Sparse GCN implementation
├── DenseGCNConv.py                  # Dense GCN implementation
├── other_models.py                  # Baseline pooling methods (SAGPool, TopK, etc.)
├── ASAP/                            # ASAP pooling variant
└── GST-recon/                       # Graph reconstruction experiments
```

---

## 🧩 Comparison with Other Pooling Methods

| Method | Pooling Strategy | Preservation | Learnable | Complexity |
|--------|-----------------|--------------|-----------|------------|
| **LocalCapsulePooling** | Capsule routing + top-k | Local topology + feature | ✅ | O(E·r) |
| TopKPool | Score-based selection | Feature importance only | ✅ | O(N) |
| SAGPool | Self-attention + top-k | Feature importance + graph | ✅ | O(N²) |
| ASAP | Attention scoring | Feature importance only | ✅ | O(N·E) |
| DiffPool | Differentiable assignment | Hierarchical structure | ✅ | O(N³) |
| MinCutPool | Spectral clustering | Graph structure | ✅ | O(N³) |

---

## 📈 Training Recipe

```bash
# 10-fold cross-validation with seed averaging (recommended for publication)
python main.py -data PROTEINS -model LocalCapsulePoolingNetwork -folds 10 -seed 7

# Reproducibility: use multiple seeds and average
for seed in 7 42 123 456 789; do
    python main.py -data PROTEINS -model LocalCapsulePoolingNetwork -seed $seed
done
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📖 Citation

If this work is helpful to your research, please consider citing:

```bibtex
@article{localcapsulepooling2024,
  title={LocalCapsulePooling: Capsule-Network-Driven Graph Pooling with Local Topology Preservation},
  author={},
  journal={},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project builds upon several excellent open-source libraries:

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) — Graph neural network foundation
- [torch-scatter / torch-sparse](https://github.com/rusty1s/pytorch_sparse) — Efficient sparse tensor operations
- Inspired by [Capsule Networks (Sabour et al., 2017)](https://arxiv.org/abs/1710.09829) and their application to graph-structured data
