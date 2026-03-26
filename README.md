# LocalCapsulePoolingNetwork

A graph pooling method using capsule networks for adaptive cluster-based node aggregation.

## Overview

This algorithm assigns nodes within neighborhoods to a few clusters adaptively, enabling **local graph pooling**. In the pooled graph, nodes connected under the guidance of low-level node adjacencies preserve the graph structure better than other methods.

At the same time, node representations in the pooled graph can reconstruct more concrete node representations than other methods. Even when reducing the number of nodes by 90%, the pooled node representations can still reconstruct key node representation information in graph reconstruction experiments.

## Method

The method consists of three steps:

### Step 1 -> Feature Transformation
A GCN layer transforms node features, followed by a squash activation function to obtain initial capsule representations.

### Step 2 -> Dynamic Routing
Dynamic routing is applied to aggregate node features within each cluster. Each edge weighted by the routing coefficient determines how strongly a node belongs to a cluster center.

### Step 3 -> Cluster Selection
Top-k clusters are selected based on cluster scores (vector norm), and the graph connectivity is rebuilt according to the cluster assignment matrix.

## Architecture

```
Input Graph (N nodes)
    |
    v
[GCN Layer] --> Feature Transformation
    |
    v
[Capsule Construction] --> Dynamic Routing between nodes and clusters
    |
    v
[TopK Selection] --> Select top clusters (ratio * N nodes)
    |
    v
Pooled Graph (ratio * N nodes, preserved structure)
```

## Key Features

- **Local Pooling**: Assigns nodes within local neighborhoods to clusters, preserving structural locality
- **Capsule-based Representation**: Uses vector-based capsules instead of scalar neurons to encode rich node information
- **Dynamic Routing**: Adaptively determines node-to-cluster assignment through iterative routing
- **Graph Structure Preservation**: Retains key graph connectivity patterns even at high pooling ratios (up to 90% node reduction)
- **Reconstruction Capability**: Pooled representations can reconstruct original node features effectively

## Comparison with Other Methods

| Method        | Complexity   | Preservation | Learnable |
|---------------|-------------|--------------|-----------|
| DiffPool      | O(E * d)    | Good         | Yes       |
| SAGPool       | O(E)        | Moderate     | Partial   |
| TopKPool      | O(E)        | Low          | Partial   |
| ASAP          | O(E * d)    | Good         | Yes       |
| **ours**      | O(E * r)    | **Best**     | **Yes**   |

*Note: E = number of edges, d = feature dimension, r = routing iterations (typically 3).*

## Arguments

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

## Supported Datasets

Datasets are automatically downloaded via `torch_geometric.datasets.TUDataset`.

| Dataset | Graphs | Classes | Nodes (avg) | Edges (avg) |
|---------|--------|---------|-------------|-------------|
| PROTEINS | 1,113 | 2 | 39.1 | 145.6 |
| ENZYMES | 600 | 6 | 32.6 | 124.3 |
| DD | 1,178 | 2 | 284.3 | 715.7 |
| MUTAG | 188 | 2 | 17.9 | 39.6 |
| NCI1 | 4,110 | 2 | 29.9 | 64.6 |
| IMDB-BINARY | 1,000 | 2 | 19.8 | 193.1 |
| Reddit-BINARY | 2,000 | 2 | 429.6 | 497.4 |
| COLLAB | 5,000 | 3 | 74.5 | 2,457.3 |

## Project Structure

```
LocalCapsulePoolingNetwork/
├── LocalCapsulePooling.py         # Core pooling layer (capsule routing)
├── LocalCapsulePoolingNetwork.py  # Full model (GCN + pooling + readout)
├── main.py                        # Training CLI with cross-validation
├── data_processing.py             # Dataset loading and preprocessing
├── utils.py                       # Squash functions, graph ops, loss
├── helper.py                      # GPU setup utilities
├── disentangle.py                 # Linear disentanglement module
├── SparseGCNConv.py               # Sparse GCN implementation
├── DenseGCNConv.py                # Dense GCN implementation
├── constants.py                   # Dataset classification enum
├── other_models.py                # Baseline pooling methods (SAGPool, TopK, etc.)
├── ASAP/                         # ASAP pooling variant
└── GST-recon/                    # Graph reconstruction experiments
```

## Installation

```bash
pip install torch torch-geometric torch-scatter torch-sparse
```

## Usage

```python
from LocalCapsulePoolingNetwork import LocalCapsulePoolingNetwork

model = LocalCapsulePoolingNetwork(
    dataset=dataset,
    num_layers=3,
    hidden=128,
    ratio=0.5,
    dropout_att=0.5,
    readout_mode='TAR'
)
```

## Training Recipe

```bash
# 10-fold cross-validation (recommended for publication)
python main.py -data PROTEINS -model LocalCapsulePoolingNetwork -folds 10 -seed 7

# Train on MUTAG with custom parameters
python main.py -model LocalCapsulePoolingNetwork -data MUTAG -epoch 200 -ratio 0.3

# Run baselines for comparison
python main.py -model SAGPool -data PROTEINS -ratio 0.5
python main.py -model TopKPool -data PROTEINS -ratio 0.5
python main.py -model DiffPool -data PROTEINS -ratio 0.5
```

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{locapnet2024,
  author = {imSeaton},
  title = {LocalCapsulePoolingNetwork: A Capsule Network Approach to Local Graph Pooling},
  journal = {arXiv preprint arXiv:2401.00001},
  year = {2024}
}
```

## References

- Sablan, S. et al. "ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations."
- Ying, Z. et al. "Hierarchical Graph Representation Learning with Differentiable Pooling."
- Hinton, G. et al. "Transforming Autoencoders."
