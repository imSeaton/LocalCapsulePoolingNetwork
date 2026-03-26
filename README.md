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

## Installation

```bash
pip install torch torch-geometric torch-scatter
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
