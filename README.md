<div align="center">
  <h1>ICGNN</h1>
  <h3>Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction</h3>
  <p>
    <a href="https://github.com/wayc04/ICGNN">
      <img alt="GitHub" src="https://img.shields.io/badge/GitHub-ICGNN-black?logo=github" />
    </a>
    <a href="https://icml.cc/Conferences/2026">
      <img alt="ICML 2026" src="https://img.shields.io/badge/ICML%202026-Poster-2ea44f" />
    </a>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.8-blue" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.10.0-ee4c2c" />
  </p>
</div>

This repository provides the PyTorch implementation for:

> Wei Ju, Wei Zhang, Siyu Yi, Zhengyang Mao, Yifan Wang, Jingyang Yuan, Zhiping Xiao, Ziyue Qiao, and Ming Zhang.<br>
> **Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction.**<br>
> International Conference on Machine Learning (ICML), 2026.

## Overview

ICGNN addresses semi-supervised node classification when labeled nodes are both scarce and corrupted by label noise. The paper observes that, under graph homophily and message passing, a labeled node receiving strong influence from nodes annotated as other classes is more likely to have an unreliable label. It therefore introduces the **Influence Contradiction Score (ICS)** as a noise indicator, then uses soft label cleaning and pseudo-labeling to train a robust GNN.

The method is organized around three components from the paper:

1. **Noise detection by influence contradiction.**
2. **Noise cleaning by neighbor aggregation.**
3. **Optimization against noisy and limited labels.**

## Paper-Grounded Methodology

### 1. Influence Contradiction Score

ICGNN first builds a graph diffusion matrix with Personalized PageRank:

```text
T = epsilon * (I - (1 - epsilon) * A_hat)^(-1)
```

Each row of `T` describes how a node influences other nodes through the graph. For a labeled node, the structure-level ICS accumulates the influence it receives from labeled nodes assigned to other classes, normalized by class size. A smaller ICS indicates that the node is more consistent with its annotated class, while a larger ICS indicates stronger contradiction and a higher probability of label noise.

The paper further notes that topology alone may overlook attribute information. ICGNN therefore extracts node representations with a GNN encoder, constructs a KNN-based representation affinity graph, computes an attribute-level diffusion matrix `R`, and derives an attribute-level ICS in the same spirit. The final noise indicator combines both signals:

```text
ICS_i = (1 - alpha) * ICS_i(T) + alpha * ICS_i(R)
```

In the paper's experimental setting, `alpha` is set to `0.5`.

### 2. GMM-Based Noise Confidence

After obtaining ICS values, ICGNN fits a two-component Gaussian Mixture Model with the EM algorithm. The posterior probability assigned to the component with the smaller mean is used as the clean-label confidence `beta_i`. This gives a learnable soft threshold for separating likely clean labels from likely noisy labels, avoiding a manually fixed hard threshold.

### 3. Noise Cleaning by Neighbor Aggregation

Instead of forcing noisy labels to a single neighbor-voted class, the paper proposes a conservative soft correction strategy. At training epoch `t`, the updated supervision for a labeled node is a convex combination of its original noisy one-hot label and the prediction aggregated from graph-diffusion neighbors:

```text
l_i^(t) = beta_i^(t) * y_i + (1 - beta_i^(t)) * h^(t)(z_i)
```

Here, `h^(t)(z_i)` aggregates neighbors' predictions using weights from the diffusion matrix `T`, followed by softmax normalization. A high `beta_i` keeps more of the original label; a low `beta_i` relies more on neighbor-aggregated prediction.

### 4. Pseudo-Labeling for Unlabeled Nodes

Because the problem setting contains few labeled nodes and many unlabeled nodes, ICGNN also applies the same neighbor-aggregation strategy to unlabeled nodes. These pseudo-labels provide auxiliary supervision and help mitigate label scarcity.

### 5. Training Objective

Following Eq. (8) in the paper, the model is optimized with cross-entropy supervision from two sources:

| Source | Supervision signal |
| --- | --- |
| Labeled nodes | Cleaned soft labels from ICS confidence and neighbor aggregation. |
| Unlabeled nodes | Neighbor-aggregated pseudo-labels. |

The appendix further clarifies that ICS computation, GMM confidence assignment, and label correction are auxiliary operations outside the direct gradient path. They iteratively refine the labels used for training, while gradients flow through the primary GNN model.

## Experimental Protocol in the Paper

The paper evaluates ICGNN on six benchmark datasets:

| Category | Datasets |
| --- | --- |
| Author network | Coauthor CS |
| Co-purchase network | Amazon Photo |
| Citation networks | Cora, Pubmed, Citeseer, DBLP |

The main experiments use:

| Setting | Value |
| --- | --- |
| Test split | 80% of nodes |
| Validation split | 10% of nodes |
| Labeled training rate | 1% for Coauthor CS, Amazon Photo, Pubmed, DBLP; 5% for Cora and Citeseer |
| Noise types | Uniform noise and pair noise |
| Default noise rate | 20% |
| Teleport probability | 0.85 |
| KNN size for representation affinity graph | 5 |
| ICS trade-off `alpha` | 0.5 |
| Training epochs | 200 |
| Evaluation | Mean accuracy and standard deviation over 5 runs |

The paper compares against GCN, Forward, Coteaching+, NRGNN, RTGNN, CGNN, CR-GNN, DND-NET, and ProCon. Its ablation study removes structure-level ICS, attribute-level ICS, noise cleaning, and pseudo-labeling, and also replaces the graph diffusion matrix with the adjacency matrix. The reported results show that both structure-level and attribute-level contradiction are complementary, noise cleaning is important for robustness, pseudo-labeling improves supervision under label scarcity, and graph diffusion is more effective than using only the local adjacency matrix.

## Requirements

The code was developed with:

```text
python == 3.8
torch == 1.10.0
torch-geometric == 2.0.2
```

Additional packages used by the training pipeline include `deeprobust`, `numpy`, `scipy`, `scikit-learn`, `networkx`, and `loguru`.

## Quick Start

Run ICGNN on Pubmed with uniform label noise:

```bash
python train.py \
  --dataset pubmed \
  --ptb_rate 0.2 \
  --noise uniform \
  --label_rate 0.01 \
  --K 75 \
  --local_conflict_weight 0.8 \
  --warmup_epochs 30 \
  --scale1 1.0 \
  --temp 0.5
```

Run ICGNN on Amazon Photo with pair noise:

```bash
python train.py \
  --dataset photo \
  --ptb_rate 0.2 \
  --noise pair \
  --label_rate 0.01 \
  --K 100 \
  --local_conflict_weight 0.8 \
  --warmup_epochs 15 \
  --temp 1.0
```

More dataset-specific commands are provided in [`run.sh`](run.sh).

## Repository Layout

```text
ICGNN/
|-- data/              # Dataset files and cached graph artifacts
|-- models/
|   |-- GCN.py          # Backbone graph convolutional network
|   `-- ICGNN.py        # ICGNN training pipeline and edge estimator
|-- dataset.py          # Dataset loading and preprocessing
|-- train.py            # Main experimental entry point
|-- utils.py            # Noise generation, ICS utilities, and metrics
`-- run.sh              # Reproduction commands
```

## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{ju2026identifying,
  title     = {Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction},
  author    = {Ju, Wei and Zhang, Wei and Yi, Siyu and Mao, Zhengyang and Wang, Yifan and Yuan, Jingyang and Xiao, Zhiping and Qiao, Ziyue and Zhang, Ming},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
