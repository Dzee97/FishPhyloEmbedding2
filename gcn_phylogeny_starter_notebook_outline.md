# GCN starter notebook for the phylogeny modality

This canvas contains a ready-to-copy notebook outline with markdown and code cells for a toy 5-taxon phylogeny represented as `edge_index`, `edge_weight`, and `nodes.csv`.

## Markdown cell 1
# GCN starter notebook for the phylogeny modality

This notebook is a compact introduction to how we can represent a phylogeny as a graph and use that graph as input to a graph convolutional network (GCN).

It is meant as a **conceptual and practical starting point** for the tree branch of the project. It mirrors the outputs of the preprocessing notebook, where the pruned Fish Tree of Life is exported as:

- `edge_index.npy`
- `edge_weight.npy`
- `nodes.csv`

To keep the ideas easy to follow, this notebook uses a **small toy phylogeny with 5 taxa** instead of the full fish tree.

## Learning goals

By the end of this notebook, you should be able to explain:

1. how a phylogeny can be stored as a graph instead of a Newick file
2. how `edge_index`, `edge_weight`, and `nodes.csv` describe that graph
3. how to turn the graph into an adjacency matrix
4. how to define a simple node feature matrix
5. what a GCN layer does conceptually
6. why leaf embeddings should reflect phylogenetic distances
7. which kinds of losses could be used to train that objective

## Markdown cell 2
## 1. Toy phylogeny

We use a small unrooted phylogeny with **5 taxa**:

- A
- B
- C
- D
- E

An unrooted binary tree with 5 taxa has **3 internal nodes**.
So our graph will have:

- 5 leaf nodes
- 3 internal nodes
- 8 total nodes

We will label the internal nodes as:

- `i0`
- `i1`
- `i2`

### Topology

The toy tree is:

- `A` and `B` connect to internal node `i0`
- `D` and `E` connect to internal node `i2`
- `C` connects to internal node `i1`
- the internal backbone is `i0 -- i1 -- i2`

### Branch lengths

We also assign simple branch lengths:

- A -- i0 : 0.2
- B -- i0 : 0.3
- i0 -- i1 : 0.5
- C -- i1 : 0.4
- i1 -- i2 : 0.6
- D -- i2 : 0.2
- E -- i2 : 0.5

This is only a toy example, but it is enough to illustrate the full pipeline.

## Code cell 1
```python
import numpy as np
import pandas as pd

# Stable node indexing, similar to the preprocessing export
# Leaves first, then internal nodes
nodes_df = pd.DataFrame([
    {"node_id": 0, "node_name": "A",  "is_leaf": True,  "species": "A"},
    {"node_id": 1, "node_name": "B",  "is_leaf": True,  "species": "B"},
    {"node_id": 2, "node_name": "C",  "is_leaf": True,  "species": "C"},
    {"node_id": 3, "node_name": "D",  "is_leaf": True,  "species": "D"},
    {"node_id": 4, "node_name": "E",  "is_leaf": True,  "species": "E"},
    {"node_id": 5, "node_name": "i0", "is_leaf": False, "species": pd.NA},
    {"node_id": 6, "node_name": "i1", "is_leaf": False, "species": pd.NA},
    {"node_id": 7, "node_name": "i2", "is_leaf": False, "species": pd.NA},
])

nodes_df
```

## Markdown cell 3
## 2. Mimicking the preprocessing outputs

In the real preprocessing pipeline, the tree is exported as three files:

- `edge_index.npy` — graph connectivity
- `edge_weight.npy` — branch lengths
- `nodes.csv` — node metadata

We recreate those same objects here.

### `edge_index`

This is a `2 x E` integer array.
Each column is one directed edge:

- first row = source node
- second row = destination node

Because the tree is exported as an **undirected graph**, each undirected edge is stored in **both directions**.

### `edge_weight`

This stores the branch length for each directed edge in `edge_index`.

### `nodes.csv`

This stores:

- integer node IDs
- node names
- whether the node is a leaf
- species name for leaf nodes

## Code cell 2
```python
# Undirected tree edges (stored in both directions)
# Format here: (u, v, weight)
undirected_edges = [
    (0, 5, 0.2),  # A - i0
    (1, 5, 0.3),  # B - i0
    (5, 6, 0.5),  # i0 - i1
    (2, 6, 0.4),  # C - i1
    (6, 7, 0.6),  # i1 - i2
    (3, 7, 0.2),  # D - i2
    (4, 7, 0.5),  # E - i2
]

src, dst, w = [], [], []
for u, v, weight in undirected_edges:
    src.extend([u, v])
    dst.extend([v, u])
    w.extend([weight, weight])

edge_index = np.array([src, dst], dtype=np.int64)
edge_weight = np.array(w, dtype=np.float32)

print("edge_index shape:", edge_index.shape)
print("edge_weight shape:", edge_weight.shape)
print()
print("edge_index:")
print(edge_index)
print()
print("edge_weight:")
print(edge_weight)
```

## Markdown cell 4
### Optional export

These are the exact objects the larger pipeline writes to disk.
The next cell shows how this toy example could be saved in the same format.

This is only for illustration; the rest of the notebook continues to work directly with the in-memory variables.

## Code cell 3
```python
from pathlib import Path

toy_out = Path("toy_phylo_graph")
toy_out.mkdir(exist_ok=True)

np.save(toy_out / "toy_edge_index.npy", edge_index)
np.save(toy_out / "toy_edge_weight.npy", edge_weight)
nodes_df.to_csv(toy_out / "toy_nodes.csv", index=False)

print("Saved:")
print(toy_out / "toy_edge_index.npy")
print(toy_out / "toy_edge_weight.npy")
print(toy_out / "toy_nodes.csv")
```

## Markdown cell 5
## 3. From graph files to adjacency matrix

A GCN usually starts from a graph representation. One common mathematical view is the **adjacency matrix** `A`.

For `N` nodes, `A` is an `N x N` matrix where:

- `A[i, j] = 1` if there is an edge from node `i` to node `j`
- `A[i, j] = 0` otherwise

For weighted graphs, we can also build a **weighted adjacency matrix** where the entry contains the branch length.

For large trees, we normally do **not** want a dense matrix, because it wastes memory.
Instead, we keep the sparse edge list (`edge_index`, `edge_weight`) and let the GNN library handle sparse operations.

Still, for learning the concepts, it is useful to build the adjacency matrix explicitly once.

## Code cell 4
```python
num_nodes = len(nodes_df)

A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
A_weighted = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for (u, v), weight in zip(edge_index.T, edge_weight):
    A[u, v] = 1.0
    A_weighted[u, v] = weight

print("Unweighted adjacency matrix:")
print(A)

print("\nWeighted adjacency matrix:")
print(A_weighted)
```

## Markdown cell 6
## 4. Node features

A GCN does not operate on graph structure alone.
It also needs a **feature vector for every node**.

This gives us a feature matrix `X` of shape:

- `N x F`
- `N` = number of nodes
- `F` = number of features per node

In the full project, node features might come from:

- transformer species embeddings for leaf nodes
- learned internal-node embeddings
- positional or structural tree features
- degree, branch-length, or distance-derived signals

For this starter notebook, we keep the features simple and interpretable. We use 3 features:

1. `is_leaf`
2. `degree`
3. `leaf_bias` — a toy feature that is `1` for leaves and `0` for internal nodes

This is intentionally simple. The goal is to understand the mechanics before plugging in real learned embeddings.

## Code cell 5
```python
degree = A.sum(axis=1)

X = np.stack([
    nodes_df["is_leaf"].astype(np.float32).to_numpy(),
    degree.astype(np.float32),
    nodes_df["is_leaf"].astype(np.float32).to_numpy(),
], axis=1)

feature_names = ["is_leaf", "degree", "leaf_bias"]

X_df = pd.DataFrame(X, columns=feature_names)
X_df.insert(0, "node_name", nodes_df["node_name"])
X_df
```

## Markdown cell 7
## 5. One GCN layer: the main idea

A graph convolution layer updates each node by mixing:

- its **own current features**
- the features of its **neighbors**

A simple conceptual form is:

\[
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
\]

where:

- \(H^{(l)}\) is the node feature matrix at layer `l`
- \(W^{(l)}\) is a learnable weight matrix
- \(\hat{A}\) is a normalized adjacency matrix
- \(\sigma\) is a nonlinearity such as ReLU

### Intuition

For each node:

1. gather messages from neighboring nodes
2. aggregate those messages
3. apply a learned linear transformation
4. apply a nonlinearity

This means information can move through the tree:

- after 1 layer, a node sees its 1-hop neighbors
- after 2 layers, it sees 2-hop neighbors
- after 3 layers, it sees 3-hop neighbors

That is why GNNs are a natural fit for phylogenies: they can propagate information through the tree topology.

## Markdown cell 8
## 6. Why normalization is used

If we used the raw adjacency matrix directly, nodes with many neighbors would accumulate larger values just because they have higher degree.

To make aggregation more stable, GCNs often use a normalized adjacency matrix.

A common choice is:

\[
\hat{A} = D^{-1/2}(A + I)D^{-1/2}
\]

where:

- `I` adds self-loops, so each node keeps its own features
- `D` is the degree matrix of `A + I`

This balances the contributions from different nodes and helps optimization.

## Code cell 6
```python
I = np.eye(num_nodes, dtype=np.float32)
A_tilde = A + I

deg_tilde = A_tilde.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_tilde))

A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

np.set_printoptions(precision=3, suppress=True)
print("Normalized adjacency matrix A_hat:")
print(A_hat)
```

## Markdown cell 9
## 7. A tiny manual message-passing example

To make the idea concrete, the next cell performs one toy GCN-like update by hand:

1. aggregate neighbors using the normalized adjacency matrix
2. apply a small linear transformation
3. apply ReLU

This is not a full model and there is no training here.
It is only meant to show what happens to the node features after one propagation step.

## Code cell 7
```python
H0 = X.astype(np.float32)

# Toy learnable weight matrix W: shape [3, 2]
W = np.array([
    [1.0, -0.5],
    [0.3,  0.8],
    [0.7,  0.2],
], dtype=np.float32)

H1_pre = A_hat @ H0 @ W
H1 = np.maximum(H1_pre, 0.0)  # ReLU

H1_df = pd.DataFrame(H1, columns=["hidden_0", "hidden_1"])
H1_df.insert(0, "node_name", nodes_df["node_name"])
H1_df
```

## Markdown cell 10
### What happened here?

Each row in `H1` is the new representation of one node after one graph-convolution step.

Because neighbors are mixed in through `A_hat`, the representation of a leaf is no longer based only on its own feature vector. It now also depends on the internal node it connects to, and vice versa.

This is the core mechanism behind GCNs:

- **topology controls message flow**
- **features provide the starting signal**
- **learned weights determine how the messages are combined**

## Markdown cell 11
## 8. Leaves vs internal nodes in this project

In our application, the phylogeny contains both:

- **leaf nodes** = observed species
- **internal nodes** = ancestral branching points

The training goal we have discussed is primarily about the **leaf embeddings**.
We want species that are close on the tree to end up close in embedding space.

Internal nodes are still important because they act as the pathways through which messages move. Even if we only supervise leaf embeddings, the internal nodes help structure the representation learning.

## Markdown cell 12
## 9. Pairwise tree distances between leaves

To train the graph branch, we need a target notion of species similarity from the phylogeny.

A natural target is the pairwise distance between leaves.
There are two common options:

1. **branch-length distance**
   sum of branch lengths along the path between two leaves

2. **topology distance**
   number of edges along the path between two leaves

For this toy notebook, we will compute the **branch-length distance matrix** between the 5 taxa.

## Code cell 8
```python
leaf_ids = nodes_df.loc[nodes_df["is_leaf"], "node_id"].to_list()
leaf_names = nodes_df.loc[nodes_df["is_leaf"], "node_name"].to_list()

# Build adjacency list for shortest-path traversal on the tree
adj = {i: [] for i in range(num_nodes)}
for (u, v), weight in zip(edge_index.T, edge_weight):
    adj[int(u)].append((int(v), float(weight)))

def tree_distances_from(start_leaf: int):
    # Because this is a tree, a simple stack is enough to visit all nodes once
    dist = {start_leaf: 0.0}
    stack = [start_leaf]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + w
                stack.append(v)
    return dist

leaf_dist = np.zeros((len(leaf_ids), len(leaf_ids)), dtype=np.float32)
for i, src_leaf in enumerate(leaf_ids):
    d = tree_distances_from(src_leaf)
    for j, dst_leaf in enumerate(leaf_ids):
        leaf_dist[i, j] = d[dst_leaf]

leaf_dist_df = pd.DataFrame(leaf_dist, index=leaf_names, columns=leaf_names)
leaf_dist_df
```

## Markdown cell 13
### Interpreting the distance matrix

A few examples from the toy tree:

- `A` and `B` are close because they meet immediately at internal node `i0`
- `D` and `E` are also close because they meet at `i2`
- `A` and `E` are farther apart because their path crosses the full internal backbone

This matrix is exactly the kind of target signal we eventually want to compare against learned leaf embeddings.

## Markdown cell 14
## 10. Training goal for the graph model

The core idea for the phylogeny modality is:

> **Leaf embeddings produced by the graph model should reflect the tree geometry.**

In other words:

- if two species are close on the tree, their embeddings should be similar
- if two species are far on the tree, their embeddings should be less similar

A common choice is to compare leaf embeddings using **cosine similarity**.

So the model should learn leaf representations such that:

- high cosine similarity ↔ small tree distance
- low cosine similarity ↔ large tree distance

## Markdown cell 15
## 11. Conceptual training pipeline

A graph-learning setup for this project could look like this:

1. load `edge_index`, `edge_weight`, and `nodes.csv`
2. create node features for all nodes
3. apply several GCN layers to propagate information through the tree
4. extract the final embeddings of the leaf nodes
5. compare leaf–leaf embedding relationships to the leaf–leaf tree distances
6. optimize a loss that encourages the two structures to agree

This notebook does not implement the training loop.
Its purpose is to make the graph objects and the learning objective concrete before model building starts.

## Markdown cell 16
## 12. Possible loss functions for relating embeddings to tree distances

There are several sensible ways to train the graph model so that the leaf embeddings reflect the phylogeny.

Below are the main options we have discussed conceptually.

## Markdown cell 17
### 12.1 Triplet loss

For an anchor species `a`, choose:

- a **positive** species `p` that is close on the tree
- a **negative** species `n` that is farther away on the tree

Then enforce:

\[
\text{sim}(a, p) > \text{sim}(a, n)
\]

often with a margin.

#### Intuition

This loss only cares about **relative ordering**:

- close relatives should be more similar than distant ones

#### Pros

- simple and intuitive
- directly encodes ranking
- robust to scale

#### Cons

- requires sampling triplets carefully
- can be noisy if triplet selection is weak
- uses only part of the available pairwise information at a time

## Markdown cell 18
### 12.2 Pairwise regression or correlation loss

Another idea is to compare all leaf pairs directly.

For example:

- compute cosine similarity for all leaf pairs
- compare that to negative tree distance
- use a loss such as MSE, Huber loss, or a correlation-based objective

#### Intuition

This tries to align the full pairwise structure more directly.

#### Pros

- uses all pairwise relationships in the batch
- conceptually straightforward

#### Cons

- absolute scales can be awkward
- cosine similarity and tree distance live on different numerical scales
- may be less stable than distribution-based losses

## Markdown cell 19
### 12.3 Anchor-wise KL loss over softmax distributions

This is the idea we discussed for the transformer species embeddings and it also applies naturally to GNN leaf embeddings.

For a given anchor leaf `i`:

1. compute embedding similarities from leaf `i` to all other leaves
2. convert those to a probability distribution with softmax
3. take the tree distances from leaf `i` to all other leaves
4. convert those to a target distribution with softmax over **negative** distances
5. minimize the KL divergence between the two distributions

#### Intuition

For each anchor species, the model should assign high probability mass to the same neighbors that are close on the tree.

#### Pros

- focuses on **relative neighborhood structure**
- uses all leaves in the batch at once
- does not require exact distance matching

#### Cons

- depends on temperature choices
- requires care with self-masking and scaling
- target distributions can become too sharp if distances are not transformed carefully

## Markdown cell 20
### Why softmax over negative tree distance?

Softmax gives higher probability to larger values.

But tree distance is a **dissimilarity**:

- small distance = close relatives
- large distance = far relatives

So before applying softmax, we flip the sign:

\[
\text{tree logits} = -d_{tree}
\]

Optionally, we can use a transformed version such as:

\[
-\log(1 + d_{tree})
\]

to compress very large distances and keep the target distribution better behaved.

## Markdown cell 21
## 13. What your first implementation could focus on

A good first GNN milestone would be:

1. load the graph export from preprocessing
2. define a simple node feature matrix
3. run 2–3 graph convolution layers
4. extract only the leaf embeddings
5. compute the pairwise cosine similarities between leaves
6. compare those to the tree-derived leaf distance matrix

The simplest first experiments would likely be:

- triplet loss
- pairwise correlation loss
- anchor-wise KL loss

All three are reasonable. The KL approach is especially attractive when we care about preserving local phylogenetic neighborhoods.

## Markdown cell 22
## 14. Summary

This toy notebook showed how to move from a phylogeny to a graph-learning view:

- a tree can be stored as `edge_index`, `edge_weight`, and `nodes.csv`
- those files define a sparse graph without needing a Newick parser
- a GCN combines **graph structure** and **node features**
- after message passing, each node has a learned embedding
- for this project, the main supervision target is on **leaf embeddings**
- the embedding geometry should reflect the pairwise distances in the tree

The next practical step is to replace the toy features with real features and build a small PyTorch Geometric prototype on top of the exported fish-tree graph.

