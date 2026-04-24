from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


from torch_geometric.nn import GCNConv

@dataclass
class Config:
    # Paths
    input_dir: Path = Path("output")
    output_dir: Path = Path("output/gnn_baseline")

    # Reproducibility
    seed: int = 42

    # Features
    use_log_branch_sum: bool = True

    # Training
    hidden_dim: int = 64
    out_dim: int = 32
    dropout: float = 0.2
    lr: float = 1e-2
    weight_decay: float = 1e-4
    num_epochs: int = 200

    # Triplet sampling
    train_triplets_per_epoch: int = 5000
    val_triplets: int = 10000
    margin: float = 0.2
    positive_k: int = 20
    negative_quantile: float = 0.7

    # KL loss
    lambda_kl: float = 5.0 
    kl_num_anchors_per_epoch: int = 256
    temp_embed: float = 0.5 
    temp_tree: float = 0.5 
    kl_neighbor_k: int = 16

    # Evaluation
    eval_pair_sample_size: int = 50000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GraphFeatures:
    def __init__(
        self,
        edge_index_np,
        edge_weight_np,
        nodes,
        species_index_df,
        branch_dist_full,
        cfg,
    ):
        self.edge_index = edge_index_np
        self.edge_weight = edge_weight_np
        self.nodes = nodes
        self.species_index_df = species_index_df
        self.branch_dist_full = branch_dist_full
        self.cfg = cfg

        self.num_nodes = len(nodes)
        self.is_leaf_mask = nodes["is_leaf"].astype(bool).to_numpy()

    # Internal helpers (unchanged)
    def build_undirected_adjacency(self):
        adj = {i: [] for i in range(self.num_nodes)}

        for (u, v), w in zip(self.edge_index.T, self.edge_weight):
            adj[int(u)].append((int(v), float(w)))

        return adj

    def compute_rooted_tree_features(self, adj, root=0):
        num_nodes = self.num_nodes
        is_leaf_mask = self.is_leaf_mask

        parent = np.full(num_nodes, -1, dtype=np.int64)
        depth_edges = np.full(num_nodes, -1, dtype=np.int32)
        depth_branch = np.zeros(num_nodes, dtype=np.float32)

        stack = [root]
        parent[root] = root
        depth_edges[root] = 0
        depth_branch[root] = 0.0

        order = []

        while stack:
            u = stack.pop()
            order.append(u)

            for v, w in adj[u]:
                if depth_edges[v] != -1:
                    continue
                parent[v] = u
                depth_edges[v] = depth_edges[u] + 1
                depth_branch[v] = depth_branch[u] + w
                stack.append(v)

        subtree_size = np.zeros(num_nodes, dtype=np.float32)

        for u in reversed(order):
            if is_leaf_mask[u]:
                subtree_size[u] = 1.0
            else:
                total = 0.0
                for v, _ in adj[u]:
                    if parent[v] == u:
                        total += subtree_size[v]
                subtree_size[u] = total

        return depth_edges.astype(np.float32), depth_branch, subtree_size


    # Exposed API (replaces script block)
    def build_features(self):
        src = self.edge_index[0]

        deg = np.bincount(src, minlength=self.num_nodes).astype(np.float32)
        weighted_deg = np.bincount(
            src, weights=self.edge_weight, minlength=self.num_nodes
        ).astype(np.float32)

        # Build rooted structural features from the tree
        adj = self.build_undirected_adjacency()

        depth_edges, depth_branch, subtree_size = self.compute_rooted_tree_features(adj)

        feature_cols = [
            self.nodes["is_leaf"].astype(np.float32).to_numpy(),
            deg,
            weighted_deg,
            depth_edges,
            depth_branch,
            subtree_size,
        ]

        feature_names = [
            "is_leaf",
            "degree",
            "weighted_degree",
            "depth_edges",
            "depth_branch",
            "subtree_size",
        ]

        if self.cfg.use_log_branch_sum:
            log_weighted_deg = np.log1p(weighted_deg).astype(np.float32)
            feature_cols.append(log_weighted_deg)
            feature_names.append("log_weighted_degree")

        bias = np.ones(self.num_nodes, dtype=np.float32)
        feature_cols.append(bias)
        feature_names.append("bias")

        X_np = np.stack(feature_cols, axis=1).astype(np.float32)

        X_df = pd.DataFrame(X_np, columns=feature_names)
        X_df.insert(0, "node_name", self.nodes["node_name"])

        print("\nNode feature preview:")
        print(X_df.head())

        return X_np, X_df
    
    def build_leaf_supervision(self):
        # Leaf extraction: leaves correspond to observed species
        leaf_mask_np = self.is_leaf_mask
        leaf_node_ids = self.nodes.loc[leaf_mask_np, "node_id"].to_numpy(dtype=np.int64)
        leaf_names = self.nodes.loc[leaf_mask_np, "node_name"].astype(str).to_numpy()

        n_leaves = len(leaf_node_ids)
        print(f"\nNumber of leaves: {n_leaves}")

        # Align species order
        # The preprocessing notebook wrote a species index table for the branch distance matrix.
        # We use it to ensure leaf order matches the branch distance matrix order.
        species_order = self.species_index_df["species"].astype(str).to_numpy()

        leaf_name_set = set(leaf_names.tolist())
        matrix_name_set = set(species_order.tolist())

        if leaf_name_set != matrix_name_set:
            missing_in_nodes = matrix_name_set - leaf_name_set
            missing_in_matrix = leaf_name_set - matrix_name_set
            raise ValueError(
                "Leaf species mismatch.\n"
                f"Missing in nodes: {list(missing_in_nodes)[:10]}\n"
                f"Missing in matrix: {list(missing_in_matrix)[:10]}"
            )

        # Reorder branch distance matrix to match the order of leaves in nodes.csv
        species_to_idx = {
            species: idx for idx, species in enumerate(species_order)
        }

        perm = np.array(
            [species_to_idx[name] for name in leaf_names],
            dtype=np.int64,
        )

        leaf_dist = self.branch_dist_full[np.ix_(perm, perm)].astype(np.float32)

        # Normalize for training stability
        max_dist = leaf_dist.max()
        if max_dist > 0:
            leaf_dist = leaf_dist / max_dist

        print("\nLeaf distance matrix loaded and aligned.")
        print(f"  shape: {leaf_dist.shape}")
        print(f"  min:   {leaf_dist.min():.6f}")
        print(f"  max:   {leaf_dist.max():.6f}")

        return {
            "leaf_node_ids": leaf_node_ids,
            "leaf_names": leaf_names,
            "leaf_dist": leaf_dist,
            "n_leaves": n_leaves,
        }

class TripletBuilder:
    def __init__(self, leaf_dist_matrix, cfg: Config):
        self.leaf_dist = leaf_dist_matrix
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.pos_pools, self.neg_pools = self.build_triplet_candidate_pools()

        print("\nTriplet candidate pools built.")
        print(f"  Example anchor 0 -> pos pool size: {len(self.pos_pools[0])}, neg pool size: {len(self.neg_pools[0])}")


    def build_triplet_candidate_pools(self):
        leaf_dist_matrix = self.leaf_dist
        positive_k = self.cfg.positive_k
        negative_quantile = self.cfg.negative_quantile

        n = leaf_dist_matrix.shape[0]
        pos_pools = []
        neg_pools = []

        for i in range(n):
            d = leaf_dist_matrix[i].copy()
            # Exclude self
            d[i] = np.inf

            # Positives = among nearest neighbors
            nearest_order = np.argsort(d)
            pos = nearest_order[:positive_k]
            pos = pos[np.isfinite(d[pos])]

            # Negatives = species farther than a chosen quantile
            d_self_zero = leaf_dist_matrix[i]
            threshold = np.quantile(d_self_zero[np.arange(n) != i], negative_quantile)
            neg = np.where(d_self_zero >= threshold)[0]
            neg = neg[neg != i]

            # Fallbacks for pathological cases
            if len(pos) == 0:
                pos = np.array([nearest_order[0]], dtype=np.int64)
            if len(neg) == 0:
                neg = np.array([nearest_order[-1]], dtype=np.int64)

            pos_pools.append(pos.astype(np.int64))
            neg_pools.append(neg.astype(np.int64))

        return pos_pools, neg_pools
    
    def sample_triplets(self, num_triplets):
        n = len(self.pos_pools)
        triplets = np.zeros((num_triplets, 3), dtype=np.int64)

        for t in range(num_triplets):
            i = self.rng.integers(0, n)
            j = self.rng.choice(self.pos_pools[i])
            k = self.rng.choice(self.neg_pools[i])
            triplets[t] = (i, j, k)

        return triplets

    def sample_kl_anchors(self):
        n_leaves = len(self.pos_pools)
        num_anchors = min(self.cfg.kl_num_anchors_per_epoch, n_leaves)

        return self.rng.choice(n_leaves, size=num_anchors, replace=False).astype(np.int64)
    

class WeightedGCN(nn.Module):
    """
    Weigthed Convolutional Graph Neural Network
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class ModelLoss:
    """
    Loss and evaluation utilities for GNN embeddings.
    """

    # Cosine distance
    @staticmethod
    def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(a, b, dim=-1)

    # Triplet loss
    @staticmethod
    def triplet_loss_from_leaf_embeddings(
        leaf_embeddings: torch.Tensor,
        triplets: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """
        leaf_embeddings: [n_leaves, d]
        triplets: [T, 3] in leaf-index space
        """
        anchor = leaf_embeddings[triplets[:, 0]]
        positive = leaf_embeddings[triplets[:, 1]]
        negative = leaf_embeddings[triplets[:, 2]]

        d_pos = ModelLoss.cosine_distance(anchor, positive)
        d_neg = ModelLoss.cosine_distance(anchor, negative)

        loss = F.relu(d_pos - d_neg + margin).mean()
        return loss

    # Triplet accuracy (eval only)
    @staticmethod
    @torch.no_grad()
    def triplet_accuracy_from_leaf_embeddings(
        leaf_embeddings: torch.Tensor,
        triplets: torch.Tensor,
    ) -> float:
        anchor = leaf_embeddings[triplets[:, 0]]
        positive = leaf_embeddings[triplets[:, 1]]
        negative = leaf_embeddings[triplets[:, 2]]

        d_pos = ModelLoss.cosine_distance(anchor, positive)
        d_neg = ModelLoss.cosine_distance(anchor, negative)

        return (d_pos < d_neg).float().mean().item()


    # Spearman correlation
    @staticmethod
    @torch.no_grad()
    def pairwise_spearman_correlation(
        leaf_embeddings: torch.Tensor,
        leaf_dist_matrix: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Compare embedding cosine similarity to negative tree distance.
        We sample random leaf pairs for speed instead of O(n^2).
        """
        z = F.normalize(leaf_embeddings, p=2, dim=1).cpu().numpy()
        n = z.shape[0]

        i_idx = rng.integers(0, n, size=sample_size)
        j_idx = rng.integers(0, n, size=sample_size)

        # Remove self-pairs
        mask = i_idx != j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]

        sims = np.sum(z[i_idx] * z[j_idx], axis=1)
        neg_dists = -leaf_dist_matrix[i_idx, j_idx]

        corr, _ = spearmanr(sims, neg_dists)
        return float(corr)


    # Anchor-wise KL loss
    @staticmethod
    def anchorwise_local_kl_loss(
        leaf_embeddings: torch.Tensor,
        leaf_dist_matrix: np.ndarray,
        anchor_indices: np.ndarray,
        neighbor_k: int = 64,
        temp_embed: float = 0.5,
        temp_tree: float = 0.5,
    ) -> torch.Tensor:
        """
        Local anchor-wise KL:
        For each anchor, compare only against k nearest leaves in tree space.

        This avoids full softmax over ~11k leaves.
        """
        device = leaf_embeddings.device
        z = F.normalize(leaf_embeddings, p=2, dim=1)

        total_kl = 0.0
        n_anchors_used = 0

        for anchor_idx in anchor_indices:
            anchor_idx = int(anchor_idx)

            # Tree distances from this anchor to all leaves
            dists = leaf_dist_matrix[anchor_idx].copy()
            # Exclude self
            dists[anchor_idx] = np.inf

            # Select k nearest neighbors in tree space
            nn_idx = np.argsort(dists)[:neighbor_k]

            # Safety check
            if len(nn_idx) == 0:
                continue

            # Predicted distribution
            sims_subset = torch.matmul(z[anchor_idx], z[nn_idx].T)
            pred_log_probs = F.log_softmax(sims_subset / temp_embed, dim=0)

            # Target distribution
            dists_subset = dists[nn_idx]

            # Standardize local distances so each anchor uses the relative spread
            # inside its own neighborhood, not the absolute raw scale.
            eps = 1e-8
            d_centered = dists_subset - dists_subset.min()
            d_scaled = d_centered / (dists_subset.std() + eps)

            tree_logits = torch.tensor(
                -d_scaled / temp_tree,
                dtype=torch.float32,
                device=device,
            )

            target_probs = F.softmax(tree_logits, dim=0)

            # KL divergence
            kl_val = F.kl_div(
                pred_log_probs,
                target_probs,
                reduction="batchmean",
            )

            total_kl = total_kl + kl_val
            n_anchors_used += 1

        if n_anchors_used == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)

        return total_kl / n_anchors_used
    

### Helper functions and global variables
CFG = Config()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_outputs(final_node_embeddings, final_leaf_embeddings, n_leaves, leaf_node_ids, leaf_names, history):
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(CFG.output_dir / "node_embeddings.npy", final_node_embeddings)
    np.save(CFG.output_dir / "leaf_embeddings.npy", final_leaf_embeddings)

    # Save leaf index mapping
    leaf_index_mapping = pd.DataFrame({
        "leaf_matrix_index": np.arange(n_leaves, dtype=np.int32),
        "node_id": leaf_node_ids.astype(np.int32),
        "species": leaf_names,
    })
    leaf_index_mapping.to_csv(CFG.output_dir / "leaf_index_mapping.csv", index=False)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(CFG.output_dir / "history.csv", index=False)

    # Save final metrics summary
    best_row = history_df.loc[history_df["val_triplet_accuracy"].idxmax()]
    metrics_summary = pd.DataFrame([{
        "best_epoch": int(best_row["epoch"]),
        "best_val_triplet_accuracy": float(best_row["val_triplet_accuracy"]),
        "spearman_at_best_epoch": float(best_row["spearman_similarity_vs_neg_distance"]),
    }])
    metrics_summary.to_csv(CFG.output_dir / "metrics_summary.csv", index=False)

    print("\nSaved outputs to:")
    print(f"  {CFG.output_dir / 'node_embeddings.npy'}")
    print(f"  {CFG.output_dir / 'leaf_embeddings.npy'}")
    print(f"  {CFG.output_dir / 'leaf_index_mapping.csv'}")
    print(f"  {CFG.output_dir / 'history.csv'}")
    print(f"  {CFG.output_dir / 'metrics_summary.csv'}")

    return history_df, metrics_summary

def visualize(history_df):
    plt.figure(figsize=(7, 4))

    plt.plot(history_df["epoch"], history_df["train_total_loss"], label="Train total loss")
    plt.plot(history_df["epoch"], history_df["train_triplet_loss"], label="Train triplet loss")
    plt.plot(history_df["epoch"], history_df["train_kl_loss"], label="Train KL loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG.output_dir / "loss_curve.png", dpi=200)
    plt.close()

    print(f"Saved: {CFG.output_dir / 'loss_curve.png'}")

def visualize_embeddings(final_leaf_embeddings):
    leaf_emb_2d = PCA(n_components=2).fit_transform(final_leaf_embeddings)

    plt.figure(figsize=(7, 6))
    plt.scatter(leaf_emb_2d[:, 0], leaf_emb_2d[:, 1], s=4, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Leaf Embeddings (PCA)")
    plt.tight_layout()
    plt.savefig(CFG.output_dir / "pca_leaf_embeddings.png", dpi=200)
    plt.close()

    print(f"Saved: {CFG.output_dir / 'pca_leaf_embeddings.png'}")

### Training loop
def main():
    
    set_seed(CFG.seed)
    device = torch.device(CFG.device)
    print(f"\nUsing device: {device}")

    ### Load input files
    edge_index_path = CFG.input_dir / "processed_fishtree_edge_index.npy"
    edge_weight_path = CFG.input_dir / "processed_fishtree_edge_weight.npy"
    nodes_path = CFG.input_dir / "processed_fishtree_nodes.csv"
    branch_distance_path = CFG.input_dir / "processed_fishtree_branch_distance.npy"
    species_index_path = CFG.input_dir / "processed_fishtree_species_index.csv"

    edge_index_np = np.load(edge_index_path)          # shape: [2, E]
    edge_weight_np = np.load(edge_weight_path)        # shape: [E]
    nodes = pd.read_csv(nodes_path)                   # node metadata
    branch_dist_full = np.load(branch_distance_path)  # shape: [n_species, n_species]
    species_index_df = pd.read_csv(species_index_path)

    print("Loaded files:")
    print(f"  edge_index:         {edge_index_np.shape}")
    print(f"  edge_weight:        {edge_weight_np.shape}")
    print(f"  nodes:              {nodes.shape}")
    print(f"  branch_dist_full:   {branch_dist_full.shape}")
    print(f"  species_index_df:   {species_index_df.shape}")

    ### Graph sanity checks
    required_node_cols = {"node_id", "node_name", "is_leaf", "species"}
    missing_cols = required_node_cols - set(nodes.columns)
    if missing_cols:
        raise ValueError(f"nodes.csv is missing required columns: {missing_cols}")

    num_nodes = len(nodes)
    if edge_index_np.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index_np.shape}")

    if edge_index_np.max() >= num_nodes or edge_index_np.min() < 0:
        raise ValueError("edge_index contains invalid node indices.")

    if len(edge_weight_np) != edge_index_np.shape[1]:
        raise ValueError("edge_weight length must match number of edges in edge_index.")

    if not nodes["node_id"].is_unique:
        raise ValueError("node_id values must be unique.")

    # Important: enforce ordering by node_id so arrays line up correctly with graph indices.
    nodes = nodes.sort_values("node_id").reset_index(drop=True)

    # Double-check node_id is exactly 0..num_nodes-1
    expected_ids = np.arange(num_nodes)
    if not np.array_equal(nodes["node_id"].to_numpy(), expected_ids):
        raise ValueError("node_id must match row order 0..N-1 after sorting.")

    print("\nGraph sanity checks passed.")
    print(f"  num_nodes: {num_nodes}")
    print(f"  num_edges (directed): {edge_index_np.shape[1]}")
    print(f"  num_leaf_nodes: {nodes['is_leaf'].sum()}")

    ### Build graph features
    feature_builder = GraphFeatures(
        edge_index_np=edge_index_np,
        edge_weight_np=edge_weight_np,
        nodes=nodes,
        species_index_df=species_index_df,
        branch_dist_full=branch_dist_full,
        cfg=CFG,
    )

    X_np, X_df = feature_builder.build_features()

    leaf_info = feature_builder.build_leaf_supervision()

    leaf_node_ids = leaf_info["leaf_node_ids"]
    leaf_names = leaf_info["leaf_names"]
    leaf_dist = leaf_info["leaf_dist"]
    n_leaves = leaf_info["n_leaves"]

    ### Sample triplets
    rng = np.random.default_rng(CFG.seed)
    trp = TripletBuilder(
    leaf_dist_matrix=leaf_dist,
    cfg=CFG
    )

    # Fixed validation triplets for monitoring
    val_triplets_np = trp.sample_triplets(
    num_triplets=CFG.val_triplets,
    )
    val_triplets = torch.tensor(val_triplets_np, dtype=torch.long, device=device)

    ### Instantiate GNN
    # Convert inputs to tensors
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32, device=device)
    leaf_node_ids_t = torch.tensor(leaf_node_ids, dtype=torch.long, device=device)

    # Model, optimizer and loss
    model = WeightedGCN(
        in_dim=X.shape[1],
        hidden_dim=CFG.hidden_dim,
        out_dim=CFG.out_dim,
        dropout=CFG.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    loss_fn = ModelLoss()

    ### Training loop
    history = []
    best_val_acc = -float("inf")
    best_state = None

    for epoch in range(1, CFG.num_epochs + 1):

        model.train()
        optimizer.zero_grad()

        # Sample batch
        train_triplets_np = trp.sample_triplets(
            num_triplets=CFG.train_triplets_per_epoch,
        )

        train_triplets = torch.tensor(train_triplets_np, dtype=torch.long, device=device)
        kl_anchor_indices = trp.sample_kl_anchors()

        # Forward pass
        node_embeddings = model(X, edge_index, edge_weight=edge_weight)
        leaf_embeddings = node_embeddings[leaf_node_ids_t]


        # Losses
        triplet_loss = loss_fn.triplet_loss_from_leaf_embeddings(
            leaf_embeddings,
            train_triplets,
            margin=CFG.margin,
        )

        kl_loss = loss_fn.anchorwise_local_kl_loss(
            leaf_embeddings=leaf_embeddings,
            leaf_dist_matrix=leaf_dist,
            anchor_indices=kl_anchor_indices,
            neighbor_k=CFG.kl_neighbor_k,
            temp_embed=CFG.temp_embed,
            temp_tree=CFG.temp_tree,
        )

        loss = triplet_loss + CFG.lambda_kl * kl_loss

        loss.backward()
        optimizer.step()


        # Validation
        model.eval()
        with torch.no_grad():
            node_embeddings_eval = model(X, edge_index, edge_weight=edge_weight)
            leaf_embeddings_eval = node_embeddings_eval[leaf_node_ids_t]

            val_acc = loss_fn.triplet_accuracy_from_leaf_embeddings(
                leaf_embeddings_eval,
                val_triplets,
            )

            spearman = loss_fn.pairwise_spearman_correlation(
                leaf_embeddings_eval,
                leaf_dist,
                CFG.eval_pair_sample_size,
                rng,
            )

        # History
        history.append({
            "epoch": epoch,
            "train_total_loss": float(loss.item()),
            "train_triplet_loss": float(triplet_loss.item()),
            "train_kl_loss": float(kl_loss.item()),
            "val_triplet_accuracy": val_acc,
            "spearman_similarity_vs_neg_distance": spearman,
        })


        # Keep best model by validation triplet accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }


        # Logging
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"triplet={triplet_loss.item():.4f} | "
                f"kl={kl_loss.item():.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"spearman={spearman:.4f}"
            )


    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        final_node_embeddings = model(X, edge_index, edge_weight=edge_weight).cpu().numpy()
        final_leaf_embeddings = final_node_embeddings[leaf_node_ids]

    # Save outputs
    history_df, metrics_summary = save_outputs(final_node_embeddings, final_leaf_embeddings, n_leaves, leaf_node_ids, leaf_names, history)

    # Visualize loss curve
    visualize(history_df)

    # Visualize final embeddings
    visualize_embeddings(final_leaf_embeddings)

    # Final summary
    print("\nFinal summary:")
    print(metrics_summary.to_string(index=False))

    print("\nDone.")

if __name__ == "__main__":
    main()