# file: smacv2_graph_contrastive_model.py
from __future__ import annotations
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GraphNorm, global_add_pool, global_max_pool


def _as_batch(payload: Union[Batch, Data, Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Accept PyG Batch/Data or dict and return a uniform dict."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (Batch, Data)):
        out = {"x": payload.x, "edge_index": payload.edge_index}
        if hasattr(payload, "batch"): out["batch"] = payload.batch
        if hasattr(payload, "ally_mask"): out["ally_mask"] = payload.ally_mask
        return out
    raise TypeError(f"Unsupported input type: {type(payload)}")


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int, p_drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GATBlock(nn.Module):
    def __init__(self, d: int, heads: int = 3, p_drop: float = 0.0):
        super().__init__()
        self.conv = GATv2Conv(in_channels=d, out_channels=d // heads, heads=heads, add_self_loops=False)
        self.norm = GraphNorm(d)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        h = self.conv(x, edge_index)      # [num_nodes, d]
        # h = (h)
        # h = self.norm(h, batch)
        # h = self.drop(h)
        return h


class SMACV2GraphContrastiveModel(nn.Module):
    """
    Contrastive encoder for SMACv2 graphs.

    Nodes: allies + enemies; node feature = [hp, shield].
    Objective graph: same topology, but ENEMY nodes have features zeroed (hp=0, shield=0).

    Forward returns:
      final_emb        : 32-D embedding from [cur_pool | obj_pool]
      final_emb_obj    : 32-D embedding from [obj_pool | obj_pool] (useful as a "target" head)
      cur_pool, obj_pool : 16-D pooled graph embeddings (diagnostics / auxiliary losses)
    """
    def __init__(self, enemy_feature_idx,
                 device: str | torch.device = "cpu",
                 d_node_in: int = 92,   # input node feature dim (e.g. [hp, shield])
                 d_node: int = 64,
                 d_pool: int = 16,
                 d_out: int = 32,
                 heads: int = 3,
                 p_drop: float = 0.1,
                 ):
        super().__init__()
        self.device = torch.device(device)

        # Node encoder (hp, shield) -> d_node
        self.node_enc = MLP(d_node_in, d_node, d_hidden=64, p_drop=p_drop)

        # Two residual GAT blocks
        self.g1 = GATBlock(d_node, heads=heads, p_drop=p_drop)
        # self.g2 = GATBlock(d_node, heads=heads, p_drop=p_drop)

        # Pool projections
        self.pool_proj = MLP((d_node//heads) * heads, d_pool, d_hidden=64, p_drop=p_drop)

        # Metric heads
        self.metric_head = MLP(d_pool * 2, d_out, d_hidden=64, p_drop=p_drop)
        self.enemy_feature_idx = tuple(enemy_feature_idx) if enemy_feature_idx is not None else None

        self.to(self.device)

    def _make_objective_features(self, x: torch.Tensor, ally_mask: torch.Tensor) -> torch.Tensor:
        # x dim is 9: [health, rel_x, rel_y, shield, type_b0, type_b1, type_b2, is_ally, is_alive]
        x_obj = x.clone()
        m = ally_mask if ally_mask.dtype == torch.bool else (ally_mask > 0.5)
        enemy_idx = (~m).nonzero(as_tuple=True)[0]
        if enemy_idx.numel() == 0:
            return x_obj
        # zero hp & shield
        x_obj[enemy_idx, 0] = 0.0  # health
        x_obj[enemy_idx, 3] = 0.0  # shield
        # set is_alive to 0 (keep is_ally=0 untouched)
        x_obj[enemy_idx, 8] = 0.0  # is_alive
        # optional: remove spatial/type info to simulate "gone"
        x_obj[enemy_idx, 1:3] = 0.0      # rel_x, rel_y
        x_obj[enemy_idx, 4:7] = 0.0      # type bits
        return x_obj

    def _encode_graph(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode nodes -> message passing -> pooled graph embedding."""
        h = self.node_enc(x)

        h = self.g1(h, edge_index, batch)
        # h = h + h1  # residual

        # h2 = self.g2(h, edge_index, batch)
        # h = h + h2  # residual

        pooled = global_add_pool(h, batch)     # [num_graphs, d_node]
        pooled = self.pool_proj(pooled)        # [num_graphs, d_pool]
        return h, pooled

    def forward(self, batch_like: Union[Batch, Data, Dict[str, Tensor]]):
        """
        Accepts a PyG Batch/Data or a dict with keys: x, edge_index, batch, ally_mask.
        """
        B = _as_batch(batch_like)
        x: Tensor = B["x"].to(self.device)
        edge_index: Tensor = B["edge_index"].to(self.device)
        batch: Tensor = B["batch"].to(self.device)
        ally_mask: Tensor = B.get("ally_mask", torch.ones(x.size(0), device=self.device)).to(self.device)

        # Current graph embedding
        _, cur_pool = self._encode_graph(x, edge_index, batch)

        # Objective graph: zero enemy features
        x_obj = self._make_objective_features(x, ally_mask)
        _, obj_pool = self._encode_graph(x_obj, edge_index, batch)

        # Final embeddings for contrastive objectives
        final_emb     = self.metric_head(torch.cat([cur_pool, obj_pool], dim=-1))
        final_emb_obj = self.metric_head(torch.cat([obj_pool, obj_pool], dim=-1))

        return final_emb, final_emb_obj, cur_pool, obj_pool
