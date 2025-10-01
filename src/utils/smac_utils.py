# ---- at top of smac_graph_dataset_state.py ----
import re
from typing import List, Dict, Tuple

import torch

ALLY_KEYS = (
    "ally_health_{i}", "ally_relative_x_{i}", "ally_relative_y_{i}",
    "ally_shield_{i}", "ally_unit_type_{i}_bit_0", "ally_unit_type_{i}_bit_1", "ally_unit_type_{i}_bit_2"
)
ENEMY_KEYS = (
    "enemy_health_{i}", "enemy_relative_x_{i}", "enemy_relative_y_{i}",
    "enemy_shield_{i}", "enemy_unit_type_{i}_bit_0", "enemy_unit_type_{i}_bit_1", "enemy_unit_type_{i}_bit_2"
)

state_features = ['ally_health_0', 'ally_cooldown_0', 'ally_relative_x_0', 'ally_relative_y_0', 'ally_shield_0',
                  'ally_unit_type_0_bit_0', 'ally_unit_type_0_bit_1', 'ally_unit_type_0_bit_2', 'ally_health_1',
                  'ally_cooldown_1', 'ally_relative_x_1', 'ally_relative_y_1', 'ally_shield_1',
                  'ally_unit_type_1_bit_0', 'ally_unit_type_1_bit_1', 'ally_unit_type_1_bit_2', 'ally_health_2',
                  'ally_cooldown_2', 'ally_relative_x_2', 'ally_relative_y_2', 'ally_shield_2',
                  'ally_unit_type_2_bit_0', 'ally_unit_type_2_bit_1', 'ally_unit_type_2_bit_2', 'ally_health_3',
                  'ally_cooldown_3', 'ally_relative_x_3', 'ally_relative_y_3', 'ally_shield_3',
                  'ally_unit_type_3_bit_0', 'ally_unit_type_3_bit_1', 'ally_unit_type_3_bit_2', 'ally_health_4',
                  'ally_cooldown_4', 'ally_relative_x_4', 'ally_relative_y_4', 'ally_shield_4',
                  'ally_unit_type_4_bit_0', 'ally_unit_type_4_bit_1', 'ally_unit_type_4_bit_2', 'enemy_health_0',
                  'enemy_relative_x_0', 'enemy_relative_y_0', 'enemy_shield_0', 'enemy_unit_type_0_bit_0',
                  'enemy_unit_type_0_bit_1', 'enemy_unit_type_0_bit_2', 'enemy_health_1', 'enemy_relative_x_1',
                  'enemy_relative_y_1', 'enemy_shield_1', 'enemy_unit_type_1_bit_0', 'enemy_unit_type_1_bit_1',
                  'enemy_unit_type_1_bit_2', 'enemy_health_2', 'enemy_relative_x_2', 'enemy_relative_y_2',
                  'enemy_shield_2', 'enemy_unit_type_2_bit_0', 'enemy_unit_type_2_bit_1', 'enemy_unit_type_2_bit_2',
                  'enemy_health_3', 'enemy_relative_x_3', 'enemy_relative_y_3', 'enemy_shield_3',
                  'enemy_unit_type_3_bit_0', 'enemy_unit_type_3_bit_1', 'enemy_unit_type_3_bit_2', 'enemy_health_4',
                  'enemy_relative_x_4', 'enemy_relative_y_4', 'enemy_shield_4', 'enemy_unit_type_4_bit_0',
                  'enemy_unit_type_4_bit_1', 'enemy_unit_type_4_bit_2', 'ally_last_action_0_action_0',
                  'ally_last_action_0_action_1', 'ally_last_action_0_action_2', 'ally_last_action_0_action_3',
                  'ally_last_action_0_action_4', 'ally_last_action_0_action_5', 'ally_last_action_0_action_6',
                  'ally_last_action_0_action_7', 'ally_last_action_0_action_8', 'ally_last_action_0_action_9',
                  'ally_last_action_0_action_10', 'ally_last_action_1_action_0', 'ally_last_action_1_action_1',
                  'ally_last_action_1_action_2', 'ally_last_action_1_action_3', 'ally_last_action_1_action_4',
                  'ally_last_action_1_action_5', 'ally_last_action_1_action_6', 'ally_last_action_1_action_7',
                  'ally_last_action_1_action_8', 'ally_last_action_1_action_9', 'ally_last_action_1_action_10',
                  'ally_last_action_2_action_0', 'ally_last_action_2_action_1', 'ally_last_action_2_action_2',
                  'ally_last_action_2_action_3', 'ally_last_action_2_action_4', 'ally_last_action_2_action_5',
                  'ally_last_action_2_action_6', 'ally_last_action_2_action_7', 'ally_last_action_2_action_8',
                  'ally_last_action_2_action_9', 'ally_last_action_2_action_10', 'ally_last_action_3_action_0',
                  'ally_last_action_3_action_1', 'ally_last_action_3_action_2', 'ally_last_action_3_action_3',
                  'ally_last_action_3_action_4', 'ally_last_action_3_action_5', 'ally_last_action_3_action_6',
                  'ally_last_action_3_action_7', 'ally_last_action_3_action_8', 'ally_last_action_3_action_9',
                  'ally_last_action_3_action_10', 'ally_last_action_4_action_0', 'ally_last_action_4_action_1',
                  'ally_last_action_4_action_2', 'ally_last_action_4_action_3', 'ally_last_action_4_action_4',
                  'ally_last_action_4_action_5', 'ally_last_action_4_action_6', 'ally_last_action_4_action_7',
                  'ally_last_action_4_action_8', 'ally_last_action_4_action_9', 'ally_last_action_4_action_10']


UNIT_FEAT_DIM = 9  # 7 base features + is_ally + is_alive
_feat_order = ("health", "rel_x", "rel_y", "shield", "type_b0", "type_b1", "type_b2")


def _fc_edge_index(n: int) -> torch.Tensor:
    if n <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    idx = torch.arange(n, dtype=torch.long)
    src = idx.repeat_interleave(n - 1)
    dst = torch.stack([torch.cat([idx[:i], idx[i + 1:]]) for i in range(n)], dim=0).reshape(-1)
    return torch.stack([src, dst], dim=0)


_num_re = re.compile(r"(\d+)")


def _index_map(feature_names: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def _collect_ids(names: List[str], side: str) -> List[int]:
    ids = []
    prefix = f"{side}_health_"
    for n in names:
        if n.startswith(prefix):
            m = _num_re.search(n)
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def _fc_edge_index(n: int) -> torch.Tensor:
    if n <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    idx = torch.arange(n, dtype=torch.long)
    src = idx.repeat_interleave(n - 1)
    dst = torch.stack([torch.cat([idx[:i], idx[i + 1:]]) for i in range(n)], dim=0).reshape(-1)
    return torch.stack([src, dst], dim=0)


# ---- build one unit feature (now 9 dims) ----
def _build_unit(vec: torch.Tensor, fmap: Dict[str, int], side: str, i: int) -> torch.Tensor:
    """
    Returns a 9-dim node feature in this order:
      [health, rel_x, rel_y, shield, type_b0, type_b1, type_b2, is_ally, is_alive]
    """
    tpl = ALLY_KEYS if side == "ally" else ENEMY_KEYS
    vals = []
    # base 7 features (fill missing with 0)
    for k in [t.format(i=i) for t in tpl]:
        j = fmap.get(k)
        vals.append(0.0 if j is None else float(vec[j]))
    # flags
    is_ally = 1.0 if side == "ally" else 0.0
    # infer alive from (health + shield) > 0
    h_idx = fmap.get(f"{side}_health_{i}")
    s_idx = fmap.get(f"{side}_shield_{i}")
    hp = float(vec[h_idx]) if h_idx is not None else 0.0
    sh = float(vec[s_idx]) if s_idx is not None else 0.0
    is_alive = 1.0 if (hp + sh) > 0.0 else 0.0

    vals.extend([is_ally, is_alive])
    return torch.tensor(vals, dtype=torch.float32)  # [9]


# ---- convert one state vector to a PyG graph (updated to 9-dim x) ----
def nodes_from_state_vector(vec: torch.Tensor, feature_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build node features from ONE flat state vector (shape [D]) and names.

    Returns:
        x: [num_nodes, 9]  (allies first, then enemies)
        ally_mask: [num_nodes]  (1 for ally, 0 for enemy)
    """
    assert vec.dim() == 1, f"expected 1D state vector, got {tuple(vec.shape)}"
    fmap = _index_map(feature_names)

    ally_ids = _collect_ids(feature_names, "ally")
    enemy_ids = _collect_ids(feature_names, "enemy")

    nodes: List[torch.Tensor] = []
    for i in ally_ids:
        nodes.append(_build_unit(vec, fmap, "ally", i))
    for j in enemy_ids:
        nodes.append(_build_unit(vec, fmap, "enemy", j))

    if len(nodes) == 0:
        return torch.zeros((0, UNIT_FEAT_DIM), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)

    x = torch.stack(nodes, dim=0)  # [N, 9]
    ally_mask = torch.cat([
        torch.ones(len(ally_ids), dtype=torch.float32),
        torch.zeros(len(enemy_ids), dtype=torch.float32)
    ], dim=0)  # [N]
    return x, ally_mask

