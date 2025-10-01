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


UNIT_FEAT_DIM = 9  # [7 base + is_ally + is_alive]

@torch.no_grad()
def nodes_from_state_batch(
    state_batch: torch.Tensor,              # [B, D_state]
    feature_names: List[str],               # same list you already have
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized node feature builder for a batch of state vectors.

    Returns:
        x:         [B, N, 9]  allies (A) first then enemies (E), N=A+E
        ally_mask: [N]        1 for allies, 0 for enemies (same for all in batch)
                               (repeat it yourself if you prefer [B*N] flat later)
    """
    if isinstance(device, str):
        device = torch.device(device)
    assert state_batch.ndim == 2, f"state_batch must be [B,D], got {tuple(state_batch.shape)}"
    B, D = state_batch.shape
    state_batch = state_batch.to(torch.float32).to(device)

    fmap = _index_map(feature_names)
    ally_ids  = _collect_ids(feature_names, "ally")    # e.g., [0,1,2,3,4]
    enemy_ids = _collect_ids(feature_names, "enemy")   # e.g., [0,1,2,3,4]
    A = len(ally_ids)
    E = len(enemy_ids)
    N = A + E

    # Helper: gather one attribute over a list of unit ids -> [B, U]
    def gather(side: str, key_tmpl: str, ids: List[int]) -> torch.Tensor:
        outs = []
        for i in ids:
            name = key_tmpl.format(i=i)
            j = fmap.get(name, -1)
            if j >= 0:
                outs.append(state_batch[:, j])
            else:
                outs.append(torch.zeros(B, dtype=torch.float32, device=device))
        return torch.stack(outs, dim=1) if ids else torch.zeros(B, 0, dtype=torch.float32, device=device)

    # Allies (A,U= A): build 7 base channels
    a_h   = gather("ally",  "ally_health_{i}",        ally_ids)   # [B,A]
    a_rx  = gather("ally",  "ally_relative_x_{i}",    ally_ids)
    a_ry  = gather("ally",  "ally_relative_y_{i}",    ally_ids)
    a_sh  = gather("ally",  "ally_shield_{i}",        ally_ids)
    a_t0  = gather("ally",  "ally_unit_type_{i}_bit_0", ally_ids)
    a_t1  = gather("ally",  "ally_unit_type_{i}_bit_1", ally_ids)
    a_t2  = gather("ally",  "ally_unit_type_{i}_bit_2", ally_ids)

    # Enemies (E): 7 base channels
    e_h   = gather("enemy", "enemy_health_{i}",        enemy_ids) # [B,E]
    e_rx  = gather("enemy", "enemy_relative_x_{i}",    enemy_ids)
    e_ry  = gather("enemy", "enemy_relative_y_{i}",    enemy_ids)
    e_sh  = gather("enemy", "enemy_shield_{i}",        enemy_ids)
    e_t0  = gather("enemy", "enemy_unit_type_{i}_bit_0", enemy_ids)
    e_t1  = gather("enemy", "enemy_unit_type_{i}_bit_1", enemy_ids)
    e_t2  = gather("enemy", "enemy_unit_type_{i}_bit_2", enemy_ids)

    # Compose allies/enemies blocks: [B,A,7] and [B,E,7]
    allies7 = torch.stack([a_h, a_rx, a_ry, a_sh, a_t0, a_t1, a_t2], dim=-1) if A > 0 else \
              torch.zeros(B, 0, 7, dtype=torch.float32, device=device)
    enemies7 = torch.stack([e_h, e_rx, e_ry, e_sh, e_t0, e_t1, e_t2], dim=-1) if E > 0 else \
               torch.zeros(B, 0, 7, dtype=torch.float32, device=device)

    # Flags
    a_is_ally  = torch.ones(B, A, 1, dtype=torch.float32, device=device)
    e_is_ally  = torch.zeros(B, E, 1, dtype=torch.float32, device=device)

    a_is_alive = ((a_h + a_sh) > 0).float().unsqueeze(-1) if A > 0 else torch.zeros(B, 0, 1, device=device)
    e_is_alive = ((e_h + e_sh) > 0).float().unsqueeze(-1) if E > 0 else torch.zeros(B, 0, 1, device=device)

    allies9  = torch.cat([allies7,  a_is_ally, a_is_alive], dim=-1)  # [B,A,9]
    enemies9 = torch.cat([enemies7, e_is_ally, e_is_alive], dim=-1)  # [B,E,9]

    # Final [B, N, 9]
    X = torch.cat([allies9, enemies9], dim=1)  # [B, N, 9]

    # ally_mask for a single graph (same across batch)
    ally_mask = torch.cat([
        torch.ones(A, dtype=torch.float32, device=device),
        torch.zeros(E, dtype=torch.float32, device=device)
    ], dim=0)  # [N]

    return X, ally_mask


@torch.no_grad()
def build_graphs_from_state_batch(
    state_batch: torch.Tensor,              # [B, D_state]
    feature_names: List[str],
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    PyG-compatible graphs for a whole batch of state vectors.
    Returns a dict with:
      x:         [B*N, 9]
      edge_index:[2, B*E]  fully-connected per-graph
      batch:     [B*N]     graph ids
      ally_mask: [B*N]     1 for allies, 0 for enemies
    """
    if isinstance(device, str):
        device = torch.device(device)

    X, ally_mask_per_graph = nodes_from_state_batch(state_batch, feature_names, device=device)  # [B,N,9], [N]
    B, N, _ = X.shape

    # Flatten node features
    x = X.reshape(B * N, UNIT_FEAT_DIM)  # [B*N, 9]

    # ally_mask repeated for the batch
    ally_mask = ally_mask_per_graph.repeat(B)  # [B*N]

    # batch vector
    batch_vec = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.long), repeats=N)  # [B*N]

    # fully-connected edges per-graph with offsets
    base_edge = _fc_edge_index(N).to(device)  # [2, N*(N-1)]
    E = base_edge.size(1)
    offs = (torch.arange(B, device=device) * N).view(1, -1, 1)     # [1,B,1]
    edge_index = (base_edge.unsqueeze(1) + offs).reshape(2, B * E) # [2, B*E]

    return {
        "x": x,
        "edge_index": edge_index,
        "batch": batch_vec,
        "ally_mask": ally_mask,
    }