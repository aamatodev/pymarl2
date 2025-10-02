#!/usr/bin/env python3
"""
Generate a SMACv2 dataset using a trained QMIX policy from PyMARL2.

Usage (example):
  python tools/make_dataset_from_policy.py \
    --checkpoint /path/to/models/td_model.pt \
    --config_name=smacv2 --map_name=MMM2 \
    --n_episodes 100 --out dataset_mmm2_qmix

Notes
- Assumes you're inside (or PYTHONPATH includes) the PyMARL2 repo so that `src.*` imports work.
- Works with QMIX (value-based) checkpoints saved by PyMARL2.
- If you want Q-values per agent, add `--save_q True` (slower & larger).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# --- PyMARL2 internals (paths must resolve) ---
from src.envs import REGISTRY as env_REGISTRY             # environment registry
from src.controllers import REGISTRY as mac_REGISTRY      # e.g., "basic_mac"
from src.components.episode_buffer import EpisodeBatch
from src.utils.logging import get_logger
from src.utils.dict2namedtuple import convert
from src.main import _get_config  # utility that builds config from defaults (present in PyMARL2)

log = get_logger()

# ---------------------------
# Helpers
# ---------------------------

def _np_concat(lst: List[np.ndarray]) -> np.ndarray:
    if len(lst) == 0:
        return np.array([])
    return np.concatenate(lst, axis=0)

def _safe_tolist(x: np.ndarray):
    return x.tolist() if isinstance(x, np.ndarray) else x

# ---------------------------
# Rollout & Dataset Writer
# ---------------------------

def rollout_and_collect(args) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Build env/MAC like evaluation, then collect episodes.
    Returns (dataset_arrs, manifest)
    """

    # ---------------------------
    # 1) Build full config (env, scheme, groups, args, etc.)
    # ---------------------------
    # Get PyMARL2 config object (expects defaults present in src/config/...)
    # We mimic `python run.py --config=qmix --env-config=smacv2 ...`
    # Here we assume trained checkpoint is QMIX and env is smacv2.
    pymarl_cfg = _get_config()  # returns a dict with defaults; we will patch
    cfg = convert(pymarl_cfg)   # dict -> nested namedtuples (PyMARL2 convention)

    # Patch key runtime bits from CLI
    # Important minimal fields:
    #   cfg.env, cfg.env_args, cfg.agent, cfg.mac, cfg.runner, cfg.t_max
    # Most are already in defaults; we override what we need.
    cfg.env = "smacv2"
    cfg.env_args = dict(getattr(cfg, "env_args", {}))
    cfg.env_args["map_name"] = args.map_name
    cfg.test_nepisode = args.n_episodes
    cfg.t_max = getattr(cfg, "t_max", 1000000)

    # Disable exploration while generating dataset
    cfg.test_greedy = True
    cfg.evaluation = True
    cfg.batch_size_run = 1  # one episode at a time for dataset clarity

    # ---------------------------
    # 2) Make environment
    # ---------------------------
    env = env_REGISTRY[cfg.env](**cfg.env_args)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    state_shape = env_info["state_shape"]
    obs_shape = env_info["obs_shape"]

    # ---------------------------
    # 3) Build scheme/groups (as EpisodeBatch would expect)
    # ---------------------------
    scheme = {
        "state": {"vshape": state_shape, "group": None},
        "obs": {"vshape": obs_shape, "group": "agents"},
        "avail_actions": {"vshape": n_actions, "group": "agents", "dtype": torch.long},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {
        "agents": n_agents
    }
    preprocess = {}

    # EpisodeBatch here is not strictly required for dataset writing, but MAC expects a certain interface.
    batch = EpisodeBatch(scheme, groups, 1, cfg.episode_limit + 1,
                         preprocess=preprocess, device="cpu")

    # ---------------------------
    # 4) Build MAC (controller) and load checkpoint
    # ---------------------------
    mac = mac_REGISTRY[cfg.mac](scheme, groups, cfg)

    # Load checkpoint (expects standard PyMARL2 save dict)
    log.info(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Typical keys: "agent", "mixer", maybe optimizer states as well
    mac.load_state(ckpt["agent"])

    mac.eval()  # no exploration
    for p in mac.parameters():
        p.requires_grad = False

    # ---------------------------
    # 5) Rollout
    # ---------------------------
    episode_limit = env_info["episode_limit"]
    rng = np.random.default_rng(args.seed if args.seed is not None else 12345)

    # Storage lists (we'll concat later)
    all_states = []
    all_obs = []
    all_avail = []
    all_actions = []
    all_rewards = []
    all_term = []
    all_ep_ids = []
    all_agent_mask = []
    all_qvals = [] if args.save_q else None

    ep_counter = 0

    while ep_counter < args.n_episodes:
        # reset env
        env.reset()
        ep_states = []
        ep_obs = []
        ep_avail = []
        ep_actions = []
        ep_rewards = []
        ep_term = []
        ep_agent_mask = []
        ep_qvals = [] if args.save_q else None

        t = 0
        terminated = False

        mac.init_hidden(batch.batch_size)

        while not terminated and t < episode_limit:
            state = env.get_state()                     # [state_shape]
            obs = env.get_obs()                         # list of length n_agents, each [obs_shape]
            avail_actions = env.get_avail_actions()     # list len n_agents, one-hot or mask over n_actions

            # Convert to episodebatch-like tensors for MAC.select_actions
            # Build one-step batch tensors
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1, S]
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)                   # [1, n_agents, O]
            avail_t = torch.from_numpy(np.asarray(avail_actions, dtype=np.int64)).unsqueeze(0)         # [1, n_agents, A]

            # Fill into EpisodeBatch for this timestep
            batch.data.transition_data["state"][:, t] = state_t
            batch.data.transition_data["obs"][:, t] = obs_t
            batch.data.transition_data["avail_actions"][:, t] = avail_t

            # Greedy action selection
            selected_actions = mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)  # [1, n_agents]
            act_np = selected_actions.squeeze(0).cpu().numpy()  # [n_agents]

            # Optionally compute Q-values
            if args.save_q:
                # forward through mac to get q-values (depends on MAC implementation API)
                mac_out = mac.forward(batch, t=t)  # [1, n_agents, n_actions]
                qvals = mac_out.squeeze(0).detach().cpu().numpy()  # [n_agents, n_actions]
            else:
                qvals = None

            # Step env
            reward, terminated, env_info_step = env.step(act_np)
            # Note: SMACv2 returns reward, terminated, info (may include 'episode_limit' or win info)

            # Log per-step
            ep_states.append(state)                                            # [S]
            ep_obs.append(np.asarray(obs, dtype=np.float32))                   # [n_agents, O]
            ep_avail.append(np.asarray(avail_actions, dtype=np.int64))         # [n_agents, A]
            ep_actions.append(act_np.astype(np.int64)[None, :])                # [1, n_agents]
            ep_rewards.append(np.asarray([reward], dtype=np.float32))          # [1]
            ep_term.append(np.asarray([terminated], dtype=np.uint8))           # [1]
            ep_agent_mask.append(np.ones((n_agents,), dtype=np.uint8))         # alive mask; refine if needed
            if args.save_q:
                ep_qvals.append(qvals[None, ...])                              # [1, n_agents, n_actions]

            t += 1

        # Concat episode along time
        ep_T = len(ep_rewards)  # number of transitions collected
        ep_id_arr = np.full((ep_T,), ep_counter, dtype=np.int32)

        all_states.append(np.stack(ep_states, axis=0))                        # [T, S]
        all_obs.append(np.stack(ep_obs, axis=0))                              # [T, n_agents, O]
        all_avail.append(np.stack(ep_avail, axis=0))                          # [T, n_agents, A]
        all_actions.append(np.concatenate(ep_actions, axis=0))                # [T, n_agents]
        all_rewards.append(np.concatenate(ep_rewards, axis=0))                # [T, 1]
        all_term.append(np.concatenate(ep_term, axis=0))                      # [T, 1]
        all_ep_ids.append(ep_id_arr)                                          # [T]
        all_agent_mask.append(np.stack(ep_agent_mask, axis=0))                # [T, n_agents]
        if args.save_q:
            all_qvals.append(np.concatenate(ep_qvals, axis=0))                # [T, n_agents, A]

        ep_counter += 1

    # ---------------------------
    # 6) Final stacking
    # ---------------------------
    dataset = {
        "state": np.concatenate(all_states, axis=0),           # [N, S]
        "obs": np.concatenate(all_obs, axis=0),                # [N, n_agents, O]
        "avail_actions": np.concatenate(all_avail, axis=0),    # [N, n_agents, A]
        "actions": np.concatenate(all_actions, axis=0),        # [N, n_agents]
        "reward": np.concatenate(all_rewards, axis=0),         # [N, 1]
        "terminated": np.concatenate(all_term, axis=0),        # [N, 1]
        "episode_id": np.concatenate(all_ep_ids, axis=0),      # [N]
        "agent_mask": np.concatenate(all_agent_mask, axis=0),  # [N, n_agents]
    }
    if args.save_q:
        dataset["q_values"] = np.concatenate(all_qvals, axis=0)  # [N, n_agents, A]

    # ---------------------------
    # 7) Manifest
    # ---------------------------
    manifest = {
        "created_utc": int(time.time()),
        "map_name": args.map_name,
        "n_episodes": args.n_episodes,
        "n_agents": n_agents,
        "n_actions": n_actions,
        "state_shape": state_shape,
        "obs_shape": obs_shape,
        "episode_limit": episode_limit,
        "policy": "QMIX",
        "checkpoint": os.path.abspath(args.checkpoint),
        "save_q": bool(args.save_q),
        "seed": args.seed,
        "shapes": {k: list(v.shape) for k, v in dataset.items()},
        "dtypes": {k: str(v.dtype) for k, v in dataset.items()},
    }

    env.close()
    return dataset, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyMARL2 QMIX checkpoint (torch .pt/.tar with 'agent' and 'mixer').")
    parser.add_argument("--map_name", type=str, required=True,
                        help="SMACv2 map name (e.g., MMM2, 3m, 8m_vs_9m, etc.).")
    parser.add_argument("--config_name", type=str, default="qmix",
                        help="(Optional) PyMARL2 base config name; we mainly use it to seed defaults.")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="dataset_out",
                        help="Output prefix (folder or basename). We'll create <out>.npz and <out>.json.")
    parser.add_argument("--save_q", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=False,
                        help="If True, stores per-step Q-values [T, n_agents, n_actions].")
    args = parser.parse_args()

    out_base = Path(args.out)
    out_npz = out_base.with_suffix(".npz")
    out_json = out_base.with_suffix(".json")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    dataset, manifest = rollout_and_collect(args)

    # Save NPZ
    np.savez_compressed(out_npz, **dataset)
    # Save manifest
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Wrote dataset arrays to: {out_npz}")
    print(f"[OK] Wrote manifest to:      {out_json}")


if __name__ == "__main__":
    main()
