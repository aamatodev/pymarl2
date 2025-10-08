#!/usr/bin/env bash
# Generate a SMACv2 dataset by running PyMARL2 *evaluation* with a trained QMIX policy.
# Drop this at the root of benellis3/pymarl2 (so `src/main.py` exists).
# Example (Protoss 10v10, 4096 eval eps):
#   bash tools/make_eval_dataset.sh \
#     --checkpoint results_smac2_final_run/10gen_protoss/10/0/models \
#     --env-config sc2_gen_protoss \
#     --n-units 10 --n-enemies 10 \
#     --episodes 4096 \
#     --out-dir smac2_dev_experiments/data/smac_2 \
#     --seed 0 --split evaluate

set -e

# --- defaults ---
ENV_CONFIG="sc2_gen_protoss"   # SMACv2 generated scenarios: sc2_gen_protoss | sc2_gen_terran | sc2_gen_zerg
N_UNITS=10
N_ENEMIES=10
EPISODES=4096
OUT_DIR="datasets/smacv2_eval"
SEED=0
SPLIT="evaluate"               # "train" or "evaluate"
WANDB="False"
RUNNER="episode"               # ensure episode runner for clean eval dumps

usage() {
  echo "Usage: $0 --checkpoint <PATH> [--env-config <sc2_gen_*>] [--n-units <N>] [--n-enemies <N>] [--episodes <N>]"
  echo "          [--out-dir <DIR>] [--seed <INT>] [--split train|evaluate]"
  exit 1
}

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2;;
    --env-config) ENV_CONFIG="$2"; shift 2;;
    --n-units)    N_UNITS="$2"; shift 2;;
    --n-enemies)  N_ENEMIES="$2"; shift 2;;
    --episodes)   EPISODES="$2"; shift 2;;
    --out-dir)    OUT_DIR="$2"; shift 2;;
    --seed)       SEED="$2"; shift 2;;
    --split)      SPLIT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -z "$CKPT" ]] && { echo "Missing --checkpoint"; usage; }

# buffer_size should match test_nepisode to collect exactly that many timesteps/eps
python src/main.py \
  --config=qmix \
  --env-config="$ENV_CONFIG" \
  with \
  runner="$RUNNER" \
  use_wandb="$WANDB" \
  evaluate=True \
  checkpoint_path="$CKPT" \
  buffer_size="$EPISODES" \
  test_nepisode="$EPISODES" \
#  save_eval_buffer=True \
#  save_eval_buffer_path="$OUT_DIR" \
  saving_eval_seed="$SEED" \
  saving_eval_type="$SPLIT" \
  env_args.capability_config.n_units="5" \
  env_args.capability_config.n_enemies="$N_ENEMIES"
