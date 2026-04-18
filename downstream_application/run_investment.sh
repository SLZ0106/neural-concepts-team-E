#!/bin/bash
#SBATCH -p 177huntington
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o steering_%j.txt
#SBATCH -e steering_%j.txt
#SBATCH -J steering

set -euo pipefail

source activate /projects/frink/wang.xil/concepts_E/econ_env

PROJECT_ROOT=/projects/frink/wang.xil/concepts_E
APP_DIR=$PROJECT_ROOT/neural-concepts-team-E/downstream_application
DATA=$APP_DIR/data/sentences_with_context.json
DIRECTION=$PROJECT_ROOT/script_outputs/synthetic_direction/direction.npy
OUT_DIR=$APP_DIR/results
mkdir -p "$OUT_DIR"

# -- Experiment config --------------------------------------------------------
# Comma-separated alpha sweep. alpha=0 is the no-steering baseline; include it
# here to get a baseline file in the same run. Model + direction load once and
# are reused across all alphas.
ALPHAS="-10,-8,-6,-4,-2,2,4,6,8,10"
LAYER=12
N_RUNS=10
NO_STATEMENT=0              # 1 to drop the CFO statement from the prompt
# -----------------------------------------------------------------------------

EXTRA_ARGS=()
if [ "$NO_STATEMENT" = "1" ]; then
    EXTRA_ARGS+=("--no-statement")
fi

python $APP_DIR/run_investment.py \
    --data "$DATA" \
    --direction "$DIRECTION" \
    --layer "$LAYER" \
    --alphas="$ALPHAS" \
    --out_dir "$OUT_DIR" \
    --n_runs "$N_RUNS" \
    "${EXTRA_ARGS[@]}"
