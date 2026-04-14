#!/bin/bash
# ProMem Experiment Suite — maps EXPERIMENT_PLAN.md to CLI commands.
# Usage: bash run_all.sh [milestone]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="python $SCRIPT_DIR/run_experiment.py"
OUTPUT="$SCRIPT_DIR/results"
MILESTONE="${1:-all}"

SEEDS_1="--seeds 42"
SEEDS_3="--seeds 42 123 456"

# M0: Sanity Check
if [[ "$MILESTONE" == "M0" || "$MILESTONE" == "all" ]]; then
    echo "=== M0: Sanity Check ==="
    $RUNNER --run_id S01 --system promem   --family F1 --steps 10000  $SEEDS_1 --output_dir $OUTPUT
    $RUNNER --run_id S02 --system reactive --family F1 --steps 10000  $SEEDS_1 --output_dir $OUTPUT
fi

# M1: Baselines (all on F1+F2 for comparable falsification)
if [[ "$MILESTONE" == "M1" || "$MILESTONE" == "all" ]]; then
    echo "=== M1: Baselines ==="
    $RUNNER --run_id B01 --system reactive       --family F1 F2 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id B02 --system retrospective  --family F1 F2 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id B03 --system scratchpad     --family F1 F2 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id B04 --system planner        --family F1 F2 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id B05 --system planner_timer  --family F1 F2 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
fi

# M2: Main Method (also run ProMem on F1+F2 for apples-to-apples comparison with M1)
if [[ "$MILESTONE" == "M2" || "$MILESTONE" == "all" ]]; then
    echo "=== M2: Main Method ==="
    $RUNNER --run_id M01 --system promem  --family F1 F2 F3 F4 F5 F6 --steps 200000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id M02 --system oracle  --family F1 F2 F3 F4 F5 F6 --steps 200000 $SEEDS_3 --output_dir $OUTPUT
fi

# M3: Ablations
if [[ "$MILESTONE" == "M3" || "$MILESTONE" == "all" ]]; then
    echo "=== M3: Ablations ==="
    $RUNNER --run_id A01 --system promem --ablation no_write       --family F1 F2 F6 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id A02 --system promem --ablation no_update      --family F1 F2 F6 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id A03 --system promem --ablation no_validity    --family F1 F2 F6 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id A04 --system promem --ablation no_cancel      --family F1 F2 F6 --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id A05 --system promem --ablation no_negative    --family F6       --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id A06 --system promem --ablation unlimited_slots --family F1 F2   --steps 100000 $SEEDS_3 --output_dir $OUTPUT
    $RUNNER --run_id A07 --system promem --ablation no_supervision --family F1 F2    --steps 100000 $SEEDS_3 --output_dir $OUTPUT
fi

# M4: Scaling (vary number of concurrent families = concurrent contracts)
if [[ "$MILESTONE" == "M4" || "$MILESTONE" == "all" ]]; then
    echo "=== M4: Scaling ==="
    $RUNNER --run_id L01 --system promem --family F1       --steps 100000 $SEEDS_3 --output_dir $OUTPUT  # 1 concurrent
    $RUNNER --run_id L02 --system promem --family F1 F2    --steps 100000 $SEEDS_3 --output_dir $OUTPUT  # 2 concurrent
    $RUNNER --run_id L03 --system promem --family F1 F2 F3 --steps 100000 $SEEDS_3 --output_dir $OUTPUT  # 3 concurrent
    $RUNNER --run_id L04 --system promem --family F1 F2 F3 F4 --steps 100000 $SEEDS_3 --output_dir $OUTPUT  # 4 concurrent
fi

echo "=== All requested milestones complete. Results in: $OUTPUT/ ==="
