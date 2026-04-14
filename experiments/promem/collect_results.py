"""
Collect and aggregate ProMem experiment results.

Reads all *_results.json files from the results directory,
produces summary tables, and updates EXPERIMENT_TRACKER.md.

Usage:
    python collect_results.py --results_dir experiments/promem/results
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def load_all_results(results_dir: str) -> dict:
    """Load all results JSON files."""
    results = {}
    for f in sorted(Path(results_dir).glob("*_results.json")):
        with open(f) as fh:
            data = json.load(fh)
        run_id = data.get('run_id', f.stem.replace('_results', ''))
        results[run_id] = data
    return results


def format_metric(val, fmt=".3f"):
    if val is None or val == '—':
        return '—'
    return f"{val:{fmt}}"


def generate_summary_table(results: dict) -> str:
    """Generate a markdown summary table."""
    lines = [
        "| Run ID | System | Families | Reward | PM Recall@50 | Obsolete-Fire | Latency |",
        "|--------|--------|----------|--------|-------------|---------------|---------|",
    ]
    for run_id, data in sorted(results.items()):
        agg = data.get('aggregate', {})
        system = data.get('system', '?')
        ablation = data.get('ablation', '')
        if ablation:
            system = f"{system} -{ablation}"
        families = ','.join(data.get('families', []))
        reward = format_metric(agg.get('mean_reward'), ".1f")
        recall = format_metric(agg.get('mean_pm_recall_at_delta'))
        obsolete = format_metric(agg.get('mean_obsolete_fire_rate'))
        latency = format_metric(agg.get('mean_trigger_to_action_latency'), ".1f")
        lines.append(f"| {run_id} | {system} | {families} | {reward} | {recall} | {obsolete} | {latency} |")
    return '\n'.join(lines)


def generate_tracker_update(results: dict) -> str:
    """Generate an updated EXPERIMENT_TRACKER.md."""
    lines = [
        "# Experiment Tracker",
        "",
        "| Run ID | Milestone | System | Task | Status | Success | PM Recall@50 | Obsolete-Fire | Notes |",
        "|--------|-----------|--------|------|--------|---------|-------------|---------------|-------|",
    ]

    # Map run_id prefix to milestone
    milestone_map = {'S': 'M0', 'B': 'M1', 'M': 'M2', 'A': 'M3', 'L': 'M4'}

    for run_id, data in sorted(results.items()):
        milestone = milestone_map.get(run_id[0], '?')
        system = data.get('system', '?')
        ablation = data.get('ablation', '')
        if ablation:
            system = f"{system} -{ablation}"
        families = ','.join(data.get('families', []))
        agg = data.get('aggregate', {})

        if agg:
            status = 'DONE'
            reward = format_metric(agg.get('mean_reward'), ".1f")
            recall = format_metric(agg.get('mean_pm_recall_at_delta'))
            obsolete = format_metric(agg.get('mean_obsolete_fire_rate'))
            n_eps = agg.get('num_episodes', 0)
            n_err = agg.get('num_errors', 0)
            notes = f"{n_eps} eps"
            if n_err > 0:
                notes += f", {n_err} errors"
        else:
            status = 'ERROR'
            reward = recall = obsolete = '—'
            notes = 'No successful episodes'

        lines.append(f"| {run_id} | {milestone} | {system} | {families} | {status} | {reward} | {recall} | {obsolete} | {notes} |")

    return '\n'.join(lines)


def generate_experiment_results_md(results: dict) -> str:
    """Generate refine-logs/EXPERIMENT_RESULTS.md."""
    lines = [
        "# Initial Experiment Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        "**Plan**: refine-logs/EXPERIMENT_PLAN.md",
        "",
        "## Summary Table",
        "",
        generate_summary_table(results),
        "",
    ]

    # Group by milestone
    milestone_map = {'S': 'M0: Sanity', 'B': 'M1: Baselines', 'M': 'M2: Main Method',
                     'A': 'M3: Ablations', 'L': 'M4: Scaling'}

    for prefix, title in milestone_map.items():
        milestone_results = {k: v for k, v in results.items() if k.startswith(prefix)}
        if milestone_results:
            lines.append(f"\n### {title}")
            lines.append("")
            for run_id, data in sorted(milestone_results.items()):
                agg = data.get('aggregate', {})
                lines.append(f"**{run_id}** ({data.get('system', '?')}): "
                           f"reward={format_metric(agg.get('mean_reward'), '.1f')}, "
                           f"recall={format_metric(agg.get('mean_pm_recall_at_delta'))}, "
                           f"obsolete={format_metric(agg.get('mean_obsolete_fire_rate'))}")

    # Check falsification criteria
    lines.extend([
        "",
        "## Falsification Check",
        "",
    ])

    m01 = results.get('M01', {}).get('aggregate', {})
    b05 = results.get('B05', {}).get('aggregate', {})
    m02 = results.get('M02', {}).get('aggregate', {})
    b01 = results.get('B01', {}).get('aggregate', {})

    if m01 and b05:
        m01_r = m01.get('mean_pm_recall_at_delta', 0)
        b05_r = b05.get('mean_pm_recall_at_delta', 0)
        if b05_r >= m01_r * 0.95:
            lines.append("- **WARNING**: B05 (planner+timer) matches M01 — PM abstraction may be unnecessary")
        else:
            lines.append(f"- PASS: M01 recall ({m01_r:.3f}) > B05 ({b05_r:.3f}) — PM adds value")

    if m02 and b01:
        m02_r = m02.get('mean_reward', 0)
        b01_r = b01.get('mean_reward', 0)
        if m02_r <= b01_r * 1.05:
            lines.append("- **WARNING**: Oracle (M02) ≈ Reactive (B01) — bottleneck is perception, not PM")
        else:
            lines.append(f"- PASS: Oracle ({m02_r:.1f}) >> Reactive ({b01_r:.1f}) — PM headroom exists")

    lines.extend([
        "",
        "## Next Step",
        "→ /auto-review-loop \"ProMem: Prospective Memory for Embodied Agents\"",
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Collect ProMem results')
    parser.add_argument('--results_dir', type=str, default='experiments/promem/results')
    parser.add_argument('--output_dir', type=str, default='doc/refine-logs')
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result files")

    # Write results summary
    results_md = generate_experiment_results_md(results)
    results_path = os.path.join(args.output_dir, 'EXPERIMENT_RESULTS.md')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(results_md)
    print(f"Results written to {results_path}")

    # Update tracker
    tracker_md = generate_tracker_update(results)
    tracker_path = os.path.join(args.output_dir, 'EXPERIMENT_TRACKER.md')
    with open(tracker_path, 'w') as f:
        f.write(tracker_md)
    print(f"Tracker updated at {tracker_path}")

    # Print summary table
    print("\n" + generate_summary_table(results))


if __name__ == '__main__':
    main()
