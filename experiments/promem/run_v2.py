"""
ProMem Experiment Runner v2.

Implements the revised experiment plan:
  M0: Calibration (trivial vs excursion)
  M1: Data collection (oracle-labeled trajectories)
  M2: Train lifecycle heads
  M3: Main closed-loop probe
  M4: OOD generalization
  M5: Cancel ablation

Usage:
    python run_v2.py M0              # calibration
    python run_v2.py M1              # data collection
    python run_v2.py M3 --delay short  # main probe, short delay
"""
import os
import sys
import json
import math
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_DIR = "experiments/promem/results_v2"
DELAY_BINS = {
    'short':  (80, 120),
    'medium': (160, 220),
    'long':   (260, 340),
    'xlong':  (400, 500),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def run_episode(env, policy, max_steps, device='cuda'):
    """Run one episode, return metrics dict."""
    import torch
    from minestudio.simulator.callbacks.promem_benchmark import get_player_pos

    obs, info = env.reset()
    memory = None
    total_reward = 0.0
    trajectory = []  # for data collection

    for step in range(max_steps):
        with torch.no_grad():
            action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Save trajectory data for oracle supervision
        trajectory.append({
            'step': step,
            'reward': float(reward),
            'pm_oracle': {k: v.tolist() if hasattr(v, 'tolist') else v
                         for k, v in info.get('pm_oracle', {}).items()},
            'pm_metrics': info.get('pm_metrics', {}),
        })

        if terminated or truncated:
            break

    pm = info.get('pm_metrics', {})
    # Get final distance to furnace if available
    cb = None
    for c in (env.callbacks if isinstance(env.callbacks, list) else []):
        if hasattr(c, 'furnace_pos') and c.furnace_pos:
            cb = c
            break

    dist_at_end = None
    if cb and cb.furnace_pos:
        pos = get_player_pos(info)
        fx, _, fz = cb.furnace_pos
        dist_at_end = math.sqrt((pos[0] - fx)**2 + (pos[2] - fz)**2)

    return {
        'reward': float(total_reward),
        'steps': step + 1,
        'pm_recall': pm.get('pm_recall_at_delta', 0.0),
        'obsolete_fire': pm.get('obsolete_fire_rate', 0.0),
        'latency': pm.get('trigger_to_action_latency', float('inf')),
        'dist_at_end': dist_at_end,
        'collected': cb.collected if cb and hasattr(cb, 'collected') else None,
        'smelt_ready': cb.smelt_ready if cb and hasattr(cb, 'smelt_ready') else None,
    }


def run_calibration(args):
    """M0: Prove headroom exists. Trivial (no excursion) vs Excursion."""
    import torch
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks.promem_benchmark import DelayedProcessCallback
    from minestudio.models import VPTPolicy

    policy = VPTPolicy.from_pretrained(
        'CraftJarvis/MineStudio_VPT.rl_from_early_game_2x').to('cuda')
    policy.eval()

    results = {'milestone': 'M0', 'timestamp': datetime.now().isoformat(), 'runs': []}

    for condition in ['trivial', 'excursion_short', 'excursion_long']:
        if condition == 'trivial':
            delay, excursion = (30, 50), 0
        elif condition == 'excursion_short':
            delay, excursion = (80, 120), 15
        else:
            delay, excursion = (260, 340), 15

        for seed in [42, 123, 456]:
            seed_all(seed)
            print(f"\n--- {condition} | seed={seed} ---")

            try:
                cb = DelayedProcessCallback(
                    smelt_delay_range=delay, add_perturbation=False,
                    auto_start_smelt=True, collection_radius=5.0,
                    forced_excursion_dist=excursion)
                env = MinecraftSim(obs_size=(128, 128), callbacks=[cb])

                ep = run_episode(env, policy, max_steps=400)
                ep['condition'] = condition
                ep['seed'] = seed
                ep['delay_range'] = list(delay)
                ep['excursion'] = excursion
                results['runs'].append(ep)

                d = ep['dist_at_end']
                d_str = f"{d:.1f}" if d is not None else "N/A"
                print(f"  reward={ep['reward']:.1f} recall={ep['pm_recall']:.3f} "
                      f"dist={d_str} collected={ep['collected']}")

                env.close()
            except Exception as e:
                print(f"  ERROR: {e} — skipping")
                results['runs'].append({
                    'condition': condition, 'seed': seed, 'error': str(e),
                    'reward': 0, 'pm_recall': 0, 'dist_at_end': None, 'collected': False,
                })
                try:
                    env.close()
                except Exception:
                    pass

    # Aggregate
    for cond in ['trivial', 'excursion_short', 'excursion_long']:
        eps = [r for r in results['runs'] if r['condition'] == cond]
        recalls = [r['pm_recall'] for r in eps]
        rewards = [r['reward'] for r in eps]
        print(f"\n{cond}: mean_recall={np.mean(recalls):.3f}±{np.std(recalls):.3f}, "
              f"mean_reward={np.mean(rewards):.1f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/M0_calibration.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/M0_calibration.json")


def run_data_collection(args):
    """M1: Collect oracle-labeled trajectories for lifecycle training."""
    import torch
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks.promem_benchmark import DelayedProcessCallback
    from minestudio.models import VPTPolicy

    policy = VPTPolicy.from_pretrained(
        'CraftJarvis/MineStudio_VPT.rl_from_early_game_2x').to('cuda')
    policy.eval()

    n_episodes = args.n_episodes or 100
    results = {'milestone': 'M1', 'timestamp': datetime.now().isoformat(), 'episodes': []}

    for ep_idx in range(n_episodes):
        seed = 1000 + ep_idx
        seed_all(seed)

        # Random delay bin + random perturbation
        delay_name = random.choice(['short', 'medium', 'long'])
        delay_range = DELAY_BINS[delay_name]
        perturb = random.random() < 0.25  # 25% perturbation rate
        excursion = random.choice([0, 10, 15, 20])  # varied excursion

        cb = DelayedProcessCallback(
            smelt_delay_range=delay_range,
            add_perturbation=perturb, perturbation_prob=0.3 if perturb else 0,
            auto_start_smelt=True, collection_radius=5.0,
            forced_excursion_dist=excursion)
        env = MinecraftSim(obs_size=(128, 128), callbacks=[cb])

        ep = run_episode(env, policy, max_steps=500)
        ep['seed'] = seed
        ep['delay_name'] = delay_name
        ep['perturbed'] = perturb
        ep['excursion'] = excursion
        ep['smelt_ready_step'] = cb.smelt_ready_step
        ep['destroyed'] = cb.destroyed if hasattr(cb, 'destroyed') else False
        results['episodes'].append(ep)

        if ep_idx % 10 == 0:
            print(f"  Episode {ep_idx}/{n_episodes}: "
                  f"delay={delay_name} excursion={excursion} perturb={perturb} "
                  f"recall={ep['pm_recall']:.2f} reward={ep['reward']:.1f}")

        try:
            env.close()
        except Exception:
            pass

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/M1_data.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    eps = results['episodes']
    print(f"\nCollected {len(eps)} episodes")
    print(f"  With excursion>0: {sum(1 for e in eps if e['excursion']>0)}")
    print(f"  Perturbed: {sum(1 for e in eps if e['perturbed'])}")
    print(f"  Mean recall: {np.mean([e['pm_recall'] for e in eps]):.3f}")
    print(f"  Trigger events: {sum(1 for e in eps if e['smelt_ready'])}")


def main():
    parser = argparse.ArgumentParser(description='ProMem v2 Experiments')
    parser.add_argument('milestone', choices=['M0', 'M1', 'M3', 'M4', 'M5'])
    parser.add_argument('--delay', choices=['short', 'medium', 'long', 'xlong'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.milestone == 'M0':
        run_calibration(args)
    elif args.milestone == 'M1':
        run_data_collection(args)
    else:
        print(f"Milestone {args.milestone} not yet implemented — needs lifecycle training first")
        print("Run M0 and M1 first, then train lifecycle heads, then M3-M5")


if __name__ == '__main__':
    main()
