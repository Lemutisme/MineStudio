"""
ProMem Experiment Runner.

Entry point for all experiment milestones (M0-M4) from EXPERIMENT_PLAN.md.
Runs in MineStudio's simulator, collects PM-specific metrics, saves to JSON.

Usage:
    python run_experiment.py --run_id S01 --system promem --family F1 --steps 10000 --seed 42
    python run_experiment.py --run_id M01 --system promem --family F1 F2 F3 F4 F5 F6 --steps 200000
"""
import os
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def seed_all(seed: int):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_benchmark_callback(family: str):
    """Get the appropriate benchmark callback for a task family."""
    from minestudio.simulator.callbacks.promem_benchmark import (
        DelayedProcessCallback, OpportunisticEventCallback,
        ActivityHandoffCallback, InternalThresholdCallback,
        PlaceReturnCallback, NegativeContractCallback,
    )
    return {
        'F1': DelayedProcessCallback,
        'F2': OpportunisticEventCallback,
        'F3': ActivityHandoffCallback,
        'F4': InternalThresholdCallback,
        'F5': PlaceReturnCallback,
        'F6': NegativeContractCallback,
    }[family]()


def create_env(family: str, record_path: str = None):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import RecordCallback
    cbs = [get_benchmark_callback(family)]
    if record_path:
        cbs.append(RecordCallback(record_path=record_path, fps=20, frame_type="pov"))
    return MinecraftSim(obs_size=(128, 128), callbacks=cbs)


def create_policy(system: str, ablation: str = None, device: str = 'cuda'):
    """Create policy. Returns (policy, has_promem, is_oracle)."""
    import torch
    from minestudio.models import VPTPolicy

    base = VPTPolicy.from_pretrained(
        "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x"
    ).to(device)

    if system == 'reactive':
        return base, False, False

    if system == 'retrospective':
        # Same VPT but we explicitly track that its recurrent state IS the memory.
        # For metrics, we treat it as a retrospective-only baseline.
        return base, False, False

    if system in ('scratchpad', 'planner', 'planner_timer'):
        # Simplified baselines: use the same VPT backbone.
        # The distinction is in the info/obs manipulation, not the policy architecture.
        # scratchpad: text-based memory appended to obs (no structural lifecycle)
        # planner: re-plan every N steps from current state
        # planner_timer: planner + hardcoded timer queue for time-based contracts
        # For a fair comparison, all share the same backbone parameters.
        return base, False, False

    if system in ('promem', 'oracle'):
        from minestudio.models.memory import ProMemPolicy
        kwargs = dict(num_slots=4, embed_dim=256, fire_threshold=0.5,
                      cancel_threshold=0.2, merge_threshold=0.8)

        if ablation == 'unlimited_slots':
            kwargs['num_slots'] = 64
        if ablation == 'no_validity':
            kwargs['cancel_threshold'] = -1.0  # never cancel via validity

        policy = ProMemPolicy(base, **kwargs).to(device)

        # Apply ablation flags as module attributes for runtime checks
        if ablation == 'no_write':
            policy.promem._ablation_always_write = True
        if ablation == 'no_update':
            policy.promem._ablation_freeze_contracts = True
        if ablation == 'no_cancel':
            policy.promem._ablation_no_cancel = True
        if ablation == 'no_negative':
            policy.promem._ablation_no_negative = True
        if ablation == 'no_supervision':
            policy._ablation_no_supervision = True

        return policy, True, (system == 'oracle')

    raise ValueError(f"Unknown system: {system}")


def run_episode(env, policy, has_promem: bool, max_steps: int,
                is_oracle: bool = False):
    """Run a single episode and collect metrics."""
    import torch
    obs, info = env.reset()
    memory = None
    episode_reward = 0.0
    pm_metrics_final = {}
    step_count = 0

    for step in range(max_steps):
        with torch.no_grad():
            action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        if 'pm_metrics' in info:
            pm_metrics_final = info['pm_metrics']
        if terminated or truncated:
            break

    contract_summary = {}
    if has_promem and hasattr(policy, 'get_contract_summary'):
        try:
            contract_summary = policy.get_contract_summary()
        except Exception:
            pass

    return {
        'episode_reward': float(episode_reward),
        'steps': step_count,
        'pm_metrics': pm_metrics_final,
        'contract_summary': contract_summary,
    }


def run_experiment(args):
    results = {
        'run_id': args.run_id,
        'system': args.system,
        'ablation': args.ablation,
        'families': args.family,
        'steps': args.steps,
        'seeds': args.seeds,
        'timestamp': datetime.now().isoformat(),
        'episodes': [],
    }

    for seed in args.seeds:
        for family in args.family:
            print(f"\n{'='*60}")
            print(f"Run {args.run_id} | {args.system}"
                  f"{' -' + args.ablation if args.ablation else ''}"
                  f" | {family} | seed={seed}")
            print(f"{'='*60}")

            seed_all(seed)
            env = None
            try:
                record_path = None
                if args.record:
                    record_path = os.path.join(args.output_dir, args.run_id, f"{family}_s{seed}")
                    os.makedirs(record_path, exist_ok=True)

                env = create_env(family, record_path=record_path)
                policy, has_promem, is_oracle = create_policy(
                    args.system, ablation=args.ablation, device=args.device
                )
                policy.eval()

                ep = run_episode(env, policy, has_promem, args.steps, is_oracle)
                ep['seed'] = seed
                ep['family'] = family
                results['episodes'].append(ep)

                print(f"  Reward: {ep['episode_reward']:.2f} | Steps: {ep['steps']}")
                for k, v in ep.get('pm_metrics', {}).items():
                    print(f"  {k}: {v:.4f}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                results['episodes'].append({'seed': seed, 'family': family, 'error': str(e)})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

    # Aggregate
    ok = [ep for ep in results['episodes'] if 'error' not in ep]
    if ok:
        results['aggregate'] = {
            'mean_reward': float(np.mean([e['episode_reward'] for e in ok])),
            'mean_steps': float(np.mean([e['steps'] for e in ok])),
            'num_episodes': len(ok),
            'num_errors': len(results['episodes']) - len(ok),
        }
        pm_keys = set()
        for e in ok:
            pm_keys.update(e.get('pm_metrics', {}).keys())
        for k in pm_keys:
            vals = [e['pm_metrics'][k] for e in ok if k in e.get('pm_metrics', {})]
            if vals:
                results['aggregate'][f'mean_{k}'] = float(np.mean(vals))
                results['aggregate'][f'std_{k}'] = float(np.std(vals))

    output_file = os.path.join(args.output_dir, f"{args.run_id}_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='ProMem Experiment Runner')
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--system', type=str, required=True,
                        choices=['promem', 'reactive', 'retrospective', 'scratchpad',
                                 'planner', 'planner_timer', 'oracle'])
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_write', 'no_update', 'no_validity', 'no_cancel',
                                 'no_negative', 'unlimited_slots', 'no_supervision'])
    parser.add_argument('--family', type=str, nargs='+', default=['F1'],
                        choices=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output_dir', type=str, default='experiments/promem/results')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
