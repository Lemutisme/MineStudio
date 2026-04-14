# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MineStudio is a Minecraft AI agent development kit for research. It provides tools for building, training (offline and online), and evaluating Minecraft agents. Built on MineRL/Project Malmo, PyTorch Lightning, Ray, and Hydra. Python 3.10+, published on PyPI as `minestudio`.

## Common Commands

```bash
# Install from source (editable)
pip install -e .

# Run tests
pytest

# Run a single test
pytest tests/test_task_configs.py

# Verify simulator installation
python -m minestudio.simulator.entry              # CPU rendering (Xvfb)
MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry  # GPU rendering (VirtualGL)

# Build docs
cd docs && make html
```

## Environment Variables

- `MINESTUDIO_SAVE_DIR` — directory for saving outputs
- `MINESTUDIO_DATABASE_DIR` — directory for dataset LMDB databases
- `MINESTUDIO_GPU_RENDER` — set to `1` to use VirtualGL instead of Xvfb

## Architecture

The package lives in `minestudio/` with seven core modules:

### Simulator (`minestudio/simulator/`)
- `entry.py` — `MinecraftSim` class (Gymnasium-based env wrapping MineRL's `HumanSurvival`)
- `CameraConfig` dataclass controls camera quantization (mu-law or linear)
- Extensible via **callbacks** (`callbacks/callback.py` defines `MinecraftCallback` base class). Callbacks hook into reset/step/close to add behavior (recording, rewards, fast reset, task evaluation, etc.)

### Data (`minestudio/data/`)
- Two dataset classes: `RawDataset` and `EventDataset` (in `data/minecraft/`)
- LMDB-backed trajectory storage for efficient random access
- **Callback-driven modal processing**: `ImageKernelCallback`, `ActionKernelCallback`, `MetaInfoKernelCallback`, `SegmentationKernelCallback` — each handles one trajectory modality. Users compose callbacks to control what data is loaded/transformed.

### Models (`minestudio/models/`)
- `MinePolicy` (in `base_policy.py`) — abstract base class extending `torch.nn.Module`. Defines the interface: `get_action()`, `initial_state()`, `reset_parameters()`. Includes action/value heads and Hugging Face `from_pretrained()` support.
- Implementations: `VPTPolicy`, `RocketPolicy` (ROCKET-1), `GrootPolicy` (GROOT), `SteveOnePolicy` (STEVE-1)
- Helper: `recursive_tensor_op()` for applying functions across nested tensor structures

### Offline Training (`minestudio/offline/`)
- `trainer.py` — PyTorch Lightning-based trainer
- `mine_callbacks/` — training objective callbacks (`ObjectiveCallback`)
- `lightning_callbacks/` — standard Lightning training callbacks
- Configured via Hydra YAML configs (see `tutorials/offline/`)

### Online Training (`minestudio/online/`)
- Distributed RL with Ray + PyTorch Lightning
- `trainer/` — `PPOTrainer`, `PPOBCTrainer`, `BaseTrainer`
- `rollout/` — rollout collection with simulator crash recovery
- `run/` — entry points for launching training

### Inference (`minestudio/inference/`)
- Ray-based distributed inference pipeline (`pipeline.py`)
- Modular components: `generator/`, `filter/`, `recorder/`

### Benchmark (`minestudio/benchmark/`)
- YAML task configs in `task_configs/` (simple and hard difficulty)
- `auto_eval/` — automated evaluation framework
- `TaskCallback` in simulator callbacks drives task-based evaluation

## Key Design Patterns

- **Callback pattern** is used extensively in both simulator and data modules. Simulator callbacks inherit from `MinecraftCallback`; data callbacks inherit from the kernel callback base in `data/minecraft/callbacks/callback.py`.
- **Hydra configuration** — offline/online training configs are YAML-based, composed via Hydra.
- **Registry pattern** — `minestudio/utils/register.py` provides `Register` and `Registers` for registering models and utilities.
- **Hugging Face integration** — models support `from_pretrained()` / `push_to_hub()` via `MinePolicy` base class.

## Tutorials

Located in `minestudio/tutorials/` with subdirectories for `simulator/`, `offline/`, and `inference/`. These contain working examples with Hydra configs.
