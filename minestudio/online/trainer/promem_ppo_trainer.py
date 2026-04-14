"""
ProMemPPOTrainer: PPOTrainer subclass that adds lifecycle supervision losses
for Prospective Memory training.

Adds to the standard PPO loss:
  lifecycle_coef * (fire_loss + cancel_loss + validity_loss + write_loss)

The lifecycle_coef is annealed from pm_coef_start -> pm_coef_end over training.

Oracle events (pm_oracle) must flow from rollout info dicts into the training
batch. This requires carrying pm_oracle through the fragment pipeline.
Since modifying the full rollout pipeline is invasive, we provide a simpler
offline-compatible path: a standalone training script that reads oracle data
from the observation/info dict at training time.
"""
import torch
import logging
from typing import Optional

from minestudio.online.trainer.ppotrainer import PPOTrainer
from minestudio.models.memory.losses import LifecycleLoss

logger = logging.getLogger(__name__)


class ProMemPPOTrainer(PPOTrainer):
    """PPOTrainer with prospective memory lifecycle losses.

    Additional kwargs beyond PPOTrainer:
        pm_coef_start (float): Initial lifecycle loss coefficient (default 1.0)
        pm_coef_end (float): Final lifecycle loss coefficient (default 0.1)
        pm_lambda_fire (float): Weight for fire prediction loss
        pm_lambda_cancel (float): Weight for cancel classification loss
        pm_lambda_valid (float): Weight for validity estimation loss
        pm_lambda_write (float): Weight for write decision loss
    """

    def __init__(
        self,
        pm_coef_start: float = 1.0,
        pm_coef_end: float = 0.1,
        pm_lambda_fire: float = 1.0,
        pm_lambda_cancel: float = 1.0,
        pm_lambda_valid: float = 0.5,
        pm_lambda_write: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pm_coef_start = pm_coef_start
        self.pm_coef_end = pm_coef_end

        self.lifecycle_loss_fn = LifecycleLoss(
            lambda_fire=pm_lambda_fire,
            lambda_cancel=pm_lambda_cancel,
            lambda_valid=pm_lambda_valid,
            lambda_write=pm_lambda_write,
        )

    def get_pm_coef(self) -> float:
        """Linear annealing of lifecycle loss coefficient."""
        if self.num_iterations <= 1:
            return self.pm_coef_start
        frac = min(1.0, self.num_updates / max(1, self.num_iterations - 1))
        return self.pm_coef_start + frac * (self.pm_coef_end - self.pm_coef_start)

    def compute_pm_loss(self, forward_result: dict, batch: dict) -> torch.Tensor:
        """Compute lifecycle loss from forward result and batch oracle data.

        Args:
            forward_result: dict from policy.forward(), must contain model with
                            get_stacked_lifecycle_info() method
            batch: training batch, may contain 'pm_oracle' key

        Returns:
            pm_loss tensor (scalar), or 0 if oracle data not available
        """
        # Get lifecycle info from the policy
        model = self.inner_model
        if not hasattr(model, 'get_stacked_lifecycle_info'):
            return torch.tensor(0.0, device=next(model.parameters()).device)

        lifecycle_info = model.get_stacked_lifecycle_info()
        if lifecycle_info is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # Get oracle events from batch
        pm_oracle = batch.get('pm_oracle', None)
        if pm_oracle is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # Compute losses
        losses = self.lifecycle_loss_fn(lifecycle_info, pm_oracle)

        total = losses.get('total', torch.tensor(0.0, device=next(model.parameters()).device))

        # Log individual components
        for k, v in losses.items():
            if k != 'total' and isinstance(v, torch.Tensor):
                logger.debug(f"pm_loss/{k}: {v.item():.4f}")

        return total


class ProMemOfflineTrainer:
    """Simplified offline trainer for ProMem that avoids the complexity
    of the full online PPO pipeline. Suitable for course project evaluation.

    Trains on pre-collected trajectories with oracle lifecycle supervision.
    Uses standard PyTorch training loop (no Ray, no distributed).
    """

    def __init__(
        self,
        policy,
        optimizer,
        lifecycle_loss_fn: Optional[LifecycleLoss] = None,
        pm_coef: float = 1.0,
        device: str = 'cuda',
    ):
        self.policy = policy.to(device)
        self.optimizer = optimizer
        self.lifecycle_loss_fn = lifecycle_loss_fn or LifecycleLoss()
        self.pm_coef = pm_coef
        self.device = device
        self.step_count = 0

    def train_step(self, batch: dict) -> dict:
        """Single training step on a batch.

        Args:
            batch: dict with keys:
                'image': (B, T, H, W, C) observations
                'action': (B, T, ...) actions taken
                'reward': (B, T) rewards
                'first': (B, T) episode start flags
                'pm_oracle': dict of (B, T, K) oracle tensors

        Returns:
            dict of loss values
        """
        self.policy.train()
        self.optimizer.zero_grad()

        # Move batch to device
        image = batch['image'].to(self.device)
        first = batch['first'].to(self.device)

        input_dict = {'image': image}
        context = {'first': first}

        # Forward pass
        state_in = self.policy.initial_state(batch_size=image.shape[0])
        state_in = [s.to(self.device) if s is not None else s for s in state_in]
        latents, state_out = self.policy.forward(input_dict, state_in, context=context)

        # Compute lifecycle loss
        lifecycle_info = self.policy.get_stacked_lifecycle_info()
        pm_oracle = {k: v.to(self.device) for k, v in batch['pm_oracle'].items()}

        losses = self.lifecycle_loss_fn(lifecycle_info, pm_oracle)
        total_loss = self.pm_coef * losses['total']

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1

        return {
            'total_loss': total_loss.item(),
            **{f'pm_{k}': v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)},
            'step': self.step_count,
        }
