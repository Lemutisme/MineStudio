"""
Lifecycle losses for ProMem training.

Implements the three-tier training scheme:
  Tier 1: Privileged lifecycle supervision (dense, from oracle events)
  Tier 2: Hindsight contract relabeling (semi-dense, per-episode)
  Tier 3: End-to-end RL (sparse, via PPO — handled externally)

Oracle events use stable contract_ids. This module aligns oracle slot indices
to model slot indices via contract_id matching before computing losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def align_oracle_to_model(
    oracle_events: Dict[str, torch.Tensor],
    model_contract_id: torch.Tensor,
    oracle_contract_id: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Remap oracle signals from oracle slot order to model slot order
    by matching on contract_id.

    Args:
        oracle_events: dict with (B, T, K_oracle) tensors
        model_contract_id: (B, T, K_model) long tensor of model slot IDs
        oracle_contract_id: (B, T, K_oracle) long tensor of oracle slot IDs
    Returns:
        dict with (B, T, K_model) tensors aligned to model slots
    """
    B, T, K_model = model_contract_id.shape
    K_oracle = oracle_contract_id.shape[2]
    device = model_contract_id.device

    aligned = {}
    for key, oracle_tensor in oracle_events.items():
        if key == 'oracle_contract_id':
            continue
        if oracle_tensor.dim() < 3:
            aligned[key] = oracle_tensor
            continue

        out = torch.zeros(B, T, K_model, *oracle_tensor.shape[3:], device=device)
        # Match by contract_id: for each model slot, find the oracle slot with same ID
        for b in range(B):
            for t in range(T):
                for km in range(K_model):
                    mid = model_contract_id[b, t, km].item()
                    if mid == 0:
                        continue
                    for ko in range(K_oracle):
                        if oracle_contract_id[b, t, ko].item() == mid:
                            out[b, t, km] = oracle_tensor[b, t, ko]
                            break
        aligned[key] = out
    return aligned


class LifecycleLoss(nn.Module):
    """Computes lifecycle supervision losses from oracle events.

    Handles contract_id alignment automatically when oracle_contract_id is provided.
    """

    def __init__(
        self,
        lambda_fire: float = 1.0,
        lambda_cancel: float = 1.0,
        lambda_valid: float = 0.5,
        lambda_write: float = 0.5,
    ):
        super().__init__()
        self.lambda_fire = lambda_fire
        self.lambda_cancel = lambda_cancel
        self.lambda_valid = lambda_valid
        self.lambda_write = lambda_write

    def fire_loss(
        self,
        trigger_hazard: torch.Tensor,
        oracle_fire: torch.Tensor,
        live_mask: torch.Tensor,
    ) -> torch.Tensor:
        """BCE on trigger hazard. Shapes: (B, T, K)."""
        mask = live_mask.float()
        n = mask.sum().clamp(min=1)
        loss = F.binary_cross_entropy(
            trigger_hazard.clamp(1e-7, 1 - 1e-7) * mask,
            oracle_fire * mask,
            reduction='none',
        )
        return (loss * mask).sum() / n

    def cancel_loss(
        self,
        cancel_logits: torch.Tensor,
        oracle_cancel: torch.Tensor,
        live_mask: torch.Tensor,
    ) -> torch.Tensor:
        """CE on cancel classification. cancel_logits: (B,T,K,3), oracle_cancel: (B,T,K) with 0-3."""
        has_label = (oracle_cancel > 0) & live_mask
        if not has_label.any():
            return torch.tensor(0.0, device=cancel_logits.device)
        targets = (oracle_cancel - 1).clamp(min=0).long()
        return F.cross_entropy(cancel_logits[has_label], targets[has_label])

    def validity_loss(
        self,
        validity: torch.Tensor,
        oracle_valid: torch.Tensor,
        live_mask: torch.Tensor,
    ) -> torch.Tensor:
        """BCE on validity estimation. Shapes: (B, T, K)."""
        mask = live_mask.float()
        n = mask.sum().clamp(min=1)
        loss = F.binary_cross_entropy(
            validity.clamp(1e-7, 1 - 1e-7) * mask,
            oracle_valid * mask,
            reduction='none',
        )
        return (loss * mask).sum() / n

    def write_loss(
        self,
        write_score: torch.Tensor,
        oracle_should_write: torch.Tensor,
    ) -> torch.Tensor:
        """BCE on write decision. write_score: (B,T,1), oracle: (B,T,1)."""
        return F.binary_cross_entropy_with_logits(
            write_score.float(), oracle_should_write.float(),
        )

    def forward(
        self,
        lifecycle_info: Dict[str, torch.Tensor],
        oracle_events: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute all lifecycle losses.

        Args:
            lifecycle_info: stacked output from ProMemPolicy.get_stacked_lifecycle_info()
                Keys: trigger_hazard (B,T,K), validity (B,T,K), cancel_logits (B,T,K,3),
                      write_score (B,T,1), live_mask (B,T,K), contract_id (B,T,K)
            oracle_events: from simulator callbacks, stacked to (B,T,...).
                Keys: oracle_fire (B,T,K), oracle_cancel (B,T,K), oracle_valid (B,T,K),
                      oracle_should_write (B,T,1), oracle_contract_id (B,T,K)
        Returns:
            dict of individual losses and 'total'
        """
        live_mask = lifecycle_info['live_mask']

        # Align oracle to model slot order if contract_ids are available
        if 'oracle_contract_id' in oracle_events and 'contract_id' in lifecycle_info:
            oracle_events = align_oracle_to_model(
                oracle_events,
                lifecycle_info['contract_id'].long(),
                oracle_events['oracle_contract_id'].long(),
            )

        losses = {}

        if 'oracle_fire' in oracle_events and 'trigger_hazard' in lifecycle_info:
            losses['fire'] = self.lambda_fire * self.fire_loss(
                lifecycle_info['trigger_hazard'], oracle_events['oracle_fire'], live_mask
            )

        if 'oracle_cancel' in oracle_events and 'cancel_logits' in lifecycle_info:
            losses['cancel'] = self.lambda_cancel * self.cancel_loss(
                lifecycle_info['cancel_logits'], oracle_events['oracle_cancel'], live_mask
            )

        if 'oracle_valid' in oracle_events and 'validity' in lifecycle_info:
            losses['valid'] = self.lambda_valid * self.validity_loss(
                lifecycle_info['validity'], oracle_events['oracle_valid'], live_mask
            )

        if 'oracle_should_write' in oracle_events and 'write_score' in lifecycle_info:
            losses['write'] = self.lambda_write * self.write_loss(
                lifecycle_info['write_score'], oracle_events['oracle_should_write']
            )

        losses['total'] = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        return losses
