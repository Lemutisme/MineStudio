"""
ProMem: Prospective Memory Module.

Implements the learned lifecycle policies (write, update, merge, fire, cancel)
over a fixed-capacity ContractBuffer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from minestudio.models.memory.contract import ContractBuffer, LifecycleState


class SlotUpdateGRU(nn.Module):
    """Per-slot GRU that updates contract hidden state from projected context."""

    def __init__(self, slot_dim: int, ctx_dim: int):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, slot_dim)
        self.gate_proj = nn.Linear(2 * slot_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)

    def forward(self, slot_hidden: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_hidden: (B, K, D) current slot hidden states
            ctx: (B, ctx_dim) context vector
        Returns:
            updated_hidden: (B, K, D)
        """
        B, K, D = slot_hidden.shape
        ctx_proj = self.ctx_proj(ctx).unsqueeze(1).expand(B, K, D)  # (B, K, D)
        # Gate: how much context to attend to per slot
        gate_input = torch.cat([slot_hidden, ctx_proj], dim=-1)  # (B, K, 2D)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, K, D)
        gated_ctx = gate * ctx_proj  # (B, K, D)
        # GRU update per slot
        flat_hidden = slot_hidden.reshape(B * K, D)
        flat_ctx = gated_ctx.reshape(B * K, D)
        updated = self.gru(flat_ctx, flat_hidden)
        return updated.reshape(B, K, D)


class ProMemModule(nn.Module):
    """Prospective Memory Module with learned lifecycle policies.

    Manages a ContractBuffer of K slots. At each step:
    1. Encode context from observation + policy state
    2. Propose new contracts (write policy)
    3. Update existing contracts (update policy)
    4. Fire triggered contracts (fire policy)
    5. Cancel invalidated contracts (cancel policy)
    """

    def __init__(
        self,
        num_slots: int = 4,
        embed_dim: int = 256,
        ctx_dim: int = 512,
        fire_threshold: float = 0.5,
        cancel_threshold: float = 0.2,
        merge_threshold: float = 0.8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        self.ctx_dim = ctx_dim
        self.fire_threshold = fire_threshold
        self.cancel_threshold = cancel_threshold
        self.merge_threshold = merge_threshold

        # Ablation flags — set externally by experiment runner
        self._ablation_always_write = False    # no_write: skip write score check
        self._ablation_freeze_contracts = False  # no_update: don't update slots
        self._ablation_no_cancel = False       # no_cancel: never retire slots
        self._ablation_no_negative = False     # no_negative: force polarity=+1

        # Context encoder: fuses observation features with contract state
        slot_pack_dim = 3 * embed_dim + 7  # from ContractBuffer.pack(): 3D + 7 scalars
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_dim + num_slots * slot_pack_dim, 512),
            nn.ReLU(),
            nn.Linear(512, ctx_dim),
        )

        # Proposal network: generates candidate contract embeddings
        self.proposal_net = nn.Sequential(
            nn.Linear(ctx_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * embed_dim),  # referent + intention
        )

        # Write scorer: value-of-remembering
        self.write_scorer = nn.Sequential(
            nn.Linear(ctx_dim + 2 * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Slot update GRU
        self.slot_updater = SlotUpdateGRU(embed_dim, ctx_dim)

        # Trigger hazard head
        self.trigger_head = nn.Sequential(
            nn.Linear(embed_dim + ctx_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Validity head
        self.validity_head = nn.Sequential(
            nn.Linear(embed_dim + ctx_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Priority head
        self.priority_head = nn.Sequential(
            nn.Linear(embed_dim + ctx_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        # Cancel head: predicts P(should cancel)
        self.cancel_head = nn.Sequential(
            nn.Linear(embed_dim + ctx_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # completed, canceled, superseded logits
        )

        # Reminder token projection: converts fired contract into policy input
        self.reminder_proj = nn.Linear(2 * embed_dim + 1, ctx_dim)  # referent + intention + polarity

    def encode_context(
        self, obs_features: torch.Tensor, contract_buffer: ContractBuffer
    ) -> torch.Tensor:
        """Encode context from observation features and current contract state.

        Args:
            obs_features: (B, ctx_dim) from the backbone policy
            contract_buffer: current contract state
        Returns:
            ctx: (B, ctx_dim) fused context vector
        """
        B = obs_features.shape[0]
        slot_pack = contract_buffer.pack()  # (B, K, pack_dim)
        slot_flat = slot_pack.reshape(B, -1)  # (B, K * pack_dim)
        combined = torch.cat([obs_features, slot_flat], dim=-1)
        return self.ctx_encoder(combined)

    def propose_contracts(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate candidate contract embeddings.

        Args:
            ctx: (B, ctx_dim)
        Returns:
            referent: (B, D)
            intention: (B, D)
            write_score: (B, 1) value-of-remembering
        """
        proposal = self.proposal_net(ctx)  # (B, 2D)
        referent = proposal[:, :self.embed_dim]
        intention = proposal[:, self.embed_dim:]
        score_input = torch.cat([ctx, referent, intention], dim=-1)
        write_score = self.write_scorer(score_input)
        return referent, intention, write_score

    def update_slots(self, ctx: torch.Tensor, contract_buffer: ContractBuffer):
        """Update all active slots: hidden state, trigger, validity, priority.

        Args:
            ctx: (B, ctx_dim)
            contract_buffer: modified in-place
        """
        B, K, D = contract_buffer.hidden.shape
        active_mask = contract_buffer.get_active_mask()  # (B, K)

        # Update hidden state via GRU
        new_hidden = self.slot_updater(contract_buffer.hidden, ctx)
        contract_buffer.hidden = torch.where(
            active_mask.unsqueeze(-1).expand_as(new_hidden),
            new_hidden, contract_buffer.hidden
        )

        # Update trigger hazard, validity, priority per slot
        ctx_expanded = ctx.unsqueeze(1).expand(B, K, -1)  # (B, K, ctx_dim)
        slot_ctx = torch.cat([contract_buffer.hidden, ctx_expanded], dim=-1)  # (B, K, D + ctx_dim)
        flat_slot_ctx = slot_ctx.reshape(B * K, -1)

        new_trigger = self.trigger_head(flat_slot_ctx).reshape(B, K)
        new_validity = self.validity_head(flat_slot_ctx).reshape(B, K)
        new_priority = self.priority_head(flat_slot_ctx).reshape(B, K)

        contract_buffer.trigger_hazard = torch.where(active_mask, new_trigger, contract_buffer.trigger_hazard)
        contract_buffer.validity = torch.where(active_mask, new_validity, contract_buffer.validity)
        contract_buffer.priority = torch.where(active_mask, new_priority, contract_buffer.priority)

        # Age and deadline decay
        contract_buffer.step_age()

    def try_write(
        self, ctx: torch.Tensor, contract_buffer: ContractBuffer,
        proposal_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Attempt to write a new contract. Returns write decision info.

        Args:
            ctx: (B, ctx_dim)
            contract_buffer: modified in-place if write succeeds
            proposal_meta: optional external metadata for the proposed contract
        Returns:
            dict with 'write_score', 'did_write' tensors
        """
        B = ctx.shape[0]
        referent, intention, write_score = self.propose_contracts(ctx)

        # Extract metadata or use defaults
        meta_polarity = 1.0
        meta_deadline = 100.0
        meta_priority = 1.0
        if proposal_meta is not None:
            meta_polarity = proposal_meta.get('polarity', torch.ones(B, device=ctx.device))
            meta_deadline = proposal_meta.get('deadline', torch.full((B,), 100.0, device=ctx.device))
            meta_priority = proposal_meta.get('priority', torch.ones(B, device=ctx.device))

        # Check merge: compare candidate with existing slots
        active_mask = contract_buffer.get_active_mask()  # (B, K)
        ref_sim = F.cosine_similarity(
            referent.unsqueeze(1), contract_buffer.referent, dim=-1
        )  # (B, K)
        int_sim = F.cosine_similarity(
            intention.unsqueeze(1), contract_buffer.intention, dim=-1
        )  # (B, K)
        merge_sim = 0.5 * ref_sim + 0.5 * int_sim  # (B, K)
        merge_candidates = (merge_sim > self.merge_threshold) & active_mask  # (B, K)

        # Find the best slot to write into (empty or lowest eviction score)
        eviction_scores = contract_buffer.eviction_score()  # (B, K)
        empty_mask = contract_buffer.state == LifecycleState.EMPTY  # (B, K)
        # Prefer empty slots; among non-empty, pick lowest eviction score
        slot_priority = torch.where(empty_mask, torch.tensor(-1e9, device=ctx.device), eviction_scores)
        target_slot = slot_priority.argmin(dim=-1)  # (B,)

        # Decide whether to write (score > 0 and no close merge candidate)
        has_merge = merge_candidates.any(dim=-1)  # (B,)
        if self._ablation_always_write:
            should_write = ~has_merge  # always write if no merge (ablation: no_write)
        else:
            should_write = (write_score.squeeze(-1) > 0) & (~has_merge)

        # Execute writes
        for b in range(B):
            if should_write[b]:
                s = target_slot[b].item()
                pol = meta_polarity[b].item() if isinstance(meta_polarity, torch.Tensor) else meta_polarity
                if self._ablation_no_negative:
                    pol = 1.0  # force positive polarity
                dl = meta_deadline[b].item() if isinstance(meta_deadline, torch.Tensor) else meta_deadline
                pri = meta_priority[b].item() if isinstance(meta_priority, torch.Tensor) else meta_priority
                contract_buffer.write_slot(b, s, referent[b], intention[b],
                                           deadline=dl, priority=pri, polarity=pol)
            elif has_merge[b]:
                # Merge: update the most similar existing slot
                merge_slot = (merge_sim[b] * active_mask[b].float()).argmax().item()
                # Exponential moving average merge
                alpha = 0.3
                contract_buffer.referent[b, merge_slot] = (
                    (1 - alpha) * contract_buffer.referent[b, merge_slot] + alpha * referent[b]
                )
                contract_buffer.intention[b, merge_slot] = (
                    (1 - alpha) * contract_buffer.intention[b, merge_slot] + alpha * intention[b]
                )

        return {
            'write_score': write_score,
            'did_write': should_write.float(),
            'did_merge': has_merge.float(),
        }

    def try_fire(self, contract_buffer: ContractBuffer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check which contracts should fire or keep reminding.

        FIRED contracts continue producing reminders until completed/canceled.
        Only PENDING->FIRED is a new state transition; already-FIRED slots
        also contribute reminders each step.

        Returns:
            reminder_token: (B, ctx_dim) aggregated reminder for policy injection
            new_fire_mask: (B, K) which slots NEWLY fired this step
        """
        fire_scores = contract_buffer.fire_score()  # (B, K)
        live_mask = contract_buffer.get_live_mask()  # PENDING or FIRED
        pending_mask = contract_buffer.get_pending_mask()

        fireable = (fire_scores > self.fire_threshold) & live_mask
        # New fires: only from PENDING slots
        new_fire_mask = fireable & pending_mask
        # Remind mask: all live slots above threshold (including already FIRED)
        remind_mask = fireable

        # Build reminder tokens from reminding contracts
        B, K = remind_mask.shape
        reminder_input = torch.cat([
            contract_buffer.referent,
            contract_buffer.intention,
            contract_buffer.polarity.unsqueeze(-1),
        ], dim=-1)  # (B, K, 2D+1)

        reminder_raw = self.reminder_proj(reminder_input)  # (B, K, ctx_dim)
        weights = fire_scores * remind_mask.float()
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights_norm = weights / weights_sum
        reminder_token = (reminder_raw * weights_norm.unsqueeze(-1)).sum(dim=1)  # (B, ctx_dim)

        # Only transition PENDING -> FIRED (not re-set already FIRED)
        contract_buffer.state = torch.where(
            new_fire_mask,
            torch.tensor(LifecycleState.FIRED, device=new_fire_mask.device),
            contract_buffer.state
        )

        return reminder_token, new_fire_mask

    def try_cancel(self, ctx: torch.Tensor, contract_buffer: ContractBuffer) -> Dict[str, torch.Tensor]:
        """Check which contracts should be canceled/completed/superseded.

        Args:
            ctx: (B, ctx_dim)
            contract_buffer: modified in-place
        Returns:
            dict with 'cancel_logits', 'cancel_mask'
        """
        B, K, D = contract_buffer.hidden.shape
        active_or_fired = (contract_buffer.state == LifecycleState.PENDING) | \
                          (contract_buffer.state == LifecycleState.FIRED)

        ctx_expanded = ctx.unsqueeze(1).expand(B, K, -1)
        slot_ctx = torch.cat([contract_buffer.hidden, ctx_expanded], dim=-1)
        flat_slot_ctx = slot_ctx.reshape(B * K, -1)

        cancel_logits = self.cancel_head(flat_slot_ctx).reshape(B, K, 3)
        # 3 classes: completed, canceled, superseded
        cancel_probs = torch.softmax(cancel_logits, dim=-1)

        # Check validity threshold for cancellation
        low_validity = contract_buffer.validity < self.cancel_threshold
        should_cancel = low_validity & active_or_fired

        # Also cancel if cancel head is confident
        max_cancel_prob, max_cancel_class = cancel_probs.max(dim=-1)
        head_cancel = (max_cancel_prob > 0.7) & active_or_fired

        cancel_mask = should_cancel | head_cancel

        # Execute cancellations
        for b in range(B):
            for k in range(K):
                if cancel_mask[b, k]:
                    cls = max_cancel_class[b, k].item()
                    if cls == 0:
                        state = LifecycleState.COMPLETED
                    elif cls == 1:
                        state = LifecycleState.CANCELED
                    else:
                        state = LifecycleState.SUPERSEDED
                    contract_buffer.retire_slot(b, k, state)

        return {
            'cancel_logits': cancel_logits,
            'cancel_mask': cancel_mask,
        }

    def step(
        self,
        obs_features: torch.Tensor,
        contract_buffer: ContractBuffer,
        proposal_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full ProMem step: encode -> write -> update -> fire -> cancel.

        Args:
            obs_features: (B, ctx_dim) features from the backbone policy
            contract_buffer: modified in-place
            proposal_meta: optional dict with 'polarity' (B,), 'deadline' (B,),
                           'priority' (B,) from instruction parser / oracle.
                           If absent, defaults are used.
        Returns:
            reminder_token: (B, ctx_dim) to inject into the action policy
            lifecycle_info: dict of tensors for loss computation.
                            All predictions are SNAPSHOTS taken before buffer mutation
                            so gradients flow through the correct values.
        """
        # 1. Encode context
        ctx = self.encode_context(obs_features, contract_buffer)

        # 2. Try to write new contract
        write_info = self.try_write(ctx, contract_buffer, proposal_meta)

        # 3. Update existing contracts (mutates buffer heads)
        if not self._ablation_freeze_contracts:
            self.update_slots(ctx, contract_buffer)

        # --- Snapshot predictions BEFORE fire/cancel mutate state ---
        snap_trigger = contract_buffer.trigger_hazard.clone()
        snap_validity = contract_buffer.validity.clone()
        snap_priority = contract_buffer.priority.clone()
        snap_live_mask = contract_buffer.get_live_mask().clone()
        snap_contract_id = contract_buffer.contract_id.clone()

        # 4. Fire triggered contracts (mutates PENDING -> FIRED)
        reminder_token, new_fire_mask = self.try_fire(contract_buffer)

        # 5. Cancel invalidated contracts (mutates -> COMPLETED/CANCELED/SUPERSEDED)
        if not self._ablation_no_cancel:
            cancel_info = self.try_cancel(ctx, contract_buffer)
        else:
            cancel_info = {
                'cancel_logits': torch.zeros(contract_buffer.B, contract_buffer.K, 3, device=ctx.device),
                'cancel_mask': torch.zeros(contract_buffer.B, contract_buffer.K, dtype=torch.bool, device=ctx.device),
            }

        lifecycle_info = {
            **write_info,
            'new_fire_mask': new_fire_mask,
            **cancel_info,
            # Snapshots for loss computation (pre-mutation)
            'trigger_hazard': snap_trigger,
            'validity': snap_validity,
            'priority': snap_priority,
            'live_mask': snap_live_mask,
            'contract_id': snap_contract_id,
        }

        return reminder_token, lifecycle_info
