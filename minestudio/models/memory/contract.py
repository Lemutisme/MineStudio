"""
Contract data structures for Prospective Memory Slots.

A contract is a structured memory object encoding a deferred intention
with a multimodal trigger, validity estimate, priority, and lifecycle state.
"""
import torch
import torch.nn as nn
from enum import IntEnum
from typing import Optional, List, Dict, Tuple


class LifecycleState(IntEnum):
    EMPTY = 0
    PENDING = 1
    FIRED = 2
    COMPLETED = 3
    CANCELED = 4
    SUPERSEDED = 5


# Number of tensor fields for state serialization
_NUM_PM_STATE_TENSORS = 12


class ContractBuffer:
    """Fixed-capacity buffer of K contract slots, stored as batched tensors.

    All tensors have shape (B, K, ...) where B is batch size and K is slot count.
    This representation is efficient for batched forward passes during training.
    """

    def __init__(self, batch_size: int, num_slots: int, embed_dim: int, device: torch.device):
        self.B = batch_size
        self.K = num_slots
        self.D = embed_dim
        self.device = device
        self.reset()

    def reset(self):
        B, K, D = self.B, self.K, self.D
        self.referent = torch.zeros(B, K, D, device=self.device)
        self.intention = torch.zeros(B, K, D, device=self.device)
        self.trigger_hazard = torch.zeros(B, K, device=self.device)
        self.validity = torch.zeros(B, K, device=self.device)
        self.deadline = torch.zeros(B, K, device=self.device)
        self.priority = torch.zeros(B, K, device=self.device)
        self.polarity = torch.ones(B, K, device=self.device)
        self.state = torch.zeros(B, K, dtype=torch.long, device=self.device)
        self.hidden = torch.zeros(B, K, D, device=self.device)
        self.age = torch.zeros(B, K, device=self.device)
        self.contract_id = torch.zeros(B, K, dtype=torch.long, device=self.device)
        self.next_id = torch.ones(B, 1, dtype=torch.long, device=self.device)

    # --- Live/active masks ---

    def get_live_mask(self) -> torch.Tensor:
        """(B,K) bool: slots that are PENDING or FIRED (still participating)."""
        return (self.state == LifecycleState.PENDING) | (self.state == LifecycleState.FIRED)

    def get_active_mask(self) -> torch.Tensor:
        """(B,K) bool: alias for live_mask (used by update/fire/cancel)."""
        return self.get_live_mask()

    def get_pending_mask(self) -> torch.Tensor:
        """(B,K) bool: slots that are PENDING (eligible for first fire)."""
        return self.state == LifecycleState.PENDING

    # --- Scores ---

    def fire_score(self) -> torch.Tensor:
        """trigger * validity * priority. Shape (B, K)."""
        return self.trigger_hazard * self.validity * self.priority

    def eviction_score(self) -> torch.Tensor:
        """Lower = more likely to evict. Shape (B, K)."""
        live = self.get_live_mask().float()
        imminence = 1.0 / (1.0 + self.deadline)
        return live * self.validity * self.priority * imminence

    # --- Pack for context encoder ---

    def pack(self) -> torch.Tensor:
        """Pack slot features into (B, K, 3D+7), masking retired content."""
        live = self.get_live_mask().float()  # (B, K)
        return torch.cat([
            self.referent * live.unsqueeze(-1),
            self.intention * live.unsqueeze(-1),
            self.hidden * live.unsqueeze(-1),
            self.trigger_hazard.unsqueeze(-1),
            self.validity.unsqueeze(-1),
            self.deadline.unsqueeze(-1),
            self.priority.unsqueeze(-1),
            self.polarity.unsqueeze(-1),
            self.state.float().unsqueeze(-1),
            torch.log1p(self.age).unsqueeze(-1),
        ], dim=-1)

    # --- Slot operations ---

    def write_slot(self, batch_idx: int, slot_idx: int,
                   referent: torch.Tensor, intention: torch.Tensor,
                   deadline: float = 100.0, priority: float = 1.0, polarity: float = 1.0):
        """Write a new contract and assign a unique contract_id."""
        self.referent[batch_idx, slot_idx] = referent
        self.intention[batch_idx, slot_idx] = intention
        self.deadline[batch_idx, slot_idx] = deadline
        self.priority[batch_idx, slot_idx] = priority
        self.polarity[batch_idx, slot_idx] = polarity
        self.validity[batch_idx, slot_idx] = 1.0
        self.trigger_hazard[batch_idx, slot_idx] = 0.0
        self.state[batch_idx, slot_idx] = LifecycleState.PENDING
        self.hidden[batch_idx, slot_idx] = 0.0
        self.age[batch_idx, slot_idx] = 0
        # Assign unique id
        self.contract_id[batch_idx, slot_idx] = self.next_id[batch_idx, 0].item()
        self.next_id[batch_idx, 0] += 1

    def retire_slot(self, batch_idx: int, slot_idx: int, new_state: LifecycleState):
        """Retire a slot: set terminal state and zero all content."""
        self.state[batch_idx, slot_idx] = new_state
        self.referent[batch_idx, slot_idx] = 0
        self.intention[batch_idx, slot_idx] = 0
        self.hidden[batch_idx, slot_idx] = 0
        self.trigger_hazard[batch_idx, slot_idx] = 0
        self.validity[batch_idx, slot_idx] = 0
        self.deadline[batch_idx, slot_idx] = 0
        self.priority[batch_idx, slot_idx] = 0
        self.polarity[batch_idx, slot_idx] = 0
        self.age[batch_idx, slot_idx] = 0
        self.contract_id[batch_idx, slot_idx] = 0

    def step_age(self):
        """Increment age for live slots and decay deadline."""
        live = self.get_live_mask().float()
        self.age += live
        self.deadline = torch.clamp(self.deadline - live, min=0.0)

    # --- State serialization for rollout ---

    def to_state_list(self) -> List[torch.Tensor]:
        """Serialize buffer into List[Tensor] for MinePolicy state_in/state_out.

        Returns 12 tensors that can be appended to the base policy's state list.
        Order matters — from_state_list must match.
        """
        return [
            self.referent,                          # 0: (B,K,D)
            self.intention,                         # 1: (B,K,D)
            self.trigger_hazard,                    # 2: (B,K)
            self.validity,                          # 3: (B,K)
            self.deadline,                          # 4: (B,K)
            self.priority,                          # 5: (B,K)
            self.polarity,                          # 6: (B,K)
            self.state.float(),                     # 7: (B,K) as float for merge/split compat
            self.hidden,                            # 8: (B,K,D)
            self.age,                               # 9: (B,K)
            self.contract_id.float(),               # 10: (B,K) as float
            self.next_id.float(),                   # 11: (B,1)
        ]

    @classmethod
    def from_state_list(cls, tensors: List[torch.Tensor], num_slots: int,
                        embed_dim: int) -> 'ContractBuffer':
        """Reconstruct buffer from serialized state list."""
        ref = tensors[0]
        B = ref.shape[0]
        device = ref.device
        buf = cls.__new__(cls)
        buf.B, buf.K, buf.D, buf.device = B, num_slots, embed_dim, device
        buf.referent = tensors[0]
        buf.intention = tensors[1]
        buf.trigger_hazard = tensors[2]
        buf.validity = tensors[3]
        buf.deadline = tensors[4]
        buf.priority = tensors[5]
        buf.polarity = tensors[6]
        buf.state = tensors[7].long()
        buf.hidden = tensors[8]
        buf.age = tensors[9]
        buf.contract_id = tensors[10].long()
        buf.next_id = tensors[11].long()
        return buf

    def detach(self):
        """Detach all tensors from the computation graph."""
        self.referent = self.referent.detach()
        self.intention = self.intention.detach()
        self.trigger_hazard = self.trigger_hazard.detach()
        self.validity = self.validity.detach()
        self.deadline = self.deadline.detach()
        self.priority = self.priority.detach()
        self.polarity = self.polarity.detach()
        self.hidden = self.hidden.detach()
        self.age = self.age.detach()
        self.contract_id = self.contract_id.detach()
        self.next_id = self.next_id.detach()

    def reset_batch_element(self, b: int):
        """Reset all slots for a single batch element (on episode boundary)."""
        self.referent[b] = 0
        self.intention[b] = 0
        self.trigger_hazard[b] = 0
        self.validity[b] = 0
        self.deadline[b] = 0
        self.priority[b] = 0
        self.polarity[b] = 1
        self.state[b] = LifecycleState.EMPTY
        self.hidden[b] = 0
        self.age[b] = 0
        self.contract_id[b] = 0
        self.next_id[b] = 1
