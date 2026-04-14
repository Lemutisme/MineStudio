"""
ProMemPolicy: Wraps any MinePolicy with Prospective Memory.

The contract buffer is serialized into the recurrent state list so it
travels correctly through MineStudio's rollout workers (merge_state/split_state).
ProMem runs at every timestep in the sequence, not just the last one.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

from minestudio.models.base_policy import MinePolicy, recursive_tensor_op
from minestudio.models.memory.contract import ContractBuffer, LifecycleState, _NUM_PM_STATE_TENSORS
from minestudio.models.memory.promem import ProMemModule


class ProMemPolicy(MinePolicy):
    """Policy wrapper that adds prospective memory to any base MinePolicy.

    State layout: base_state_tensors... + pm_state_tensors (12 tensors)

    The contract buffer is serialized into / deserialized from the state list
    at every forward call, ensuring correct handling by rollout workers.
    """

    def __init__(
        self,
        base_policy: MinePolicy,
        num_slots: int = 4,
        embed_dim: int = 256,
        fire_threshold: float = 0.5,
        cancel_threshold: float = 0.2,
        merge_threshold: float = 0.8,
    ):
        hiddim = base_policy.net.hidsize if hasattr(base_policy, 'net') else 512
        # MinePolicy doesn't store action_space; pass None to use the default
        super().__init__(hiddim=hiddim, action_space=None)

        self.base_policy = base_policy
        self.num_slots = num_slots
        self.embed_dim = embed_dim

        # Count how many tensors the base policy uses for state
        dummy_state = base_policy.initial_state(batch_size=1)
        self._base_state_len = len(dummy_state) if dummy_state else 0

        # ProMem module
        self.promem = ProMemModule(
            num_slots=num_slots,
            embed_dim=embed_dim,
            ctx_dim=hiddim,
            fire_threshold=fire_threshold,
            cancel_threshold=cancel_threshold,
            merge_threshold=merge_threshold,
        )

        # Feature projection: base policy latent -> ProMem context
        self.feature_proj = nn.Linear(hiddim, hiddim)

        # Reminder injection into the latent space
        self.reminder_gate = nn.Sequential(
            nn.Linear(2 * hiddim, hiddim),
            nn.Sigmoid(),
        )
        self.reminder_proj = nn.Linear(hiddim, hiddim)

        # Reuse base policy heads
        self.pi_head = base_policy.pi_head
        self.value_head = base_policy.value_head

        # Cache for training
        self._lifecycle_info_sequence = None  # List of per-timestep lifecycle_info

    # ---- State management ----

    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        """Base state + serialized empty contract buffer."""
        base_state = self.base_policy.initial_state(batch_size)
        if base_state is None:
            base_state = []
        B = batch_size or 1
        # Infer device from base state tensors if available, else from parameters
        if base_state:
            dev = base_state[0].device
        elif len(list(self.parameters())) > 0:
            dev = next(self.parameters()).device
        else:
            dev = torch.device('cpu')
        empty_buf = ContractBuffer(B, self.num_slots, self.embed_dim, device=dev)
        return base_state + empty_buf.to_state_list()

    def _split_state(self, state_in: List[torch.Tensor]):
        """Split combined state into base state + contract buffer."""
        n = self._base_state_len
        base_state = state_in[:n] if n > 0 else []
        pm_tensors = state_in[n:n + _NUM_PM_STATE_TENSORS]
        buf = ContractBuffer.from_state_list(pm_tensors, self.num_slots, self.embed_dim)
        return base_state, buf

    def _join_state(self, base_state: List[torch.Tensor], buf: ContractBuffer) -> List[torch.Tensor]:
        """Join base state + contract buffer into combined state."""
        return (base_state if base_state else []) + buf.to_state_list()

    def merge_state(self, states: List[List[torch.Tensor]]) -> Optional[List[torch.Tensor]]:
        """Merge per-env states into a batch. Delegates to base for base portion,
        then concatenates PM tensors along batch dim."""
        if not states:
            return None
        n = self._base_state_len
        # Merge base state
        base_states = [s[:n] for s in states]
        if n > 0 and hasattr(self.base_policy, 'merge_state'):
            merged_base = self.base_policy.merge_state(base_states)
        elif n > 0:
            merged_base = [torch.cat([s[i] for s in base_states], dim=0) for i in range(n)]
        else:
            merged_base = []
        # Merge PM tensors
        pm_parts = [s[n:n + _NUM_PM_STATE_TENSORS] for s in states]
        merged_pm = [torch.cat([p[i] for p in pm_parts], dim=0) for i in range(_NUM_PM_STATE_TENSORS)]
        return merged_base + merged_pm

    def split_state(self, state: List[torch.Tensor], split_num: int) -> Optional[List[List[torch.Tensor]]]:
        """Split batched state into per-env states."""
        if state is None:
            return None
        n = self._base_state_len
        # Split base state
        if n > 0 and hasattr(self.base_policy, 'split_state'):
            base_splits = self.base_policy.split_state(state[:n], split_num)
        elif n > 0:
            base_splits = [[s[i:i+1] for s in state[:n]] for i in range(split_num)]
        else:
            base_splits = [[] for _ in range(split_num)]
        # Split PM tensors
        pm_tensors = state[n:n + _NUM_PM_STATE_TENSORS]
        pm_splits = [[t[i:i+1] for t in pm_tensors] for i in range(split_num)]
        return [base_splits[i] + pm_splits[i] for i in range(split_num)]

    # ---- Forward pass ----

    def forward(
        self,
        input: Dict[str, Any],
        state_in: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Forward pass with prospective memory injected between backbone and heads.

        1. Run base backbone (ImpalaCNN + recurrence) -> pre-head latent (B,T,H)
        2. For each timestep: run ProMem lifecycle step -> reminder token
        3. ADD reminder into pre-head latent (the actual injection)
        4. Apply pi_head and value_head on the augmented latent
        """
        # --- Deserialize state ---
        if state_in is None:
            state_in = self.initial_state(batch_size=1)
        base_state, buf = self._split_state(state_in)

        # --- Run base backbone directly (NOT base_policy.forward) ---
        # This gives us the pre-head latent so we can inject ProMem before heads.
        base_net = self.base_policy.net
        B, T = input["image"].shape[:2]

        # Build first flags and base state
        first = kwargs.get('context', {}).get('first', None)
        if first is None:
            first = input.get('first', torch.zeros(B, T, dtype=torch.bool, device=input["image"].device))
        if base_state:
            net_state_in = base_state
        else:
            net_state_in = base_net.initial_state(B)

        # Run backbone: image preprocessing + CNN + recurrence
        (pi_h, v_h), base_state_out = base_net(
            input, net_state_in, context={"first": first}
        )
        # pi_h, v_h: (B, T, hidsize) — pre-head latents

        # --- ProMem injection ---
        obs_features = self.feature_proj(pi_h)  # (B, T, hidsize)

        # Ensure buffer matches batch size
        if buf.B != B:
            buf = ContractBuffer(B, self.num_slots, self.embed_dim, pi_h.device)

        # Run ProMem for each timestep
        reminders = []
        lifecycle_seq = []

        for t in range(T):
            # Reset buffer on episode boundaries
            if first is not None:
                first_t = first[:, t] if first.dim() >= 2 else first
                for b in range(B):
                    if first_t[b]:
                        buf.reset_batch_element(b)

            feat_t = obs_features[:, t, :]  # (B, hidsize)
            reminder_t, lifecycle_t = self.promem.step(feat_t, buf)
            reminders.append(reminder_t)
            lifecycle_seq.append(lifecycle_t)

        reminder_stack = torch.stack(reminders, dim=1)  # (B, T, hidsize)

        # Gated reminder injection into latent
        gate_in = torch.cat([obs_features, reminder_stack], dim=-1)  # (B, T, 2H)
        gate_val = self.reminder_gate(gate_in)  # (B, T, H)
        reminder_bias = self.reminder_proj(gate_val * reminder_stack)  # (B, T, H)

        # *** THE KEY INJECTION: add reminder directly to pre-head latent ***
        augmented_pi_h = pi_h + reminder_bias
        augmented_v_h = v_h  # value head sees original latent (no reminder bias)

        # --- Apply heads on augmented latent ---
        pi_logits = self.pi_head(augmented_pi_h)
        vpred = self.value_head(augmented_v_h)

        # Cache lifecycle for loss computation
        self._lifecycle_info_sequence = lifecycle_seq

        result = {
            'pi_logits': pi_logits,
            'vpred': vpred,
            'pm_reminder_bias': reminder_bias,
            'pm_active_contracts': buf.get_live_mask().sum(dim=-1).float(),
        }

        # --- Serialize state out ---
        buf.detach()
        state_out = self._join_state(base_state_out if base_state_out else [], buf)

        return result, state_out

    @torch.inference_mode()
    def get_action(
        self,
        input: Dict[str, Any],
        state_in: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
        input_shape: str = 'BT*',
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Get action with prospective memory.

        Mirrors base MinePolicy.get_action: batchify on entry, unbatchify on exit.
        """
        import numpy as np
        from minestudio.models.base_policy import dict_map

        def _to_tensor_batchify(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
                if x.dim() == 3:  # (H,W,C) -> (1,1,H,W,C)
                    return x.unsqueeze(0).unsqueeze(0)
                elif x.dim() == 4:  # (B,H,W,C) -> (B,1,H,W,C)
                    return x.unsqueeze(1)
                return x.unsqueeze(0).unsqueeze(0)
            return x

        if input_shape == '*':
            input = dict_map(_to_tensor_batchify, input)
            if state_in is None:
                state_in = self.initial_state(batch_size=1)
            else:
                # Re-add batch dim that was stripped on previous call
                state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)

        latents, state_out = self.forward(input, state_in, **kwargs)

        pi_logits = latents['pi_logits']
        action = self.pi_head.sample(pi_logits, deterministic=deterministic)
        self.vpred = latents.get('vpred', None)
        self.cache_latents = latents

        if input_shape == 'BT*':
            return action, state_out
        elif input_shape == '*':
            action = dict_map(lambda t: t[0][0] if isinstance(t, torch.Tensor) else t, action)
            state_out = recursive_tensor_op(lambda x: x[0], state_out)
            return action, state_out
        else:
            raise NotImplementedError(f"Unknown input_shape: {input_shape}")

    # ---- Training helpers ----

    def get_lifecycle_info_sequence(self) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Get per-timestep lifecycle info from last forward (for loss computation).
        Returns list of T dicts, each with keys from ProMemModule.step()."""
        return self._lifecycle_info_sequence

    def get_stacked_lifecycle_info(self) -> Optional[Dict[str, torch.Tensor]]:
        """Stack per-timestep lifecycle info into (B, T, ...) tensors for loss."""
        seq = self._lifecycle_info_sequence
        if seq is None or len(seq) == 0:
            return None
        keys = seq[0].keys()
        stacked = {}
        for k in keys:
            tensors = [step[k] for step in seq]
            try:
                stacked[k] = torch.stack(tensors, dim=1)  # (B, T, ...)
            except RuntimeError:
                pass  # skip keys with incompatible shapes
        return stacked

    def get_contract_summary(self) -> Dict[str, Any]:
        """Human-readable summary for logging."""
        state = self.initial_state(1)
        _, buf = self._split_state(state)
        live = buf.get_live_mask()[0]
        return {
            'active_contracts': live.sum().item(),
            'slots': [
                {
                    'slot': k,
                    'state': LifecycleState(buf.state[0, k].item()).name,
                    'trigger': f"{buf.trigger_hazard[0, k].item():.3f}",
                    'validity': f"{buf.validity[0, k].item():.3f}",
                    'age': int(buf.age[0, k].item()),
                }
                for k in range(self.num_slots)
                if buf.state[0, k].item() != LifecycleState.EMPTY
            ],
        }
