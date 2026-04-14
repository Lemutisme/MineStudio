"""
Prospective Memory Benchmark Callbacks — Fixed for MineStudio's actual info dict.

Info dict keys used:
  - inventory: {slot_id: {'type': str, 'quantity': int}} (36 slots)
  - pickup: {item_name: count} (cumulative)
  - craft_item: {item_name: count} (cumulative)
  - use_item: {item_name: count} (cumulative)
  - mine_block: {block_name: count} (cumulative)
  - kill_entity: {entity_name: count} (cumulative)
  - player_pos: {'x': float, 'y': float, 'z': float, 'pitch': float, 'yaw': float}
  - location_stats: {'xpos', 'ypos', 'zpos', 'pitch', 'yaw', ...}
  - food_level: int (0-20)
  - health: float (0-20)
  - mobs: list of nearby mob dicts

Detection pattern (from rewards.py): compare info[event][obj] between steps to detect deltas.
"""
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from minestudio.simulator.callbacks.callback import MinecraftCallback


# ============================================================
# Shared utilities
# ============================================================

def get_inventory_items(info: dict) -> Dict[str, int]:
    """Extract {item_type: total_quantity} from slot-based inventory."""
    items = defaultdict(int)
    inv = info.get('inventory', {})
    for slot_id, slot in inv.items():
        if isinstance(slot, dict):
            t = slot.get('type', 'none')
            q = slot.get('quantity', 0)
            if t != 'none' and q > 0:
                items[t] += q
    return dict(items)


def get_stat_delta(info: dict, prev_info: dict, event_type: str, obj: str) -> int:
    """Get the delta of a cumulative stat between steps."""
    def _get(d, et, o):
        val = d.get(et, {})
        if isinstance(val, dict):
            return val.get(o, 0)
        return 0
    return _get(info, event_type, obj) - _get(prev_info, event_type, obj)


def get_player_pos(info: dict) -> Tuple[float, float, float]:
    """Get player (x, y, z) from info."""
    pp = info.get('player_pos', {})
    if isinstance(pp, dict):
        return pp.get('x', 0), pp.get('y', 0), pp.get('z', 0)
    ls = info.get('location_stats', {})
    return ls.get('xpos', 0), ls.get('ypos', 0), ls.get('zpos', 0)


class PMOracleEvents:
    """Container for oracle lifecycle events."""

    def __init__(self, num_slots: int = 4):
        self.K = num_slots
        self.reset()

    def reset(self):
        self.events = {
            'oracle_fire': np.zeros(self.K, dtype=np.float32),
            'oracle_cancel': np.zeros(self.K, dtype=np.float32),
            'oracle_valid': np.ones(self.K, dtype=np.float32),
            'oracle_should_write': np.array([0.0], dtype=np.float32),
            'oracle_contract_id': np.zeros(self.K, dtype=np.int64),
        }
        self._next_contract_id = 1

    def register_contract(self, slot_id: int) -> int:
        cid = self._next_contract_id
        self._next_contract_id += 1
        self.events['oracle_contract_id'][slot_id] = cid
        self.events['oracle_valid'][slot_id] = 1.0
        return cid

    def signal_fire(self, slot_id: int):
        self.events['oracle_fire'][slot_id] = 1.0

    def signal_cancel(self, slot_id: int, reason: int = 2):
        self.events['oracle_cancel'][slot_id] = reason
        self.events['oracle_valid'][slot_id] = 0.0

    def signal_completed(self, slot_id: int):
        self.events['oracle_cancel'][slot_id] = 1
        self.events['oracle_valid'][slot_id] = 0.0

    def signal_invalid(self, slot_id: int):
        self.events['oracle_valid'][slot_id] = 0.0

    def signal_should_write(self):
        self.events['oracle_should_write'][0] = 1.0

    def get_events(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.events.items()}

    def clear_step_events(self):
        self.events['oracle_fire'][:] = 0.0
        self.events['oracle_cancel'][:] = 0.0
        self.events['oracle_should_write'][0] = 0.0


class PMMetrics:
    """Tracks PM-specific metrics during an episode."""

    def __init__(self, tolerance_delta: int = 50):
        self.delta = tolerance_delta
        self.reset()

    def reset(self):
        self.triggers = []
        self.actions = []
        self.fires = []
        self.total_steps = 0

    def record_trigger(self, step: int, slot_id: int, valid: bool):
        self.triggers.append((step, slot_id, valid))

    def record_action(self, step: int, slot_id: int):
        self.actions.append((step, slot_id))

    def record_fire(self, step: int, slot_id: int, obsolete: bool):
        self.fires.append((step, slot_id, obsolete))

    def compute(self) -> Dict[str, float]:
        metrics = {}
        valid_triggers = [(s, sid) for s, sid, v in self.triggers if v]
        if valid_triggers:
            recalled = 0
            latencies = []
            for t_step, t_slot in valid_triggers:
                matching = [a_step for a_step, a_slot in self.actions
                           if a_slot == t_slot and 0 <= a_step - t_step <= self.delta]
                if matching:
                    recalled += 1
                    latencies.append(min(matching) - t_step)
            metrics['pm_recall_at_delta'] = recalled / len(valid_triggers)
            metrics['trigger_to_action_latency'] = float(np.mean(latencies)) if latencies else float('inf')
        else:
            metrics['pm_recall_at_delta'] = 0.0
            metrics['trigger_to_action_latency'] = 0.0

        if self.fires:
            obsolete = sum(1 for _, _, obs in self.fires if obs)
            metrics['obsolete_fire_rate'] = obsolete / len(self.fires)
        else:
            metrics['obsolete_fire_rate'] = 0.0

        return metrics


# ============================================================
# Family 1: Delayed Process (Time-Based PM)
# ============================================================

class DelayedProcessCallback(MinecraftCallback):
    """Agent must place ore in furnace, leave, return to collect smelted output.

    Detection: uses pickup['iron_ingot'] delta (smelted output collected)
    and use_item['raw_iron'] delta (ore placed in furnace).
    Since simulating actual furnace smelting is complex, we SIMULATE the
    delayed reward: after the agent uses raw_iron (places in furnace),
    we start a timer. When the timer expires, we give the reward if the
    agent is near the furnace. This tests PM: remember to return.
    """

    def __init__(self, smelt_delay_range=(120, 300), add_perturbation=True,
                 perturbation_prob=0.05, auto_start_smelt=True,
                 collection_radius=5.0, forced_excursion_dist=12):
        super().__init__()
        self.smelt_delay_range = smelt_delay_range
        self.add_perturbation = add_perturbation
        self.perturbation_prob = perturbation_prob
        self.auto_start_smelt = auto_start_smelt
        self.collection_radius = collection_radius
        self.forced_excursion_dist = forced_excursion_dist  # teleport agent away after start
        self.oracle = PMOracleEvents()
        self.metrics = PMMetrics()

    def after_reset(self, sim, obs, info):
        self.oracle.reset()
        self.metrics.reset()
        self.step_count = 0
        self.prev_info = info.copy()
        self.furnace_pos = None
        self.smelt_started = False
        self.smelt_ready = False
        self.smelt_ready_step = 0
        self.collected = False
        self.destroyed = False

        # Place furnace and give items
        sim.env.execute_cmd("/setblock ~3 ~ ~2 furnace")
        sim.env.execute_cmd("/give @s minecraft:raw_iron 4")
        sim.env.execute_cmd("/give @s minecraft:coal 4")

        # Record furnace position relative to spawn
        px, py, pz = get_player_pos(info)
        self.furnace_pos = (px + 3, py, pz + 2)

        obs['task'] = {
            'name': 'smelt_and_collect',
            'text': 'Place iron ore in the nearby furnace to smelt it. '
                    'Go explore while it smelts. Return to collect the iron ingots.',
        }

        # Auto-start smelting: simulate the primary action so we only test PM
        if self.auto_start_smelt:
            delay = random.randint(*self.smelt_delay_range)
            self.smelt_started = True
            self.smelt_ready_step = delay  # relative to step 0
            self.oracle.register_contract(0)
            self.oracle.signal_should_write()
        self._needs_excursion = self.forced_excursion_dist > 0

        info['pm_oracle'] = self.oracle.get_events()
        info['pm_task_family'] = 'delayed_process'
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.step_count += 1
        self.oracle.clear_step_events()

        # Forced excursion: teleport on first step (after env is ready)
        if self._needs_excursion and self.step_count == 1:
            self._needs_excursion = False
            d = self.forced_excursion_dist
            sim.env.execute_cmd(f"/tp @s ~{d} ~ ~{d}")

        # Detect: agent used raw_iron (placed in furnace)
        if not self.smelt_started:
            delta = get_stat_delta(info, self.prev_info, 'use_item', 'raw_iron')
            if delta > 0:
                self.smelt_started = True
                delay = random.randint(*self.smelt_delay_range)
                self.smelt_ready_step = self.step_count + delay
                self.oracle.register_contract(0)
                self.oracle.signal_should_write()

        # Timer: smelting complete
        if self.smelt_started and not self.smelt_ready and not self.destroyed:
            if self.step_count >= self.smelt_ready_step:
                self.smelt_ready = True
                self.oracle.signal_fire(0)
                self.metrics.record_trigger(self.step_count, 0, True)

        # Perturbation: furnace destroyed
        if (self.add_perturbation and self.smelt_started and not self.smelt_ready
                and not self.destroyed and random.random() < self.perturbation_prob):
            self.destroyed = True
            self.oracle.signal_cancel(0, reason=2)

        # Check: agent returns to furnace after smelting
        if self.smelt_ready and not self.collected and not self.destroyed:
            px, py, pz = get_player_pos(info)
            fx, fy, fz = self.furnace_pos
            dist = ((px - fx)**2 + (pz - fz)**2) ** 0.5
            if dist < self.collection_radius:
                self.collected = True
                self.oracle.signal_completed(0)
                self.metrics.record_action(self.step_count, 0)
                reward += 10.0
                # Actually give the ingot as reward
                sim.env.execute_cmd("/give @s minecraft:iron_ingot 1")

        self.prev_info = info.copy()
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_metrics'] = self.metrics.compute()
        info['pm_step'] = self.step_count
        return obs, reward, terminated, truncated, info


# ============================================================
# Family 2: Opportunistic Event (Event-Based PM)
# ============================================================

class OpportunisticEventCallback(MinecraftCallback):
    """Agent gets instruction 'if you see a cow, kill it'. Cow spawns randomly.

    Detection: kill_entity['cow'] delta > 0.
    Uses cow+kill instead of sheep+shear for reliability (VPT knows how to attack).
    """

    def __init__(self, target_entity="cow", spawn_delay_range=(100, 500)):
        super().__init__()
        self.target_entity = target_entity
        self.spawn_delay_range = spawn_delay_range
        self.oracle = PMOracleEvents()
        self.metrics = PMMetrics()

    def after_reset(self, sim, obs, info):
        self.oracle.reset()
        self.metrics.reset()
        self.step_count = 0
        self.prev_info = info.copy()
        self.entity_spawned = False
        self.action_completed = False
        self.spawn_step = random.randint(*self.spawn_delay_range)

        sim.env.execute_cmd("/give @s minecraft:iron_sword 1")

        obs['task'] = {
            'name': f'opportunistic_{self.target_entity}',
            'text': f'Explore the area. If you encounter a {self.target_entity}, '
                    f'immediately kill it.',
        }
        self.oracle.register_contract(0)
        self.oracle.signal_should_write()

        info['pm_oracle'] = self.oracle.get_events()
        info['pm_task_family'] = 'opportunistic_event'
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.step_count += 1
        self.oracle.clear_step_events()

        # Spawn entity at random time
        if not self.entity_spawned and self.step_count >= self.spawn_step:
            self.entity_spawned = True
            sim.env.execute_cmd(
                f"/execute as @p at @p run summon minecraft:{self.target_entity} ~3 ~0 ~3"
            )
            # Entity is now near the agent — this IS the trigger
            self.oracle.signal_fire(0)
            self.metrics.record_trigger(self.step_count, 0, True)

        # Detect kill via cumulative counter delta
        if self.entity_spawned and not self.action_completed:
            delta = get_stat_delta(info, self.prev_info, 'kill_entity', self.target_entity)
            if delta > 0:
                self.action_completed = True
                self.oracle.signal_completed(0)
                self.metrics.record_action(self.step_count, 0)
                reward += 15.0

        self.prev_info = info.copy()
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_metrics'] = self.metrics.compute()
        return obs, reward, terminated, truncated, info


# ============================================================
# Family 3: Activity-Based Handoff
# ============================================================

class ActivityHandoffCallback(MinecraftCallback):
    """Agent must mine N blocks, then craft an item.

    Detection: mine_block cumulative counter >= target, then craft_item delta.
    """

    def __init__(self, target_block="stone", target_count=8):
        super().__init__()
        self.target_block = target_block
        self.target_count = target_count
        self.oracle = PMOracleEvents()
        self.metrics = PMMetrics()

    def after_reset(self, sim, obs, info):
        self.oracle.reset()
        self.metrics.reset()
        self.step_count = 0
        self.prev_info = info.copy()
        self.milestone_reached = False
        self.handoff_completed = False
        self.base_count = 0

        sim.env.execute_cmd("/give @s minecraft:stone_pickaxe 1")
        sim.env.execute_cmd("/give @s minecraft:stick 4")

        # Record baseline mine count
        self.base_count = info.get('mine_block', {}).get(self.target_block, 0)

        obs['task'] = {
            'name': 'activity_handoff',
            'text': f'Mine {self.target_count} {self.target_block} blocks. '
                    f'After that, immediately craft stone_stairs.',
        }
        self.oracle.register_contract(0)
        self.oracle.signal_should_write()
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_task_family'] = 'activity_handoff'
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.step_count += 1
        self.oracle.clear_step_events()

        current_count = info.get('mine_block', {}).get(self.target_block, 0)
        mined = current_count - self.base_count

        if not self.milestone_reached and mined >= self.target_count:
            self.milestone_reached = True
            self.oracle.signal_fire(0)
            self.metrics.record_trigger(self.step_count, 0, True)

        if self.milestone_reached and not self.handoff_completed:
            delta = get_stat_delta(info, self.prev_info, 'craft_item', 'stone_stairs')
            if delta > 0:
                self.handoff_completed = True
                self.oracle.signal_completed(0)
                self.metrics.record_action(self.step_count, 0)
                reward += 15.0

        self.prev_info = info.copy()
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_metrics'] = self.metrics.compute()
        return obs, reward, terminated, truncated, info


# ============================================================
# Family 4: Internal Threshold
# ============================================================

class InternalThresholdCallback(MinecraftCallback):
    """Agent must eat when food_level drops below threshold.

    Detection: food_level from info dict (top-level key, not nested).
    """

    def __init__(self, threshold=10, direction="below"):
        super().__init__()
        self.threshold = threshold
        self.direction = direction
        self.oracle = PMOracleEvents()
        self.metrics = PMMetrics()

    def after_reset(self, sim, obs, info):
        self.oracle.reset()
        self.metrics.reset()
        self.step_count = 0
        self.prev_food = info.get('food_level', 20)
        self.threshold_crossed = False
        self.action_taken = False

        sim.env.execute_cmd("/give @s minecraft:cooked_beef 8")

        obs['task'] = {
            'name': 'threshold_food',
            'text': f'Explore and mine. When your hunger drops below '
                    f'{self.threshold}, immediately eat food.',
        }
        self.oracle.register_contract(0)
        self.oracle.signal_should_write()
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_task_family'] = 'internal_threshold'
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.step_count += 1
        self.oracle.clear_step_events()

        food = info.get('food_level', 20)

        crossed = food < self.threshold if self.direction == "below" else food > self.threshold
        if not self.threshold_crossed and crossed:
            self.threshold_crossed = True
            self.oracle.signal_fire(0)
            self.metrics.record_trigger(self.step_count, 0, True)

        if self.threshold_crossed and not self.action_taken:
            if food > self.prev_food:  # food level increased = ate something
                self.action_taken = True
                self.oracle.signal_completed(0)
                self.metrics.record_action(self.step_count, 0)
                reward += 10.0

        self.prev_food = food
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_metrics'] = self.metrics.compute()
        return obs, reward, terminated, truncated, info


# ============================================================
# Family 5: Place Return (Referent-Bound PM)
# ============================================================

class PlaceReturnCallback(MinecraftCallback):
    """Agent must gather items then return to a marked location.

    Detection: player_pos proximity to the marked position + mine_block count.
    """

    def __init__(self, target_block="stone", min_count=5, return_radius=4.0):
        super().__init__()
        self.target_block = target_block
        self.min_count = min_count
        self.return_radius = return_radius
        self.oracle = PMOracleEvents()
        self.metrics = PMMetrics()

    def after_reset(self, sim, obs, info):
        self.oracle.reset()
        self.metrics.reset()
        self.step_count = 0
        self.gather_complete = False
        self.returned = False

        # Mark return point = spawn position
        self.return_pos = get_player_pos(info)
        self.base_mine = info.get('mine_block', {}).get(self.target_block, 0)

        # Place a visible marker
        sim.env.execute_cmd("/setblock ~ ~2 ~ glowstone")
        sim.env.execute_cmd("/give @s minecraft:stone_pickaxe 1")

        obs['task'] = {
            'name': 'place_return',
            'text': f'Remember this location (marked with glowstone above you). '
                    f'Go mine {self.min_count} {self.target_block}, then return here.',
        }
        self.oracle.register_contract(0)
        self.oracle.signal_should_write()
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_task_family'] = 'place_return'
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.step_count += 1
        self.oracle.clear_step_events()

        current_mine = info.get('mine_block', {}).get(self.target_block, 0)
        mined = current_mine - self.base_mine

        if not self.gather_complete and mined >= self.min_count:
            self.gather_complete = True
            self.oracle.signal_fire(0)
            self.metrics.record_trigger(self.step_count, 0, True)

        if self.gather_complete and not self.returned:
            px, py, pz = get_player_pos(info)
            rx, ry, rz = self.return_pos
            dist = ((px - rx)**2 + (pz - rz)**2) ** 0.5
            if dist < self.return_radius:
                self.returned = True
                self.oracle.signal_completed(0)
                self.metrics.record_action(self.step_count, 0)
                reward += 20.0

        info['pm_oracle'] = self.oracle.get_events()
        info['pm_metrics'] = self.metrics.compute()
        return obs, reward, terminated, truncated, info


# ============================================================
# Family 6: Negative Contract
# ============================================================

class NegativeContractCallback(MinecraftCallback):
    """Agent must avoid a dangerous zone until it has a specific tool.

    Detection: player_pos proximity + inventory check for required item.
    """

    def __init__(self, danger_offset=(8, 8), required_item="shield"):
        super().__init__()
        self.danger_offset = danger_offset
        self.required_item = required_item
        self.oracle = PMOracleEvents()
        self.metrics = PMMetrics()

    def after_reset(self, sim, obs, info):
        self.oracle.reset()
        self.metrics.reset()
        self.step_count = 0
        self.first_visit_failed = False
        self.has_required = False
        self.returned_success = False

        px, py, pz = get_player_pos(info)
        dx, dz = self.danger_offset
        self.danger_pos = (px + dx, py, pz + dz)

        # Place lava at danger zone
        sim.env.execute_cmd(f"/setblock ~{dx} ~-1 ~{dz} lava")

        obs['task'] = {
            'name': 'avoid_then_return',
            'text': f'There is a lava zone ahead. Do NOT go there without a {self.required_item}. '
                    f'Find or craft a {self.required_item} first, then approach safely.',
        }
        self.oracle.register_contract(0)  # negative contract: avoid danger
        self.oracle.signal_should_write()
        info['pm_oracle'] = self.oracle.get_events()
        info['pm_task_family'] = 'negative_contract'
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.step_count += 1
        self.oracle.clear_step_events()

        # Check if agent acquired required item
        inv_items = get_inventory_items(info)
        if not self.has_required and self.required_item in inv_items:
            self.has_required = True
            self.oracle.signal_cancel(0, reason=3)  # supersede negative
            self.oracle.register_contract(1)  # new positive: "return now"
            self.oracle.signal_should_write()
            self.oracle.signal_fire(1)
            self.metrics.record_trigger(self.step_count, 1, True)

        # Check proximity to danger zone
        px, py, pz = get_player_pos(info)
        dx, dy, dz = self.danger_pos
        dist = ((px - dx)**2 + (pz - dz)**2) ** 0.5
        near = dist < 4.0

        if near:
            if not self.has_required and not self.first_visit_failed:
                self.first_visit_failed = True
                reward -= 5.0
                self.metrics.record_fire(self.step_count, 0, True)  # violated negative contract

            elif self.has_required and not self.returned_success:
                self.returned_success = True
                self.oracle.signal_completed(1)
                self.metrics.record_action(self.step_count, 1)
                reward += 20.0

        info['pm_oracle'] = self.oracle.get_events()
        info['pm_metrics'] = self.metrics.compute()
        return obs, reward, terminated, truncated, info
