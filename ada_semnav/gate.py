"""Recovery-Need gate with fixed and heuristic trigger modes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ada_semnav.env_runner import Position3D


@dataclass
class GateInputs:
    step_idx: int
    novelty: float
    revisit_rate: float
    collision_rate: float
    oscillation_rate: float
    no_semantic_progress: float
    visual_stagnation: float = 0.0


@dataclass
class GateDecision:
    should_trigger: bool
    score: float
    reason: str
    mode: str


def _distance(a: Position3D, b: Position3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _deduplicate_positions(positions: list[Position3D], min_dist: float = 0.05) -> list[Position3D]:
    """Remove consecutive near-duplicate positions (e.g. from in-place rotations)."""
    if not positions:
        return []
    result = [positions[0]]
    for p in positions[1:]:
        if _distance(result[-1], p) > min_dist:
            result.append(p)
    return result


def compute_novelty(history: list[Position3D], current: Position3D, window: int = 40) -> float:
    if not history:
        return 1.0
    deduped = _deduplicate_positions(history)
    # Drop trailing entry if it matches current (avoid self-comparison;
    # the last execute_action already appended current to history).
    if deduped and _distance(deduped[-1], current) < 0.10:
        deduped = deduped[:-1]
    recent = deduped[-max(1, int(window)) :]
    if not recent:
        return 1.0
    min_dist = min(_distance(current, p) for p in recent)
    # 0~1 normalization in indoor scale.
    return float(max(0.0, min(1.0, min_dist / 2.0)))


def compute_revisit_rate(
    history: list[Position3D],
    current: Position3D,
    window: int = 20,
    revisit_dist: float = 0.75,
) -> float:
    if not history:
        return 0.0
    deduped = _deduplicate_positions(history)
    recent = deduped[-max(1, int(window)) :]
    if not recent:
        return 0.0
    hit = sum(1 for p in recent if _distance(current, p) <= revisit_dist)
    return float(hit) / float(len(recent))


def compute_oscillation_rate(actions: list[str], window: int = 12) -> float:
    if len(actions) < 2:
        return 0.0
    recent = actions[-max(2, int(window)) :]
    flips = 0
    total = 0
    for a, b in zip(recent[:-1], recent[1:]):
        total += 1
        if (a == "turn_left" and b == "turn_right") or (a == "turn_right" and b == "turn_left"):
            flips += 1
    if total == 0:
        return 0.0
    return float(flips) / float(total)


def compute_no_semantic_progress(distance_history: list[float], window: int = 10, eps: float = 0.05) -> float:
    if len(distance_history) < 2:
        return 0.0
    recent = distance_history[-max(2, int(window)) :]
    improve = recent[0] - min(recent)
    return 1.0 if improve < eps else 0.0


class RecoveryNeedGate:
    """Gate implementing the staged paper logic: fixed trigger and heuristic trigger."""

    def __init__(
        self,
        *,
        mode: str = "heuristic",
        fixed_interval: int = 20,
        warmup_steps: int = 3,
        early_trigger_step: int = 0,
        early_trigger_min_active_rules: int = 0,
        cooldown_steps: int = 3,
        min_active_rules: int = 2,
        stall_trigger_steps: int = 0,
        stall_require_context: bool = True,
        novelty_threshold: float = 0.30,
        revisit_threshold: float = 0.55,
        collision_threshold: float = 0.35,
        oscillation_threshold: float = 0.40,
        stage_switch_step: int = 0,
        stage2_warmup_steps: int = -1,
        stage2_cooldown_steps: int = -1,
        stage2_min_active_rules: int = 0,
        stage2_novelty_threshold: float = -1.0,
        stage2_revisit_threshold: float = -1.0,
        stage2_collision_threshold: float = -1.0,
        stage2_oscillation_threshold: float = -1.0,
    ) -> None:
        self.mode = mode
        self.fixed_interval = max(1, int(fixed_interval))
        self.warmup_steps = max(0, int(warmup_steps))
        self.early_trigger_step = max(0, int(early_trigger_step))
        self.early_trigger_min_active_rules = max(0, int(early_trigger_min_active_rules))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.min_active_rules = max(1, int(min_active_rules))
        self.stall_trigger_steps = max(0, int(stall_trigger_steps))
        self.stall_require_context = bool(stall_require_context)
        self.novelty_threshold = float(novelty_threshold)
        self.revisit_threshold = float(revisit_threshold)
        self.collision_threshold = float(collision_threshold)
        self.oscillation_threshold = float(oscillation_threshold)
        self.stage_switch_step = max(0, int(stage_switch_step))
        self.stage2_warmup_steps = int(stage2_warmup_steps)
        self.stage2_cooldown_steps = int(stage2_cooldown_steps)
        self.stage2_min_active_rules = int(stage2_min_active_rules)
        self.stage2_novelty_threshold = float(stage2_novelty_threshold)
        self.stage2_revisit_threshold = float(stage2_revisit_threshold)
        self.stage2_collision_threshold = float(stage2_collision_threshold)
        self.stage2_oscillation_threshold = float(stage2_oscillation_threshold)
        self._last_trigger_step = -10**9
        self._stall_steps = 0

    def reset_episode(self) -> None:
        """Reset per-episode state (cooldown, stall counter)."""
        self._last_trigger_step = -10**9
        self._stall_steps = 0

    def _in_cooldown(self, step_idx: int, cooldown_steps: int) -> bool:
        return (step_idx - self._last_trigger_step) <= cooldown_steps

    def _maybe_trigger(self, step_idx: int, score: float, reason: str, cooldown_steps: int) -> GateDecision:
        if self._in_cooldown(step_idx, cooldown_steps):
            return GateDecision(False, score, "cooldown", self.mode)
        self._stall_steps = 0
        self._last_trigger_step = step_idx
        return GateDecision(True, score, reason, self.mode)

    def _stage_active(self, step_idx: int) -> bool:
        return self.stage_switch_step > 0 and step_idx >= self.stage_switch_step

    def _warmup_for_step(self, step_idx: int) -> int:
        if self._stage_active(step_idx) and self.stage2_warmup_steps >= 0:
            return max(0, self.stage2_warmup_steps)
        return self.warmup_steps

    def _cooldown_for_step(self, step_idx: int) -> int:
        if self._stage_active(step_idx) and self.stage2_cooldown_steps >= 0:
            return max(0, self.stage2_cooldown_steps)
        return self.cooldown_steps

    def _min_active_rules_for_step(self, step_idx: int) -> int:
        if self._stage_active(step_idx) and self.stage2_min_active_rules > 0:
            return max(1, self.stage2_min_active_rules)
        return self.min_active_rules

    def _thresholds_for_step(self, step_idx: int) -> tuple[float, float, float, float]:
        if not self._stage_active(step_idx):
            return (
                self.novelty_threshold,
                self.revisit_threshold,
                self.collision_threshold,
                self.oscillation_threshold,
            )
        novelty = self.stage2_novelty_threshold if self.stage2_novelty_threshold >= 0 else self.novelty_threshold
        revisit = self.stage2_revisit_threshold if self.stage2_revisit_threshold >= 0 else self.revisit_threshold
        collision = self.stage2_collision_threshold if self.stage2_collision_threshold >= 0 else self.collision_threshold
        oscillation = (
            self.stage2_oscillation_threshold if self.stage2_oscillation_threshold >= 0 else self.oscillation_threshold
        )
        return (novelty, revisit, collision, oscillation)

    def active_rule_count(self, gate_inputs: GateInputs) -> tuple[int, float, list[str]]:
        step_idx = int(gate_inputs.step_idx)
        novelty_th, revisit_th, collision_th, osc_th = self._thresholds_for_step(step_idx)
        c_novelty = gate_inputs.novelty < novelty_th
        c_revisit = gate_inputs.revisit_rate > revisit_th
        c_collision = gate_inputs.collision_rate > collision_th
        c_osc = gate_inputs.oscillation_rate > osc_th
        c_sem = gate_inputs.no_semantic_progress >= 0.5
        c_visual = gate_inputs.visual_stagnation >= 0.5
        active = [c_novelty, c_revisit, c_collision, c_osc, c_sem, c_visual]
        active_count = sum(1 for x in active if x)
        score = float(active_count) / 6.0
        reasons: list[str] = []
        if c_novelty:
            reasons.append("low_novelty")
        if c_revisit:
            reasons.append("high_revisit")
        if c_collision:
            reasons.append("high_collision")
        if c_osc:
            reasons.append("high_oscillation")
        if c_sem:
            reasons.append("no_semantic_progress")
        if c_visual:
            reasons.append("visual_stagnation")
        return active_count, score, reasons

    def observe(self, gate_inputs: GateInputs) -> GateDecision:
        step_idx = int(gate_inputs.step_idx)
        cooldown_steps = self._cooldown_for_step(step_idx)
        warmup_steps = self._warmup_for_step(step_idx)
        min_active_rules = self._min_active_rules_for_step(step_idx)
        if self.mode in {"off", "none"}:
            return GateDecision(False, 0.0, "disabled", self.mode)
        if self.mode == "fixed":
            if step_idx < warmup_steps:
                return GateDecision(False, 0.0, "warmup", self.mode)
            if step_idx % self.fixed_interval == 0:
                return self._maybe_trigger(step_idx, 1.0, "fixed_interval", cooldown_steps)
            return GateDecision(False, 0.0, "fixed_idle", self.mode)

        if gate_inputs.no_semantic_progress >= 0.5:
            self._stall_steps += 1
        else:
            self._stall_steps = 0

        # Heuristic gate: trigger when at least two risk conditions are active.
        active_count, score, reasons = self.active_rule_count(gate_inputs)
        c_novelty = "low_novelty" in reasons
        c_revisit = "high_revisit" in reasons
        c_collision = "high_collision" in reasons
        c_osc = "high_oscillation" in reasons

        if step_idx < warmup_steps:
            early_min_rules = self.early_trigger_min_active_rules or min_active_rules
            if self.early_trigger_step > 0 and step_idx >= self.early_trigger_step and active_count >= early_min_rules:
                return self._maybe_trigger(step_idx, score, "early_" + "+".join(reasons), cooldown_steps)
            return GateDecision(False, score, "warmup", self.mode)
        if self.stall_trigger_steps > 0 and self._stall_steps >= self.stall_trigger_steps:
            context_ok = c_novelty or c_revisit or c_collision or c_osc
            if (not self.stall_require_context) or context_ok:
                return self._maybe_trigger(step_idx, score, "stall_force", cooldown_steps)
        if active_count >= min_active_rules:
            return self._maybe_trigger(step_idx, score, "+".join(reasons), cooldown_steps)
        return GateDecision(False, score, "heuristic_idle", self.mode)
