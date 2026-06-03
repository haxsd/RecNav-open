#!/usr/bin/env python
"""Stage 3-5 transitional PixNav host runner.

This script keeps the current external planner path but routes local navigation
through the Stage 2 PixNavHost contract. It is intended for B0/B1 comparison:

- B0/B1: host-only vs monitor-only attribution (Stage 3)
- F0/F2/F2b smoke/dev comparison with lightweight recovery modes (Stage 4)
- G0/G1/G2 gate comparison with fixed minimal backend (Stage 5)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PIXNAV_DIR = REPO_ROOT / "pixnav"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIXNAV_DIR) not in sys.path:
    sys.path.insert(0, str(PIXNAV_DIR))

import habitat  # type: ignore
from habitat.config.read_write import read_write  # type: ignore
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height  # type: ignore

from ada_semnav import (
    BudgetCaps,
    BudgetManager,
    MetricsLogger,
    PixNavHost,
    Position3D,
    RecoverableSkeletonMemory,
    log_budget_snapshot_event,
)
from ada_semnav.gate import GateDecision, GateInputs, RecoveryNeedGate, compute_novelty, compute_revisit_rate
from ada_semnav.recovery import EdgeConstrainedRecovery
from config_utils import hm3d_config
from constants import HM3D_CONFIG_PATH
from cv_utils.detection_tools import initialize_dino_model
from cv_utils.segmentation_tools import initialize_sam_model
from gpt4v_planner import GPT4V_Planner


os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def append_metric_row(metric: dict[str, object], path: Path) -> None:
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open(mode="a", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(metric.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metric)


def load_completed_episode_rows(path: Path, eval_episodes: int) -> tuple[set[int], list[dict[str, object]]]:
    if not path.exists() or path.stat().st_size == 0:
        return set(), []

    completed: set[int] = set()
    rows: list[dict[str, object]] = []
    with path.open(mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                episode_idx = int(str(row.get("episode_idx", "")).strip())
            except ValueError:
                continue
            if 0 <= episode_idx < eval_episodes and episode_idx not in completed:
                completed.add(episode_idx)
                rows.append(dict(row))
    return completed, rows


def prune_telemetry_to_completed_episodes(telemetry_dir: Path, completed_episode_indices: set[int]) -> None:
    if not telemetry_dir.exists() or not completed_episode_indices:
        return

    for name in ("step_trace.jsonl", "episode_summary.jsonl", "events.jsonl", "memory_snapshot.jsonl"):
        path = telemetry_dir / name
        if not path.exists():
            continue
        kept: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    payload = json.loads(line)
                    episode_idx = payload.get("episode_idx")
                    keep_line = episode_idx is None or int(episode_idx) in completed_episode_indices
                except (json.JSONDecodeError, TypeError, ValueError):
                    keep_line = True
                if keep_line:
                    kept.append(line)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("".join(kept), encoding="utf-8")
        tmp_path.replace(path)


def adjust_topdown(metrics: dict[str, object]) -> np.ndarray:
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics["top_down_map"], 1024), cv2.COLOR_BGR2RGB)


_RECENT_IMAGE_CAP = 24


def record_observation(
    obs: dict[str, np.ndarray],
    habitat_env: habitat.Env,
    recent_images: list[np.ndarray],
    rgb_frames: list[np.ndarray] | None = None,
    topdown_frames: list[np.ndarray] | None = None,
) -> None:
    recent_images.append(obs["rgb"])
    if len(recent_images) > _RECENT_IMAGE_CAP:
        recent_images.pop(0)
    if rgb_frames is not None:
        rgb_frames.append(obs["rgb"])
    if topdown_frames is not None:
        topdown_frames.append(adjust_topdown(habitat_env.get_metrics()))


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_planner_trace(path: Path) -> dict[int, list[dict[str, object]]]:
    """Load planner trace grouped by episode_idx for per-episode replay."""
    grouped: dict[int, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                ep = int(rec["episode_idx"])
                grouped.setdefault(ep, []).append(rec)
    return grouped


VALID_ACTION_IDS = {0, 1, 2, 3, 4, 5}
ACTION_ID_TO_NAME = {
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
}
ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}


def inverse_action_sequence(action: int) -> list[int]:
    """Approximate inverse primitives for minimal deterministic backtrack."""
    if action == 1:  # move_forward -> turn-around + move_forward + turn-back
        return [2, 2, 1, 2, 2]
    if action == 2:
        return [3]
    if action == 3:
        return [2]
    if action == 4:
        return [5]
    if action == 5:
        return [4]
    return []


def build_minimal_backtrack_plan(action_history: list[int], step_budget: int) -> list[int]:
    if step_budget <= 0:
        return []
    plan: list[int] = []
    for action in reversed(action_history):
        if action == 0:
            continue
        plan.extend(inverse_action_sequence(int(action)))
        if len(plan) >= step_budget:
            break
    return plan[:step_budget]


def probe_macro_candidates(
    habitat_env: habitat.Env,
    fan_offsets: list[int],
    memory: RecoverableSkeletonMemory | None = None,
    scene_id: str = "",
    probe_steps: int = 1,
) -> list[dict]:
    """Probe DTG for each fan-out direction via sim state save/restore.

    Each offset is N turns from current heading (positive=right, negative=left,
    each unit = one turn action = 30 deg).  Probes ``probe_steps`` forward
    steps so the evaluation horizon matches the execution horizon.

    Uses the same geodesic_distance(pos, goals, episode) call that the env
    metric uses, so the returned DTG is directly comparable to dtg_before.

    When *memory* is provided, each result includes ``visit_count`` – the
    cumulative visit count of the memory cell at the probe position.  This
    allows memory-informed ranking to penalise frequently-visited areas.

    The probe does NOT touch env-level bookkeeping.

    Returns list of {offset, dtg, collided, visit_count} sorted by input order.
    """
    sim = habitat_env.sim
    episode = habitat_env.current_episode
    saved = sim.get_agent_state()
    saved_pos = np.array(saved.position, dtype=np.float32)
    saved_rot = saved.rotation

    goal_positions = [
        np.array(g.position, dtype=np.float32)
        for g in episode.goals
    ]

    _TURN_R = 3
    _TURN_L = 2
    _FWD = 1

    results: list[dict] = []

    for target_offset in fan_offsets:
        sim.set_agent_state(saved_pos, saved_rot)

        turn_action = _TURN_R if target_offset > 0 else _TURN_L
        for _ in range(abs(target_offset)):
            sim.step(turn_action)

        collided = False
        for _ps in range(max(1, probe_steps)):
            sim.step(_FWD)
            if sim.previous_step_collided:
                collided = True
                break

        new_pos = np.array(
            sim.get_agent_state().position, dtype=np.float32
        )
        dtg = float(sim.geodesic_distance(new_pos, goal_positions, episode))

        visit_count = 0
        mem_signals: dict[str, float] = {}
        if memory is not None and scene_id:
            probe_pos = Position3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
            mem_signals = memory.node_signals_at(scene_id, probe_pos)
            visit_count = int(mem_signals.get("visit_count", 0))

        results.append({
            "offset": target_offset, "dtg": dtg, "collided": collided,
            "visit_count": visit_count,
            "degree": mem_signals.get("degree", 0.0),
            "unexplored_edges": mem_signals.get("unexplored_edges", 0.0),
            "blocked_edges": mem_signals.get("blocked_edges", 0.0),
        })

    sim.set_agent_state(saved_pos, saved_rot)
    return results


def choose_heuristic_edge_action(action_history: list[int], window: int) -> int:
    recent = action_history[-max(1, int(window)) :]
    counts = {action: recent.count(action) for action in (1, 2, 3)}
    return min((1, 2, 3), key=lambda action: (counts[action], action))


def choose_naive_llm_action(goal_rotate: int) -> int:
    """Map planner heading suggestion to a single unconstrained primitive."""
    rotate = int(goal_rotate) % 12
    if rotate == 0:
        return 1  # move_forward
    if rotate <= 6:
        return 3  # turn_right
    return 2  # turn_left


def choose_memory_constrained_action(
    *,
    memory: RecoverableSkeletonMemory | None,
    current_node: str | None,
) -> int | None:
    if memory is None or current_node is None or current_node not in memory.nodes:
        return None
    edges = memory.query_edges(
        current_node=current_node,
        actions=["move_forward", "turn_left", "turn_right"],
        create_missing=False,
    )
    if not edges:
        return None
    best = min(
        edges,
        key=lambda edge: (
            1 if edge.blocked else 0,
            1 if edge.is_low_yield else 0,
            edge.fail_count,
            edge.visit_count,
            -edge.success_count,
            edge.id,
        ),
    )
    for action_id, action_name in ACTION_ID_TO_NAME.items():
        if best.action == action_name:
            return action_id
    return None


def composite_score(candidate: dict, dtg_before: float) -> float:
    """Compute composite ranking score for a macro-action probe result.

    Lower score = better candidate.  Components:
      1. DTG normalised by dtg_before (core: closer to goal is better)
      2. Visit penalty (mild: avoid revisiting same cell in deadlock)
      3. Collision penalty (avoid directions that immediately collide)

    NOTE: Exploration signals (unexplored_edges, branch degree, blocked_edges)
    removed after ablation showed they override DTG at recovery's ~1m
    displacement scale, causing 22% of recoveries to pick worse directions
    (composite RSR=75% vs pure-DTG RSR=97%, 25/25 episodes worse, 0 better).
    """
    _W_DTG = 1.0
    _W_VISIT = 0.15
    _W_UNEXPLORED = 0.0
    _W_BRANCH = 0.0
    _W_BLOCKED = 0.0
    _W_COLLISION = 0.5

    dtg_norm = candidate["dtg"] / max(0.1, dtg_before)
    visit = min(float(candidate.get("visit_count", 0)), 8.0) / 8.0
    unexplored = min(float(candidate.get("unexplored_edges", 0)), 4.0) / 4.0
    branch = min(float(candidate.get("degree", 0)), 6.0) / 6.0
    blocked = min(float(candidate.get("blocked_edges", 0)), 4.0) / 4.0
    collided = float(candidate.get("collided", False))

    return (
        _W_DTG * dtg_norm
        + _W_VISIT * visit
        + _W_UNEXPLORED * unexplored
        + _W_BRANCH * branch
        + _W_BLOCKED * blocked
        + _W_COLLISION * collided
    )


def proxy_score(candidate: dict) -> float:
    """Non-oracle ranking using only memory and collision signals. Lower = better.

    Replaces DTG-based composite_score when simulator progress feedback
    is unavailable.  Rewards novelty (unexplored edges, high branch degree)
    and penalises revisiting and collisions.
    """
    _W_VISIT = 0.40
    _W_COLLISION = 0.80
    _W_UNEXPLORED = -0.30
    _W_BRANCH = -0.15
    _W_BLOCKED = 0.25

    visit = min(float(candidate.get("visit_count", 0)), 8.0) / 8.0
    collided = float(candidate.get("collided", False))
    unexplored = min(float(candidate.get("unexplored_edges", 0)), 4.0) / 4.0
    branch = min(float(candidate.get("degree", 0)), 6.0) / 6.0
    blocked = min(float(candidate.get("blocked_edges", 0)), 4.0) / 4.0

    return (
        _W_VISIT * visit
        + _W_COLLISION * collided
        + _W_UNEXPLORED * unexplored
        + _W_BRANCH * branch
        + _W_BLOCKED * blocked
    )


def compute_collision_ratio(collision_history: list[int], window: int) -> float:
    if not collision_history:
        return 0.0
    recent = collision_history[-max(1, int(window)) :]
    return float(sum(int(x) for x in recent)) / float(len(recent))


def compute_turn_oscillation_ratio(action_history: list[int], window: int) -> float:
    recent = action_history[-max(1, int(window)) :]
    if len(recent) < 2:
        return 0.0
    oscillations = 0
    for prev_action, curr_action in zip(recent[:-1], recent[1:]):
        if (prev_action == 2 and curr_action == 3) or (prev_action == 3 and curr_action == 2):
            oscillations += 1
        if (prev_action == 4 and curr_action == 5) or (prev_action == 5 and curr_action == 4):
            oscillations += 1
    return float(oscillations) / float(len(recent) - 1)


def compute_visual_stagnation(
    recent_images: list[np.ndarray],
    window: int = 8,
    threshold: float = 0.95,
) -> bool:
    """Detect visual stagnation by comparing RGB observation similarity.

    Downsamples each frame to 8×8 grayscale and computes mean cosine
    similarity between the current frame and the recent window.  Returns
    True when the agent's visual observation has not meaningfully changed.
    Cost: pure numpy, ~0.1 ms per call.
    """
    if len(recent_images) < max(2, window):
        return False

    def _hash(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        v = cv2.resize(gray, (8, 8)).flatten().astype(np.float32)
        v -= v.mean()
        return v

    current = _hash(recent_images[-1])
    current_norm = np.linalg.norm(current)
    if current_norm < 1e-8:
        return False
    recent = [_hash(f) for f in recent_images[-window - 1 : -1]]
    if not recent:
        return False
    sims = [
        float(np.dot(current, r) / (current_norm * np.linalg.norm(r) + 1e-8))
        for r in recent
    ]
    return float(np.mean(sims)) > threshold


def is_failure_opportunity(
    dtg_history: list[float],
    *,
    window: int,
    progress_epsilon: float,
) -> bool:
    """Pose-free failure-opportunity definition used by Stage 5 trigger audit.

    A failure opportunity is counted when DTG improvement over the latest N
    executed primitives does not exceed epsilon.
    """
    effective_window = max(1, int(window))
    if len(dtg_history) < effective_window + 1:
        return False
    dtg_before = float(dtg_history[-(effective_window + 1)])
    dtg_after = float(dtg_history[-1])
    return (dtg_before - dtg_after) <= float(progress_epsilon)


def get_agent_position(habitat_env: habitat.Env) -> Position3D | None:
    try:
        pos = habitat_env.sim.get_agent_state().position
    except Exception:  # noqa: BLE001
        return None
    if pos is None or len(pos) < 3:
        return None
    return Position3D(float(pos[0]), float(pos[1]), float(pos[2]))


def _rebuild_memory_indexes(memory: RecoverableSkeletonMemory) -> None:
    memory._node_by_cell = {}
    for node in memory.nodes.values():
        key = (node.scene_id, node.cell_x, node.cell_z)
        memory._node_by_cell.setdefault(key, []).append(node.id)
    memory._edge_by_action = {(edge.from_node, edge.action): eid for eid, edge in memory.edges.items()}


def apply_memory_mode(
    *,
    memory: RecoverableSkeletonMemory,
    memory_mode: str,
    recent_nodes: list[str],
    fifo_history_size: int,
) -> None:
    if memory_mode == "skeleton":
        return
    current = memory.last_node_id
    if current is None or current not in memory.nodes:
        return
    if memory_mode == "none":
        memory.nodes = {current: memory.nodes[current]}
        memory.edges = {}
        recent_nodes[:] = [current]
        _rebuild_memory_indexes(memory)
        return

    keep_recent = max(4, int(fifo_history_size))
    recent_nodes[:] = recent_nodes[-keep_recent:]
    keep_nodes = {nid for nid in recent_nodes if nid in memory.nodes}
    keep_nodes.add(current)
    memory.nodes = {nid: memory.nodes[nid] for nid in keep_nodes if nid in memory.nodes}
    memory.edges = {
        eid: edge
        for eid, edge in memory.edges.items()
        if edge.from_node in keep_nodes and edge.to_node in keep_nodes
    }
    _rebuild_memory_indexes(memory)


def count_branching_nodes(memory: RecoverableSkeletonMemory) -> int:
    return sum(1 for nid in memory.nodes if memory.node_degree(nid) >= 3)


def count_recovery_useful_edges(memory: RecoverableSkeletonMemory) -> int:
    return sum(
        1
        for edge in memory.edges.values()
        if edge.success_count > 0 or edge.fail_count > 0 or edge.is_low_yield or edge.blocked
    )


def update_memory_state(
    habitat_env: habitat.Env,
    *,
    memory: RecoverableSkeletonMemory | None,
    recent_nodes: list[str],
    memory_stats: dict[str, float],
    memory_mode: str,
    fifo_history_size: int,
    token_budget: int,
    memory_compression_strategy: str,
    scene_id: str,
    action: int,
    collision: bool,
    step_idx: int,
    source: str,
) -> None:
    if memory is None:
        return
    position = get_agent_position(habitat_env)
    if position is None:
        return

    action_name = ACTION_ID_TO_NAME.get(int(action))
    node_id = memory.observe(
        scene_id=scene_id,
        position=position,
        step_idx=int(step_idx),
        action=action_name,
        collision=bool(collision),
        source=str(source),
    )
    recent_nodes.append(node_id)
    apply_memory_mode(
        memory=memory,
        memory_mode=str(memory_mode),
        recent_nodes=recent_nodes,
        fifo_history_size=int(fifo_history_size),
    )

    token_cap = max(1, int(token_budget))
    token_estimate = float(memory.estimate_tokens())
    if token_estimate > float(token_cap):
        removed = memory.compress(strategy=str(memory_compression_strategy))
        if removed:
            memory_stats["memory_compress_events"] += 1.0
        token_estimate = float(memory.estimate_tokens())

    node_count = float(len(memory.nodes))
    edge_count = float(len(memory.edges))
    branch_count = float(count_branching_nodes(memory))
    useful_edge_count = float(count_recovery_useful_edges(memory))

    memory_stats["memory_entry_count"] += 1.0
    memory_stats["memory_samples"] += 1.0
    memory_stats["retained_nodes_sum"] += node_count
    memory_stats["retained_branching_points_sum"] += branch_count
    memory_stats["recovery_useful_edges_sum"] += useful_edge_count
    memory_stats["memory_node_peak"] = max(memory_stats["memory_node_peak"], node_count)
    memory_stats["memory_edge_peak"] = max(memory_stats["memory_edge_peak"], edge_count)
    memory_stats["max_retained_memory_tokens"] = max(memory_stats["max_retained_memory_tokens"], token_estimate)


def log_memory_snapshot(
    logger: MetricsLogger | None,
    *,
    episode_idx: int,
    episode_id: str,
    scene_id: str,
    step_idx: int,
    source: str,
    memory: RecoverableSkeletonMemory | None,
    memory_stats: dict[str, float] | None,
    memory_mode: str,
) -> None:
    if logger is None or memory is None:
        return
    logger.log_memory(
        {
            "episode": int(episode_idx),
            "episode_id": str(episode_id),
            "scene_id": str(scene_id),
            "step": int(step_idx),
            "source": str(source),
            "memory_mode": str(memory_mode),
            "last_node_id": str(memory.last_node_id or ""),
            "node_count": int(len(memory.nodes)),
            "edge_count": int(len(memory.edges)),
            "branching_node_count": int(count_branching_nodes(memory)),
            "recovery_useful_edge_count": int(count_recovery_useful_edges(memory)),
            "token_estimate": float(memory.estimate_tokens()),
            "memory_entry_count": float((memory_stats or {}).get("memory_entry_count", 0.0)),
            "memory_compress_events": float((memory_stats or {}).get("memory_compress_events", 0.0)),
        }
    )


def normalize_optional_int_cap(value: int) -> int | None:
    normalized = int(value)
    return normalized if normalized >= 0 else None


def log_budget_snapshot(
    logger: MetricsLogger | None,
    *,
    budget_manager: BudgetManager | None,
    episode_idx: int,
    episode_id: str,
    scene_id: str,
    step_idx: int,
    reason: str,
) -> None:
    if logger is None or budget_manager is None:
        return
    log_budget_snapshot_event(
        logger,
        budget_snapshot=budget_manager.snapshot(),
        episode=int(episode_idx),
        step=int(step_idx),
        episode_id=str(episode_id),
        scene_id=str(scene_id),
        reason=str(reason),
    )


def execute_action(
    habitat_env: habitat.Env,
    obs: dict[str, np.ndarray],
    action: int,
    *,
    recent_images: list[np.ndarray],
    rgb_frames: list[np.ndarray] | None,
    topdown_frames: list[np.ndarray] | None,
    action_history: list[int],
    collision_history: list[int],
    dtg_history: list[float],
    step_idx: int,
    memory: RecoverableSkeletonMemory | None = None,
    recent_nodes: list[str] | None = None,
    memory_stats: dict[str, float] | None = None,
    memory_mode: str = "disabled",
    fifo_history_size: int = 80,
    token_budget: int = 256,
    memory_compression_strategy: str = "recoverability",
    scene_id: str = "",
    source: str = "base",
    budget_manager: BudgetManager | None = None,
    logger: MetricsLogger | None = None,
    episode_idx: int = 0,
    episode_id: str = "",
    position_history: list[Position3D] | None = None,
) -> tuple[dict[str, np.ndarray], int, bool]:
    try:
        next_obs = habitat_env.step(int(action))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] env.step failed at action={action}: {exc}")
        return obs, step_idx, False
    record_observation(next_obs, habitat_env, recent_images, rgb_frames, topdown_frames)
    action_history.append(int(action))
    collided = bool(habitat_env.sim.previous_step_collided)
    collision_history.append(int(collided))
    dtg_history.append(float(habitat_env.get_metrics().get("distance_to_goal", 0.0)))
    if position_history is not None:
        _pos = get_agent_position(habitat_env)
        if _pos is not None:
            position_history.append(_pos)
    if memory is not None and recent_nodes is not None and memory_stats is not None:
        update_memory_state(
            habitat_env,
            memory=memory,
            recent_nodes=recent_nodes,
            memory_stats=memory_stats,
            memory_mode=str(memory_mode),
            fifo_history_size=int(fifo_history_size),
            token_budget=int(token_budget),
            memory_compression_strategy=str(memory_compression_strategy),
            scene_id=str(scene_id),
            action=int(action),
            collision=collided,
            step_idx=int(step_idx + 1),
            source=str(source),
        )
        log_memory_snapshot(
            logger,
            episode_idx=int(episode_idx),
            episode_id=str(episode_id),
            scene_id=str(scene_id),
            step_idx=int(step_idx + 1),
            source=str(source),
            memory=memory,
            memory_stats=memory_stats,
            memory_mode=str(memory_mode),
        )
        if budget_manager is not None:
            budget_manager.note_memory_tokens(float(memory.estimate_tokens()))
    return next_obs, step_idx + 1, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PixNav host-only or monitor-only evaluation.")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=0,
                        help="Override Habitat max_episode_steps (0=use YAML default of 500)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--monitor-only", action="store_true", help="Export telemetry without altering host actions.")
    parser.add_argument(
        "--export-telemetry",
        action="store_true",
        help="Write telemetry logs without forcing monitor-only host behavior.",
    )
    parser.add_argument("--save_rgb_video", action="store_true")
    parser.add_argument("--save_topdown_video", action="store_true")
    parser.add_argument("--save_planner_monitor", action="store_true")
    parser.add_argument("--planner_image_scale", type=float, default=0.5)
    parser.add_argument("--planner_trace_path", type=str, default=None)
    parser.add_argument("--replay_planner_trace", action="store_true")
    parser.add_argument(
        "--planner_replan_policy",
        type=str,
        default="always",
        choices=["always", "stride"],
        help="Planner refresh policy on replan events. 'always' keeps legacy behavior.",
    )
    parser.add_argument(
        "--planner_replan_stride",
        type=int,
        default=3,
        help="When planner_replan_policy=stride, refresh vision planner every N replan events.",
    )
    parser.add_argument(
        "--recovery_mode",
        type=str,
        default="host_only",
        choices=["host_only", "minimal_backtrack", "heuristic_edge", "naive_llm", "constrained_edge", "macro_action"],
    )
    parser.add_argument("--recovery_backtrack_steps", type=int, default=6)
    parser.add_argument("--recovery_macro_max_fan", type=int, default=13,
                        help="Max fan-out candidates for macro_action recovery (offset 0, +1, -1, ..., +6, -6 = full 360 deg)")
    parser.add_argument("--recovery_macro_forward_steps", type=int, default=3,
                        help="Forward steps per macro candidate (adaptive: DTG-guided stop)")
    parser.add_argument("--recovery_macro_probe_steps", type=int, default=1,
                        help="Probe steps for direction selection (-1=use forward_steps for P3 alignment, default 1=baseline)")
    parser.add_argument("--recovery_macro_abstain", type=int, default=1,
                        help="If 1, abstain (skip execution) when no probed direction improves DTG")
    parser.add_argument("--recovery_macro_ranking", type=str, default="dtg",
                        choices=["dtg", "memory_dtg", "composite", "random", "frontier_nearest"],
                        help="Probe ranking: dtg=pure DTG, memory_dtg=DTG+visit penalty, composite=multi-factor, random=random, frontier_nearest=least-visited+exploration")
    parser.add_argument("--recovery_heuristic_window", type=int, default=24)
    parser.add_argument("--recovery_progress_threshold", type=float, default=0.05)
    parser.add_argument("--recovery_max_retries", type=int, default=2,
                        help="Max retry attempts for macro_action recovery (closed-loop)")
    parser.add_argument("--recovery_closed_loop", type=int, default=1,
                        help="If 1, update memory after each recovery attempt and retry on failure")
    parser.add_argument("--recovery_continuation_steps", type=int, default=4,
                        help="Forward steps after edge selection to produce spatial displacement")
    parser.add_argument("--recovery_dtg_guided_continuation", type=int, default=1,
                        help="Stop continuation early if DTG increases (1=enabled)")
    parser.add_argument(
        "--recovery_edge_query_mode",
        type=str,
        default="current_node",
        choices=["current_node", "anchor_frontier", "skeleton_frontier", "query_aware_subgraph", "query_aware_edge_pool"],
    )
    parser.add_argument(
        "--progress_signal",
        type=str,
        default="oracle",
        choices=["oracle", "proxy", "none"],
        help="Progress signal for recovery: oracle=simulator DTG (default), proxy=memory-visual signals only, none=random ranking",
    )
    parser.add_argument(
        "--gate_mode",
        type=str,
        default="legacy_stop_signal",
        choices=["legacy_stop_signal", "fixed_interval", "heuristic", "pose_free"],
    )
    parser.add_argument("--gate_interval_events", type=int, default=2)
    parser.add_argument("--gate_failure_window", type=int, default=8)
    parser.add_argument("--gate_progress_epsilon", type=float, default=0.05)
    parser.add_argument("--gate_collision_threshold", type=float, default=0.30)
    parser.add_argument("--gate_oscillation_threshold", type=float, default=0.30)
    parser.add_argument("--gate_score_threshold", type=float, default=0.75)
    parser.add_argument("--gate_cooldown_events", type=int, default=1)
    parser.add_argument("--gate_min_active_rules", type=int, default=2,
                        help="In pose_free mode, minimum active signal count to trigger recovery.")
    parser.add_argument("--gate_novelty_threshold", type=float, default=0.30,
                        help="Novelty below this triggers low_novelty signal (0~1 scale).")
    parser.add_argument("--gate_novelty_window", type=int, default=20,
                        help="Position history window for novelty computation.")
    parser.add_argument("--gate_revisit_threshold", type=float, default=0.40,
                        help="Revisit ratio (0~1) above this triggers revisit signal.")
    parser.add_argument("--gate_visual_stagnation_window", type=int, default=8,
                        help="RGB frame window for visual stagnation detection.")
    parser.add_argument("--gate_visual_stagnation_threshold", type=float, default=0.95,
                        help="Cosine similarity above this triggers visual stagnation.")
    parser.add_argument(
        "--gate_failure_alone_triggers",
        action="store_true",
        help="In pose_free mode, allow failure_opportunity alone to trigger recovery (bypass score threshold).",
    )
    parser.add_argument("--gate_goal_hysteresis_window", type=int, default=2,
                        help="Suppress recovery if goal_flag was True in any of the last N planner calls (0=disabled).")
    parser.add_argument("--gate_late_budget_reserve", type=int, default=20,
                        help="Suppress recovery if remaining episode steps < this value (0=disabled).")
    parser.add_argument(
        "--disable_memory_constrained_edge",
        action="store_true",
        help="Disable memory-constrained edge selection and force heuristic action selection in heuristic_edge mode.",
    )
    parser.add_argument(
        "--memory_mode",
        type=str,
        default="disabled",
        choices=["disabled", "none", "fifo", "skeleton"],
    )
    parser.add_argument("--memory_cell_size", type=float, default=0.75)
    parser.add_argument("--memory_max_nodes", type=int, default=50)
    parser.add_argument("--fifo_history_size", type=int, default=80)
    parser.add_argument("--token_budget", type=int, default=256)
    parser.add_argument(
        "--call_budget",
        type=int,
        default=-1,
        help="Per-episode recovery-call cap for the plugin slow process. Negative disables the cap.",
    )
    parser.add_argument(
        "--memory_compression_strategy",
        type=str,
        default="recoverability",
        choices=["recoverability", "normal"],
    )
    parser.add_argument("--gate_dtg_protect_margin", type=float, default=0.0,
                        help="Suppress recovery if current DTG <= best-so-far DTG + margin (0=disabled). P1 experiment showed margin=0.5 is harmful.")
    parser.add_argument("--post_recovery_replan", type=str, default="skip",
                        choices=["skip", "full"],
                        help="Post-recovery replan mode: skip=current view only (default), full=330-deg rescan (P2 experiment showed full is harmful).")
    parser.add_argument("--proximity_stop_distance", type=float, default=0.0,
                        help="Force STOP when DTG < this value (0=disabled). Eliminates GroundingDINO false-negative near-misses.")
    parser.add_argument("--episode_seed", type=int, default=-1,
                        help="Seed for Habitat episode sampling (-1=use --seed). Separate from --seed to fix episode set across variants.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep existing CSV rows and skip completed episode_idx values after env.reset().")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    completed_episode_indices: set[int] = set()
    existing_episode_rows: list[dict[str, object]] = []
    if args.resume:
        completed_episode_indices, existing_episode_rows = load_completed_episode_rows(csv_path, int(args.eval_episodes))
        if completed_episode_indices:
            print(
                f"[resume] loaded {len(completed_episode_indices)} completed episodes "
                f"from {csv_path}; remaining={max(0, int(args.eval_episodes) - len(completed_episode_indices))}"
            )
    elif csv_path.exists():
        csv_path.unlink()
    planner_trace_path = Path(args.planner_trace_path) if args.planner_trace_path else None
    if planner_trace_path is not None and not args.replay_planner_trace and planner_trace_path.exists() and not args.resume:
        planner_trace_path.unlink()
    replay_trace_by_ep = load_planner_trace(planner_trace_path) if args.replay_planner_trace and planner_trace_path else None
    if replay_trace_by_ep is not None:
        planner_trace_path = None  # prevent writing to the source trace file
    replay_ep_seq: list[dict[str, object]] = []
    replay_ep_idx = 0
    planner_replan_policy = str(args.planner_replan_policy).strip().lower()
    planner_replan_stride = max(1, int(args.planner_replan_stride))

    habitat_config = hm3d_config(stage=args.split, episodes=args.eval_episodes,
                                  max_episode_steps=int(args.max_episode_steps))
    _episode_seed = int(args.episode_seed) if int(args.episode_seed) >= 0 else args.seed
    if _episode_seed is not None:
        with read_write(habitat_config):
            habitat_config.habitat.seed = _episode_seed
            habitat_config.habitat.environment.iterator_options.shuffle = False

    habitat_env = habitat.Env(habitat_config)
    detection_model = None
    segmentation_model = None
    try:
        detection_model = initialize_dino_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] GroundingDINO init failed, fallback planner will be used: {exc}")
    try:
        segmentation_model = initialize_sam_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] SAM init failed, fallback planner will be used: {exc}")

    nav_planner = GPT4V_Planner(
        detection_model,
        segmentation_model,
        save_monitor_image=args.save_planner_monitor,
        image_scale=args.planner_image_scale,
    )
    host = PixNavHost(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        enable_debug_images=False,
    )
    telemetry_enabled = bool(args.monitor_only or args.export_telemetry)
    if telemetry_enabled and args.resume:
        prune_telemetry_to_completed_episodes(output_dir / "telemetry", completed_episode_indices)
    logger = MetricsLogger(output_dir / "telemetry", append=args.resume) if telemetry_enabled else None
    episode_rows_written: list[dict[str, object]] = list(existing_episode_rows)
    seen_scenes: set[str] = {str(row.get("scene_id", "")) for row in existing_episode_rows if row.get("scene_id")}
    effective_call_budget = normalize_optional_int_cap(int(args.call_budget))
    budget_manager = BudgetManager(
        BudgetCaps(
            call_budget=effective_call_budget,
        )
    )
    recovery_engine = EdgeConstrainedRecovery(
        backend="heuristic",
        candidate_query_mode=str(args.recovery_edge_query_mode),
    ) if args.recovery_mode == "constrained_edge" else None

    # --- Gate instantiation (M4: refactored to use RecoveryNeedGate) ---
    recovery_gate: RecoveryNeedGate | None = None
    if args.recovery_mode != "host_only" and args.gate_mode != "legacy_stop_signal":
        _gate_cooldown = max(0, int(args.gate_cooldown_events) - 1)
        if args.gate_mode == "fixed_interval":
            recovery_gate = RecoveryNeedGate(
                mode="fixed",
                fixed_interval=max(1, int(args.gate_interval_events)),
                cooldown_steps=_gate_cooldown,
            )
        elif args.gate_mode == "heuristic":
            recovery_gate = RecoveryNeedGate(
                mode="heuristic",
                min_active_rules=1,
                cooldown_steps=_gate_cooldown,
                novelty_threshold=-1.0,
                revisit_threshold=999.0,
                collision_threshold=float(args.gate_collision_threshold),
                oscillation_threshold=float(args.gate_oscillation_threshold),
            )
        elif args.gate_mode == "pose_free":
            recovery_gate = RecoveryNeedGate(
                mode="heuristic",
                min_active_rules=int(args.gate_min_active_rules),
                cooldown_steps=_gate_cooldown,
                novelty_threshold=float(args.gate_novelty_threshold),
                revisit_threshold=float(args.gate_revisit_threshold),
                collision_threshold=float(args.gate_collision_threshold),
                oscillation_threshold=float(args.gate_oscillation_threshold),
            )

    try:
        for episode_idx in range(args.eval_episodes):
            budget_manager.reset_episode(episode_idx=int(episode_idx))
            if recovery_gate is not None:
                recovery_gate.reset_episode()
            obs = habitat_env.reset()
            episode = habitat_env.current_episode
            episode_id = str(episode.episode_id)
            scene_id = str(episode.scene_id)
            if episode_idx in completed_episode_indices:
                print(f"[resume] skip completed episode_idx={episode_idx} episode_id={episode_id}")
                continue
            episode_dir = output_dir / f"trajectory_{episode_idx}"
            if args.save_rgb_video or args.save_topdown_video:
                episode_dir.mkdir(parents=True, exist_ok=True)

            heading_offset = 0
            _replay_active = True
            if replay_trace_by_ep is not None:
                replay_ep_seq = list(replay_trace_by_ep.get(int(episode_idx), []))
                replay_ep_idx = 0
            nav_planner.reset(episode.object_category)
            recent_images = [obs["rgb"]]
            rgb_frames = [obs["rgb"]] if args.save_rgb_video else None
            topdown_frames = [adjust_topdown(habitat_env.get_metrics())] if args.save_topdown_video else None
            step_idx = 0
            action_history: list[int] = []
            collision_history: list[int] = []
            position_history: list[Position3D] = []
            _init_pos = get_agent_position(habitat_env)
            if _init_pos is not None:
                position_history.append(_init_pos)
            dtg_history: list[float] = [float(habitat_env.get_metrics().get("distance_to_goal", 0.0))]
            recovery_stats = {
                "recovery_attempts": 0,
                "recovery_success_count": 0,
                "recovery_failure_count": 0,
                "selection_illegal_count": 0,
                "execution_broken_count": 0,
                "closed_loop_fail_count": 0,
            }
            budget_stats = {
                "call_budget_skip_count": 0,
            }
            gate_stats = {
                "replan_event_count": 0,
                "trigger_count": 0,
                "true_trigger_count": 0,
                "false_trigger_count": 0,
                "failure_opportunity_count": 0,
                "trigger_overlap_count": 0,
            }
            cached_goal_rotate: int | None = None
            cached_goal_flag = False
            _goal_flag_history: list[bool] = []
            memory_stats: dict[str, float] = {
                "memory_entry_count": 0.0,
                "memory_samples": 0.0,
                "retained_nodes_sum": 0.0,
                "retained_branching_points_sum": 0.0,
                "recovery_useful_edges_sum": 0.0,
                "memory_node_peak": 0.0,
                "memory_edge_peak": 0.0,
                "max_retained_memory_tokens": 0.0,
                "memory_compress_events": 0.0,
            }
            memory_mode = str(args.memory_mode)
            memory: RecoverableSkeletonMemory | None = None
            recent_nodes: list[str] = []
            if memory_mode != "disabled":
                memory = RecoverableSkeletonMemory(
                    cell_size=float(args.memory_cell_size),
                    max_nodes=int(args.memory_max_nodes),
                )
                update_memory_state(
                    habitat_env,
                    memory=memory,
                    recent_nodes=recent_nodes,
                    memory_stats=memory_stats,
                    memory_mode=memory_mode,
                    fifo_history_size=int(args.fifo_history_size),
                    token_budget=int(args.token_budget),
                    memory_compression_strategy=str(args.memory_compression_strategy),
                    scene_id=scene_id,
                    action=0,
                    collision=False,
                    step_idx=int(step_idx),
                    source="base",
                )
                log_memory_snapshot(
                    logger,
                    episode_idx=int(episode_idx),
                    episode_id=episode_id,
                    scene_id=scene_id,
                    step_idx=int(step_idx),
                    source="base_init",
                    memory=memory,
                    memory_stats=memory_stats,
                    memory_mode=memory_mode,
                )
                budget_manager.note_memory_tokens(float(memory.estimate_tokens()))
            else:
                budget_manager.note_memory_tokens(0.0)
            log_budget_snapshot(
                logger,
                budget_manager=budget_manager,
                episode_idx=int(episode_idx),
                episode_id=episode_id,
                scene_id=scene_id,
                step_idx=int(step_idx),
                reason="episode_start",
            )
            for _ in range(11):
                obs, step_idx, ok = execute_action(
                    habitat_env,
                    obs,
                    3,
                    recent_images=recent_images,
                    rgb_frames=rgb_frames,
                    topdown_frames=topdown_frames,
                    action_history=action_history,
                    collision_history=collision_history,
                    dtg_history=dtg_history,
                    step_idx=step_idx,
                    memory=memory,
                    recent_nodes=recent_nodes,
                    memory_stats=memory_stats,
                    memory_mode=memory_mode,
                    fifo_history_size=int(args.fifo_history_size),
                    token_budget=int(args.token_budget),
                    memory_compression_strategy=str(args.memory_compression_strategy),
                    scene_id=scene_id,
                    budget_manager=budget_manager,
                    logger=logger,
                    episode_idx=int(episode_idx),
                    episode_id=episode_id,
                    position_history=position_history,
                )
                if not ok:
                    break
            if _replay_active and replay_trace_by_ep is not None and replay_ep_idx < len(replay_ep_seq):
                replay_row = replay_ep_seq[replay_ep_idx]
                replay_ep_idx += 1
                goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan_from_direction(
                    recent_images[-12:],
                    int(replay_row["direction"]),
                    goal_flag=bool(replay_row["goal_flag"]),
                    return_debug_image=False,
                )
                cached_goal_rotate = int(goal_rotate)
                cached_goal_flag = bool(goal_flag)
                _goal_flag_history.append(cached_goal_flag)
            else:
                goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan(
                    recent_images[-12:],
                    return_debug_image=False,
                )
                cached_goal_rotate = int(goal_rotate)
                cached_goal_flag = bool(goal_flag)
                _goal_flag_history.append(cached_goal_flag)
                if planner_trace_path is not None:
                    with planner_trace_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "episode_idx": int(episode_idx),
                                    "step_idx": int(step_idx),
                                    "direction": int(goal_rotate),
                                    "goal_flag": int(bool(goal_flag)),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            for _ in range(min(11 - goal_rotate, 1 + goal_rotate)):
                obs, step_idx, ok = execute_action(
                    habitat_env,
                    obs,
                    3 if goal_rotate < 6 else 2,
                    recent_images=recent_images,
                    rgb_frames=rgb_frames,
                    topdown_frames=topdown_frames,
                    action_history=action_history,
                    collision_history=collision_history,
                    dtg_history=dtg_history,
                    step_idx=step_idx,
                    memory=memory,
                    recent_nodes=recent_nodes,
                    memory_stats=memory_stats,
                    memory_mode=memory_mode,
                    fifo_history_size=int(args.fifo_history_size),
                    token_budget=int(args.token_budget),
                    memory_compression_strategy=str(args.memory_compression_strategy),
                    scene_id=scene_id,
                    budget_manager=budget_manager,
                    logger=logger,
                    episode_idx=int(episode_idx),
                    episode_id=episode_id,
                    position_history=position_history,
                )
                if not ok:
                    break
            host.reset(
                episode.object_category,
                goal_image=goal_image,
                goal_mask=goal_mask,
            )

            while not habitat_env.episode_over:
                action = host.act(obs["rgb"], collided=habitat_env.sim.previous_step_collided)
                if logger is not None:
                    logger.log_step(
                        {
                            "episode": int(episode_idx),
                            "step": int(step_idx),
                            "episode_id": episode_id,
                            "scene_id": scene_id,
                            "action": int(action),
                            "collision": int(bool(habitat_env.sim.previous_step_collided)),
                            "goal_flag": int(bool(goal_flag)),
                            **host.export_telemetry(),
                        }
                    )
                _proximity_stop = False
                if (
                    action == 0
                    and not goal_flag
                    and float(args.proximity_stop_distance) > 0
                ):
                    _dtg_now = float(habitat_env.get_metrics().get("distance_to_goal", 999.0))
                    if _dtg_now < float(args.proximity_stop_distance):
                        _proximity_stop = True
                if action != 0 or goal_flag or _proximity_stop:
                    if action == 4:
                        heading_offset += 1
                    elif action == 5:
                        heading_offset -= 1
                    obs, step_idx, ok = execute_action(
                        habitat_env,
                        obs,
                        int(action),
                        recent_images=recent_images,
                        rgb_frames=rgb_frames,
                        topdown_frames=topdown_frames,
                        action_history=action_history,
                        collision_history=collision_history,
                        dtg_history=dtg_history,
                        step_idx=step_idx,
                        memory=memory,
                        recent_nodes=recent_nodes,
                        memory_stats=memory_stats,
                        memory_mode=memory_mode,
                        fifo_history_size=int(args.fifo_history_size),
                        token_budget=int(args.token_budget),
                        memory_compression_strategy=str(args.memory_compression_strategy),
                        scene_id=scene_id,
                        budget_manager=budget_manager,
                        logger=logger,
                        episode_idx=int(episode_idx),
                        episode_id=episode_id,
                        position_history=position_history,
                    )
                    if not ok:
                        recovery_stats["execution_broken_count"] += 1
                        break
                else:
                    if habitat_env.episode_over:
                        break
                    gate_stats["replan_event_count"] += 1
                    failure_opportunity = is_failure_opportunity(
                        dtg_history,
                        window=int(args.gate_failure_window),
                        progress_epsilon=float(args.gate_progress_epsilon),
                    )
                    if str(args.progress_signal) != "oracle":
                        failure_opportunity = False
                    if failure_opportunity:
                        gate_stats["failure_opportunity_count"] += 1
                    collision_ratio = compute_collision_ratio(collision_history, int(args.gate_failure_window))
                    oscillation_ratio = compute_turn_oscillation_ratio(action_history, int(args.gate_failure_window))

                    # M4: Compute additional gate signals
                    _current_pos = get_agent_position(habitat_env)
                    _novelty = (
                        compute_novelty(position_history, _current_pos, window=int(args.gate_novelty_window))
                        if _current_pos and position_history
                        else 1.0
                    )
                    _revisit_rate = (
                        compute_revisit_rate(position_history, _current_pos, window=int(args.gate_novelty_window))
                        if _current_pos and position_history
                        else 0.0
                    )
                    _visual_stag = float(compute_visual_stagnation(
                        recent_images,
                        window=int(args.gate_visual_stagnation_window),
                        threshold=float(args.gate_visual_stagnation_threshold),
                    ))

                    _skip_post_recovery_scan = False
                    trigger_recovery = False
                    gate_score = 0.0
                    gate_reason = ""
                    if args.recovery_mode != "host_only":
                        if recovery_gate is None:
                            trigger_recovery = True
                            gate_reason = "legacy_stop_signal"
                        else:
                            _gate_inputs = GateInputs(
                                step_idx=int(gate_stats["replan_event_count"]),
                                novelty=_novelty,
                                revisit_rate=_revisit_rate,
                                collision_rate=collision_ratio,
                                oscillation_rate=oscillation_ratio,
                                no_semantic_progress=float(failure_opportunity),
                                visual_stagnation=_visual_stag,
                            )
                            _gate_decision = recovery_gate.observe(_gate_inputs)
                            trigger_recovery = _gate_decision.should_trigger
                            gate_score = _gate_decision.score
                            gate_reason = _gate_decision.reason
                            if not trigger_recovery and args.gate_failure_alone_triggers and failure_opportunity:
                                trigger_recovery = True
                                gate_reason = "failure_alone_bypass"

                    _veto_reason = ""
                    if trigger_recovery:
                        _ghw = int(args.gate_goal_hysteresis_window)
                        if _ghw > 0 and any(_goal_flag_history[-_ghw:]):
                            trigger_recovery = False
                            _veto_reason = "goal_seen_recently"
                    if trigger_recovery:
                        _lbr = int(args.gate_late_budget_reserve)
                        if _lbr > 0:
                            _max_ep_steps = int(args.max_episode_steps) if int(args.max_episode_steps) > 0 else 500
                            if (_max_ep_steps - step_idx) < _lbr:
                                trigger_recovery = False
                                _veto_reason = "late_budget"
                    if trigger_recovery:
                        _dtg_margin = float(args.gate_dtg_protect_margin)
                        if _dtg_margin > 0 and len(dtg_history) > 1:
                            _best_dtg = min(dtg_history)
                            _current_dtg = dtg_history[-1]
                            if _current_dtg <= _best_dtg + _dtg_margin:
                                trigger_recovery = False
                                _veto_reason = "dtg_near_best"

                    if trigger_recovery:
                        gate_stats["trigger_count"] += 1
                        if failure_opportunity:
                            gate_stats["true_trigger_count"] += 1
                            gate_stats["trigger_overlap_count"] += 1
                        else:
                            gate_stats["false_trigger_count"] += 1
                    budget_allows_recovery = True
                    budget_skip_reason = ""
                    if trigger_recovery:
                        budget_allows_recovery, budget_skip_reason = budget_manager.can_start_slow_call()
                    if logger is not None:
                        logger.log_event(
                            "gate_decision",
                            {
                                "episode": int(episode_idx),
                                "step": int(step_idx),
                                "episode_id": episode_id,
                                "scene_id": scene_id,
                                "gate_mode": str(args.gate_mode),
                                "failure_opportunity": int(failure_opportunity),
                                "collision_ratio": float(collision_ratio),
                                "oscillation_ratio": float(oscillation_ratio),
                                "novelty": float(_novelty),
                                "revisit_rate": float(_revisit_rate),
                                "visual_stagnation": float(_visual_stag),
                                "gate_score": float(gate_score),
                                "gate_reason": str(gate_reason),
                                "trigger_recovery": int(trigger_recovery),
                                "veto_reason": str(_veto_reason),
                                "budget_allows_recovery": int(budget_allows_recovery),
                                "budget_skip_reason": str(budget_skip_reason),
                                "call_budget": -1 if effective_call_budget is None else int(effective_call_budget),
                                "slow_calls_episode": int(budget_manager.slow_calls_episode),
                                "remaining_budget_ratio": float(budget_manager.remaining_budget_ratio()),
                                "hard_cap_reached": int(budget_manager.hard_cap_reached()),
                            },
                        )

                    if trigger_recovery and budget_allows_recovery:
                        _replay_active = False
                        budget_manager.reserve_slow_call()
                        log_budget_snapshot(
                            logger,
                            budget_manager=budget_manager,
                            episode_idx=int(episode_idx),
                            episode_id=episode_id,
                            scene_id=scene_id,
                            step_idx=int(step_idx),
                            reason="recovery_started",
                        )
                        recovery_stats["recovery_attempts"] += 1
                        dtg_before = float(habitat_env.get_metrics().get("distance_to_goal", 0.0))
                        executed_recovery_steps = 0
                        recovery_broken = False
                        _ce_edge_result = None
                        dtg_after_backtrack = dtg_before
                        dtg_after_selected_edge = dtg_before
                        dtg_after_continuation = dtg_before
                        if args.recovery_mode == "constrained_edge" and memory is not None:
                            # --- Full constrained-edge recovery ---
                            # Phase 1: Backtrack to a branch node
                            backtrack_plan = build_minimal_backtrack_plan(
                                action_history,
                                int(args.recovery_backtrack_steps),
                            )
                            for bk_action in backtrack_plan:
                                if int(bk_action) not in VALID_ACTION_IDS:
                                    continue
                                obs, step_idx, ok = execute_action(
                                    habitat_env,
                                    obs,
                                    int(bk_action),
                                    recent_images=recent_images,
                                    rgb_frames=rgb_frames,
                                    topdown_frames=topdown_frames,
                                    action_history=action_history,
                                    collision_history=collision_history,
                                    dtg_history=dtg_history,
                                    step_idx=step_idx,
                                    memory=memory,
                                    recent_nodes=recent_nodes,
                                    memory_stats=memory_stats,
                                    memory_mode=memory_mode,
                                    fifo_history_size=int(args.fifo_history_size),
                                    token_budget=int(args.token_budget),
                                    memory_compression_strategy=str(args.memory_compression_strategy),
                                    scene_id=scene_id,
                                    source="recovery_backtrack",
                                    budget_manager=budget_manager,
                                    logger=logger,
                                    episode_idx=int(episode_idx),
                                    episode_id=episode_id,
                                    position_history=position_history,
                                )
                                if not ok:
                                    recovery_stats["execution_broken_count"] += 1
                                    recovery_broken = True
                                    break
                                executed_recovery_steps += 1
                                if habitat_env.episode_over:
                                    break
                            dtg_after_backtrack = float(
                                habitat_env.get_metrics().get("distance_to_goal", dtg_before)
                            )
                            # Phase 2: Constrained edge selection via recovery engine
                            if not recovery_broken and not habitat_env.episode_over:
                                _ce = {"obs": obs, "step": step_idx, "steps": 0}

                                def _ce_collision_fn(action_name: str, edge_id: str) -> bool:
                                    aid = ACTION_NAME_TO_ID.get(action_name)
                                    if aid is None:
                                        return True
                                    new_obs, new_step, ok = execute_action(
                                        habitat_env,
                                        _ce["obs"],
                                        int(aid),
                                        recent_images=recent_images,
                                        rgb_frames=rgb_frames,
                                        topdown_frames=topdown_frames,
                                        action_history=action_history,
                                        collision_history=collision_history,
                                        dtg_history=dtg_history,
                                        step_idx=_ce["step"],
                                        memory=memory,
                                        recent_nodes=recent_nodes,
                                        memory_stats=memory_stats,
                                        memory_mode=memory_mode,
                                        fifo_history_size=int(args.fifo_history_size),
                                        token_budget=int(args.token_budget),
                                        memory_compression_strategy=str(args.memory_compression_strategy),
                                        scene_id=scene_id,
                                        source="recovery_edge",
                                        budget_manager=budget_manager,
                                        logger=logger,
                                        episode_idx=int(episode_idx),
                                        episode_id=episode_id,
                                        position_history=position_history,
                                    )
                                    _ce["obs"] = new_obs
                                    _ce["step"] = new_step
                                    if not ok:
                                        return True
                                    _ce["steps"] += 1
                                    return bool(habitat_env.sim.previous_step_collided)

                                planning_node = memory.last_node_id
                                target_hint = str(habitat_env.current_episode.object_category)
                                _ce_edge_result = recovery_engine.run(
                                    memory=memory,
                                    current_node=planning_node,
                                    recent_nodes=recent_nodes,
                                    actions=["move_forward", "turn_left", "turn_right"],
                                    target_hint=target_hint,
                                    max_retries=int(args.recovery_max_retries),
                                    collision_fn=_ce_collision_fn,
                                )
                                obs = _ce["obs"]
                                step_idx = _ce["step"]
                                executed_recovery_steps += _ce["steps"]
                                dtg_after_selected_edge = float(
                                    habitat_env.get_metrics().get("distance_to_goal", dtg_before)
                                )
                                # Phase 3: Forward continuation — walk forward
                                # after direction selection to produce spatial
                                # displacement that can actually improve DTG.
                                _cont_steps = int(args.recovery_continuation_steps)
                                _cont_dtg_guided = bool(int(args.recovery_dtg_guided_continuation))
                                if (
                                    _ce_edge_result is not None
                                    and _ce_edge_result.success
                                    and _cont_steps > 0
                                    and not recovery_broken
                                    and not habitat_env.episode_over
                                ):
                                    _cont_dtg_prev = float(
                                        habitat_env.get_metrics().get("distance_to_goal", 0.0)
                                    )
                                    _fwd_id = ACTION_NAME_TO_ID["move_forward"]
                                    for _ci in range(_cont_steps):
                                        if habitat_env.episode_over:
                                            break
                                        obs, step_idx, ok = execute_action(
                                            habitat_env,
                                            obs,
                                            _fwd_id,
                                            recent_images=recent_images,
                                            rgb_frames=rgb_frames,
                                            topdown_frames=topdown_frames,
                                            action_history=action_history,
                                            collision_history=collision_history,
                                            dtg_history=dtg_history,
                                            step_idx=step_idx,
                                            memory=memory,
                                            recent_nodes=recent_nodes,
                                            memory_stats=memory_stats,
                                            memory_mode=memory_mode,
                                            fifo_history_size=int(args.fifo_history_size),
                                            token_budget=int(args.token_budget),
                                            memory_compression_strategy=str(
                                                args.memory_compression_strategy
                                            ),
                                            scene_id=scene_id,
                                            source="recovery_continuation",
                                            budget_manager=budget_manager,
                                            logger=logger,
                                            episode_idx=int(episode_idx),
                                            episode_id=episode_id,
                                            position_history=position_history,
                                        )
                                        if not ok:
                                            recovery_broken = True
                                            recovery_stats["execution_broken_count"] += 1
                                            break
                                        executed_recovery_steps += 1
                                        if habitat_env.sim.previous_step_collided:
                                            break
                                        if _cont_dtg_guided:
                                            _cont_dtg_now = float(
                                                habitat_env.get_metrics().get(
                                                    "distance_to_goal", _cont_dtg_prev
                                                )
                                            )
                                            if _cont_dtg_now > _cont_dtg_prev + 0.01:
                                                break
                                            _cont_dtg_prev = _cont_dtg_now
                                dtg_after_continuation = float(
                                    habitat_env.get_metrics().get("distance_to_goal", dtg_before)
                                )
                        elif args.recovery_mode == "macro_action":
                            # --- Probe-then-execute macro recovery (v2) ---
                            # Phase 1: Build candidate offsets
                            _macro_max_fan = int(args.recovery_macro_max_fan)
                            _macro_fwd_steps = max(1, int(args.recovery_macro_forward_steps))
                            _macro_probe_steps = int(args.recovery_macro_probe_steps)
                            if _macro_probe_steps < 0:
                                _macro_probe_steps = _macro_fwd_steps
                            else:
                                _macro_probe_steps = max(1, _macro_probe_steps)
                            _macro_abstain_enabled = bool(int(args.recovery_macro_abstain))
                            _fan_offsets = [0]
                            for _fi in range(1, (_macro_max_fan + 1) // 2 + 1):
                                _fan_offsets.append(_fi)
                                _fan_offsets.append(-_fi)
                            _fan_offsets = _fan_offsets[:_macro_max_fan]

                            # Phase 2: Probe all candidates via sim save/restore (no bookkeeping cost)
                            # Probe uses 1 forward step for direction selection only.
                            _macro_ranking = str(args.recovery_macro_ranking).strip().lower()
                            _progress_signal = str(args.progress_signal).strip().lower()
                            if _progress_signal == "proxy":
                                _macro_ranking = "proxy"
                                _needs_memory = True
                            elif _progress_signal == "none":
                                _macro_ranking = "random"
                                _needs_memory = True
                            else:
                                _needs_memory = _macro_ranking in ("memory_dtg", "composite", "frontier_nearest")
                            _probe_results = probe_macro_candidates(
                                habitat_env, _fan_offsets,
                                memory=memory if _needs_memory else None,
                                scene_id=scene_id,
                                probe_steps=_macro_probe_steps,
                            )

                            # Phase 3: Rank candidates
                            _REVISIT_PENALTY = 0.3
                            if _macro_ranking == "random":
                                _ranked = list(_probe_results)
                                random.shuffle(_ranked)
                            elif _macro_ranking == "composite":
                                _ranked = sorted(
                                    _probe_results,
                                    key=lambda r: composite_score(r, dtg_before),
                                )
                            elif _macro_ranking == "proxy":
                                _ranked = sorted(
                                    _probe_results,
                                    key=lambda r: proxy_score(r),
                                )
                            elif _macro_ranking == "frontier_nearest":
                                _ranked = sorted(
                                    _probe_results,
                                    key=lambda r: (r.get("visit_count", 0), -r.get("unexplored_edges", 0), r["dtg"]),
                                )
                            elif _macro_ranking == "memory_dtg":
                                _ranked = sorted(
                                    _probe_results,
                                    key=lambda r: r["dtg"] + _REVISIT_PENALTY * r.get("visit_count", 0),
                                )
                            else:
                                _ranked = sorted(_probe_results, key=lambda r: r["dtg"])
                            _macro_success = False
                            _macro_tried = len(_fan_offsets)
                            _macro_winning_offset = None
                            _macro_abstained = False
                            _macro_attempts = 0
                            _best: dict = {}
                            _TURN_R = 3
                            _TURN_L = 2
                            _FWD = 1
                            _closed_loop = bool(int(args.recovery_closed_loop))
                            _max_retries = max(0, int(args.recovery_max_retries))

                            # M6: Constraint — filter out memory-blocked directions
                            _failed_offsets: set[int] = set()
                            if memory is not None and _needs_memory:
                                for _pr in _ranked:
                                    if _pr.get("blocked_edges", 0) > 0 and _pr.get("visit_count", 0) > 0:
                                        _failed_offsets.add(_pr["offset"])

                            # Phase 4: Closed-loop retry
                            for _attempt in range(1 + _max_retries if _closed_loop else 1):
                                _available = [r for r in _ranked if r["offset"] not in _failed_offsets]
                                if not _available:
                                    _macro_abstained = True
                                    _macro_winning_offset = -999
                                    break
                                _best = _available[0]

                                if _macro_abstain_enabled and _progress_signal == "oracle" and _best["dtg"] >= dtg_before:
                                    _macro_abstained = True
                                    _macro_winning_offset = -999
                                    break

                                _target_offset = _best["offset"]
                                _macro_winning_offset = _target_offset
                                _macro_attempts += 1
                                _attempt_start_pos = np.array(
                                    habitat_env.sim.get_agent_state().position, dtype=np.float32
                                )

                                # Execute turns to winning direction
                                _turn_id = _TURN_R if _target_offset > 0 else _TURN_L
                                for _ in range(abs(_target_offset)):
                                    if recovery_broken or habitat_env.episode_over:
                                        break
                                    obs, step_idx, ok = execute_action(
                                        habitat_env, obs, _turn_id,
                                        recent_images=recent_images,
                                        rgb_frames=rgb_frames,
                                        topdown_frames=topdown_frames,
                                        action_history=action_history,
                                        collision_history=collision_history,
                                        dtg_history=dtg_history,
                                        step_idx=step_idx,
                                        memory=memory,
                                        recent_nodes=recent_nodes,
                                        memory_stats=memory_stats,
                                        memory_mode=memory_mode,
                                        fifo_history_size=int(args.fifo_history_size),
                                        token_budget=int(args.token_budget),
                                        memory_compression_strategy=str(
                                            args.memory_compression_strategy
                                        ),
                                        scene_id=scene_id,
                                        source="recovery_macro_turn",
                                        budget_manager=budget_manager,
                                        logger=logger,
                                        episode_idx=int(episode_idx),
                                        episode_id=episode_id,
                                        position_history=position_history,
                                    )
                                    if not ok:
                                        recovery_broken = True
                                        recovery_stats["execution_broken_count"] += 1
                                        break
                                    executed_recovery_steps += 1

                                # Execute forward with DTG-guided stopping
                                _fwd_collided = False
                                if not recovery_broken and not habitat_env.episode_over:
                                    _cont_dtg_prev = float(
                                        habitat_env.get_metrics().get(
                                            "distance_to_goal", dtg_before
                                        )
                                    )
                                    for _fwd_i in range(_macro_fwd_steps):
                                        if habitat_env.episode_over:
                                            break
                                        obs, step_idx, ok = execute_action(
                                            habitat_env, obs, _FWD,
                                            recent_images=recent_images,
                                            rgb_frames=rgb_frames,
                                            topdown_frames=topdown_frames,
                                            action_history=action_history,
                                            collision_history=collision_history,
                                            dtg_history=dtg_history,
                                            step_idx=step_idx,
                                            memory=memory,
                                            recent_nodes=recent_nodes,
                                            memory_stats=memory_stats,
                                            memory_mode=memory_mode,
                                            fifo_history_size=int(args.fifo_history_size),
                                            token_budget=int(args.token_budget),
                                            memory_compression_strategy=str(
                                                args.memory_compression_strategy
                                            ),
                                            scene_id=scene_id,
                                            source="recovery_macro_forward",
                                            budget_manager=budget_manager,
                                            logger=logger,
                                            episode_idx=int(episode_idx),
                                            episode_id=episode_id,
                                            position_history=position_history,
                                        )
                                        if not ok:
                                            recovery_broken = True
                                            recovery_stats["execution_broken_count"] += 1
                                            break
                                        executed_recovery_steps += 1
                                        if habitat_env.sim.previous_step_collided:
                                            _fwd_collided = True
                                            break
                                        _cont_dtg_now = float(
                                            habitat_env.get_metrics().get(
                                                "distance_to_goal", _cont_dtg_prev
                                            )
                                        )
                                        if _progress_signal == "oracle" and _cont_dtg_now > _cont_dtg_prev + 0.01:
                                            break
                                        _cont_dtg_prev = _cont_dtg_now

                                # M6: Evaluate attempt outcome
                                _attempt_dtg = float(
                                    habitat_env.get_metrics().get("distance_to_goal", dtg_before)
                                )
                                if _progress_signal == "oracle":
                                    _attempt_improved = _attempt_dtg < dtg_before - 0.01
                                else:
                                    _attempt_end_pos = np.array(
                                        habitat_env.sim.get_agent_state().position, dtype=np.float32
                                    )
                                    _displacement = float(np.linalg.norm(_attempt_end_pos - _attempt_start_pos))
                                    _attempt_improved = _displacement > 0.15 and not _fwd_collided

                                if _attempt_improved:
                                    _macro_success = True
                                elif not _fwd_collided and not _closed_loop:
                                    _macro_success = True

                                # M6: Closed-loop memory update
                                if _closed_loop and memory is not None and memory.last_node_id:
                                    _cl_node = memory.last_node_id
                                    if _cl_node in memory.nodes:
                                        if _attempt_improved:
                                            memory.nodes[_cl_node].is_recovery_anchor = True
                                        else:
                                            memory.nodes[_cl_node].meta["low_yield_recovery"] = True

                                if _macro_success or recovery_broken or habitat_env.episode_over:
                                    break

                                # M6: Mark failed offset and retry with next best
                                _failed_offsets.add(_target_offset)
                                # Re-probe from current (moved) position
                                dtg_before = _attempt_dtg
                                _probe_results = probe_macro_candidates(
                                    habitat_env, _fan_offsets,
                                    memory=memory if _needs_memory else None,
                                    scene_id=scene_id,
                                    probe_steps=_macro_probe_steps,
                                )
                                if _macro_ranking == "random":
                                    _ranked = list(_probe_results)
                                    random.shuffle(_ranked)
                                elif _macro_ranking == "composite":
                                    _ranked = sorted(_probe_results, key=lambda r: composite_score(r, dtg_before))
                                elif _macro_ranking == "proxy":
                                    _ranked = sorted(_probe_results, key=lambda r: proxy_score(r))
                                elif _macro_ranking == "frontier_nearest":
                                    _ranked = sorted(_probe_results, key=lambda r: (r.get("visit_count", 0), -r.get("unexplored_edges", 0), r["dtg"]))
                                elif _macro_ranking == "memory_dtg":
                                    _ranked = sorted(_probe_results, key=lambda r: r["dtg"] + _REVISIT_PENALTY * r.get("visit_count", 0))
                                else:
                                    _ranked = sorted(_probe_results, key=lambda r: r["dtg"])

                            dtg_after_selected_edge = float(
                                habitat_env.get_metrics().get("distance_to_goal", dtg_before)
                            )
                            dtg_after_continuation = dtg_after_selected_edge
                            if not _macro_abstained and executed_recovery_steps > 0 and not recovery_broken:
                                if str(args.post_recovery_replan) != "full":
                                    _skip_post_recovery_scan = True
                        else:
                            # --- Standard plan-based recovery (legacy modes) ---
                            if args.recovery_mode == "minimal_backtrack":
                                recovery_plan = build_minimal_backtrack_plan(
                                    action_history,
                                    int(args.recovery_backtrack_steps),
                                )
                            elif args.recovery_mode == "naive_llm":
                                try:
                                    _, _, _, naive_rotate, _ = nav_planner.make_plan(
                                        recent_images[-12:],
                                        return_debug_image=False,
                                    )
                                    recovery_plan = [choose_naive_llm_action(int(naive_rotate))]
                                except Exception:  # noqa: BLE001
                                    recovery_plan = [
                                        choose_heuristic_edge_action(
                                            action_history,
                                            int(args.recovery_heuristic_window),
                                        )
                                    ]
                            else:
                                constrained_action = None
                                if not args.disable_memory_constrained_edge:
                                    constrained_action = choose_memory_constrained_action(
                                        memory=memory,
                                        current_node=memory.last_node_id if memory is not None else None,
                                    )
                                recovery_plan = [
                                    int(constrained_action)
                                    if constrained_action is not None
                                    else choose_heuristic_edge_action(
                                        action_history,
                                        int(args.recovery_heuristic_window),
                                    )
                                ]
                            for recovery_action in recovery_plan:
                                if int(recovery_action) not in VALID_ACTION_IDS:
                                    recovery_stats["selection_illegal_count"] += 1
                                    continue
                                obs, step_idx, ok = execute_action(
                                    habitat_env,
                                    obs,
                                    int(recovery_action),
                                    recent_images=recent_images,
                                    rgb_frames=rgb_frames,
                                    topdown_frames=topdown_frames,
                                    action_history=action_history,
                                    collision_history=collision_history,
                                    dtg_history=dtg_history,
                                    step_idx=step_idx,
                                    memory=memory,
                                    recent_nodes=recent_nodes,
                                    memory_stats=memory_stats,
                                    memory_mode=memory_mode,
                                    fifo_history_size=int(args.fifo_history_size),
                                    token_budget=int(args.token_budget),
                                    memory_compression_strategy=str(args.memory_compression_strategy),
                                    scene_id=scene_id,
                                    source="recovery",
                                    budget_manager=budget_manager,
                                    logger=logger,
                                    episode_idx=int(episode_idx),
                                    episode_id=episode_id,
                                    position_history=position_history,
                                )
                                if not ok:
                                    recovery_stats["execution_broken_count"] += 1
                                    recovery_broken = True
                                    break
                                executed_recovery_steps += 1
                                if habitat_env.episode_over:
                                    break
                        dtg_after = float(habitat_env.get_metrics().get("distance_to_goal", dtg_before))
                        dtg_improved = (dtg_before - dtg_after) > float(args.recovery_progress_threshold)
                        edge_executable_success = (
                            _ce_edge_result.success if _ce_edge_result is not None else None
                        )
                        paper_recovery_success = (
                            executed_recovery_steps > 0
                            and not recovery_broken
                            and dtg_improved
                        )
                        if paper_recovery_success:
                            recovery_stats["recovery_success_count"] += 1
                        else:
                            recovery_stats["recovery_failure_count"] += 1
                            recovery_stats["closed_loop_fail_count"] += 1
                        budget_manager.finalize_slow_call(tokens_this_call=0.0, latency_this_call_ms=0.0)
                        log_budget_snapshot(
                            logger,
                            budget_manager=budget_manager,
                            episode_idx=int(episode_idx),
                            episode_id=episode_id,
                            scene_id=scene_id,
                            step_idx=int(step_idx),
                            reason="recovery_finished",
                        )
                        if logger is not None:
                            _rec_log: dict[str, object] = {
                                "episode": int(episode_idx),
                                "step": int(step_idx),
                                "episode_id": episode_id,
                                "scene_id": scene_id,
                                "recovery_mode": str(args.recovery_mode),
                                "executed_recovery_steps": int(executed_recovery_steps),
                                "dtg_before": float(dtg_before),
                                "dtg_after": float(dtg_after),
                                "dtg_delta": float(dtg_before - dtg_after),
                                "broken": int(recovery_broken),
                                "call_budget": -1 if effective_call_budget is None else int(effective_call_budget),
                                "slow_calls_episode": int(budget_manager.slow_calls_episode),
                            }
                            _rec_log["improved"] = int(paper_recovery_success)
                            _rec_log["dtg_improved"] = int(dtg_improved)
                            _rec_log["edge_executable_success"] = (
                                int(edge_executable_success)
                                if edge_executable_success is not None
                                else -1
                            )
                            _rec_log["dtg_after_backtrack"] = float(dtg_after_backtrack)
                            _rec_log["dtg_after_selected_edge"] = float(dtg_after_selected_edge)
                            _rec_log["dtg_after_continuation"] = float(dtg_after_continuation)
                            if _ce_edge_result is not None:
                                _rec_log["edge_result_reason"] = str(_ce_edge_result.reason)
                                _rec_log["edge_selected_action"] = str(_ce_edge_result.selected_action or "")
                                _rec_log["edge_retries"] = int(_ce_edge_result.retries)
                                _rec_log["edge_candidate_count"] = int(_ce_edge_result.candidate_count_initial)
                                _rec_log["planning_node"] = str(_ce_edge_result.planning_node or "")
                                _rec_log["selected_edge_from_node"] = str(
                                    _ce_edge_result.selected_edge_from_node or ""
                                )
                                _rec_log["source_is_local"] = int(
                                    _ce_edge_result.selected_edge_from_node == _ce_edge_result.planning_node
                                ) if _ce_edge_result.selected_edge_from_node and _ce_edge_result.planning_node else -1
                                _rec_log["edge_query_mode"] = str(
                                    _ce_edge_result.candidate_query_mode or ""
                                )
                            if args.recovery_mode == "macro_action":
                                _rec_log["macro_success"] = int(_macro_success)
                                _rec_log["macro_tried"] = int(_macro_tried)
                                _rec_log["macro_winning_offset"] = (
                                    int(_macro_winning_offset)
                                    if _macro_winning_offset is not None
                                    else -999
                                )
                                _rec_log["macro_max_fan"] = int(args.recovery_macro_max_fan)
                                _rec_log["macro_abstained"] = int(_macro_abstained)
                                _rec_log["macro_best_probe_dtg"] = float(_best["dtg"]) if _best else -1.0
                                _rec_log["macro_ranking"] = str(args.recovery_macro_ranking)
                                _rec_log["macro_best_visit_count"] = int(_best.get("visit_count", 0)) if _best else 0
                                _rec_log["macro_attempts"] = int(_macro_attempts)
                                _rec_log["macro_closed_loop"] = int(args.recovery_closed_loop)
                                _rec_log["macro_blocked_filtered"] = int(len(_failed_offsets))
                            logger.log_event("recovery_attempt", _rec_log)
                    elif trigger_recovery:
                        budget_stats["call_budget_skip_count"] += 1
                        log_budget_snapshot(
                            logger,
                            budget_manager=budget_manager,
                            episode_idx=int(episode_idx),
                            episode_id=episode_id,
                            scene_id=scene_id,
                            step_idx=int(step_idx),
                            reason=f"recovery_skipped_{budget_skip_reason or 'budget'}",
                        )
                    if _skip_post_recovery_scan:
                        heading_offset = 0
                        goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan_from_direction(
                            [obs["rgb"]], 0, goal_flag=False, return_debug_image=False,
                        )
                        goal_rotate = 11
                        cached_goal_rotate = 11
                        cached_goal_flag = False
                        _goal_flag_history.append(False)
                    else:
                      for _ in range(abs(heading_offset)):
                        if habitat_env.episode_over:
                            break
                        if heading_offset > 0:
                            heading_action = 5
                            heading_offset -= 1
                        else:
                            heading_action = 4
                            heading_offset += 1
                        obs, step_idx, ok = execute_action(
                            habitat_env,
                            obs,
                            heading_action,
                            recent_images=recent_images,
                            rgb_frames=rgb_frames,
                            topdown_frames=topdown_frames,
                            action_history=action_history,
                            collision_history=collision_history,
                            dtg_history=dtg_history,
                            step_idx=step_idx,
                            memory=memory,
                            recent_nodes=recent_nodes,
                            memory_stats=memory_stats,
                            memory_mode=memory_mode,
                            fifo_history_size=int(args.fifo_history_size),
                            token_budget=int(args.token_budget),
                            memory_compression_strategy=str(args.memory_compression_strategy),
                            scene_id=scene_id,
                            budget_manager=budget_manager,
                            logger=logger,
                            episode_idx=int(episode_idx),
                            episode_id=episode_id,
                            position_history=position_history,
                        )
                        if not ok:
                            recovery_stats["execution_broken_count"] += 1
                            break
                      for _ in range(11):
                        if habitat_env.episode_over:
                            break
                        obs, step_idx, ok = execute_action(
                            habitat_env,
                            obs,
                            3,
                            recent_images=recent_images,
                            rgb_frames=rgb_frames,
                            topdown_frames=topdown_frames,
                            action_history=action_history,
                            collision_history=collision_history,
                            dtg_history=dtg_history,
                            step_idx=step_idx,
                            memory=memory,
                            recent_nodes=recent_nodes,
                            memory_stats=memory_stats,
                            memory_mode=memory_mode,
                            fifo_history_size=int(args.fifo_history_size),
                            token_budget=int(args.token_budget),
                            memory_compression_strategy=str(args.memory_compression_strategy),
                            scene_id=scene_id,
                            budget_manager=budget_manager,
                            logger=logger,
                            episode_idx=int(episode_idx),
                            episode_id=episode_id,
                            position_history=position_history,
                        )
                        if not ok:
                            recovery_stats["execution_broken_count"] += 1
                            break
                    if not _skip_post_recovery_scan:
                      if _replay_active and replay_trace_by_ep is not None and replay_ep_idx < len(replay_ep_seq):
                        replay_row = replay_ep_seq[replay_ep_idx]
                        replay_ep_idx += 1
                        goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan_from_direction(
                            recent_images[-12:],
                            int(replay_row["direction"]),
                            goal_flag=bool(replay_row["goal_flag"]),
                            return_debug_image=False,
                        )
                        cached_goal_rotate = int(goal_rotate)
                        cached_goal_flag = bool(goal_flag)
                        _goal_flag_history.append(cached_goal_flag)
                      else:
                        use_cached_plan = (
                            planner_replan_policy == "stride"
                            and cached_goal_rotate is not None
                            and (int(gate_stats["replan_event_count"]) % planner_replan_stride) != 0
                        )
                        if use_cached_plan:
                            goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan_from_direction(
                                recent_images[-12:],
                                int(cached_goal_rotate),
                                goal_flag=bool(cached_goal_flag),
                                return_debug_image=False,
                            )
                        else:
                            goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan(
                                recent_images[-12:],
                                return_debug_image=False,
                            )
                            cached_goal_rotate = int(goal_rotate)
                            cached_goal_flag = bool(goal_flag)
                            _goal_flag_history.append(cached_goal_flag)
                            if planner_trace_path is not None:
                                with planner_trace_path.open("a", encoding="utf-8") as f:
                                    f.write(
                                        json.dumps(
                                            {
                                                "episode_idx": int(episode_idx),
                                                "step_idx": int(step_idx),
                                                "direction": int(goal_rotate),
                                                "goal_flag": int(bool(goal_flag)),
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
                    for _ in range(min(11 - goal_rotate, goal_rotate + 1)):
                        if habitat_env.episode_over:
                            break
                        obs, step_idx, ok = execute_action(
                            habitat_env,
                            obs,
                            3 if goal_rotate < 6 else 2,
                            recent_images=recent_images,
                            rgb_frames=rgb_frames,
                            topdown_frames=topdown_frames,
                            action_history=action_history,
                            collision_history=collision_history,
                            dtg_history=dtg_history,
                            step_idx=step_idx,
                            memory=memory,
                            recent_nodes=recent_nodes,
                            memory_stats=memory_stats,
                            memory_mode=memory_mode,
                            fifo_history_size=int(args.fifo_history_size),
                            token_budget=int(args.token_budget),
                            memory_compression_strategy=str(args.memory_compression_strategy),
                            scene_id=scene_id,
                            budget_manager=budget_manager,
                            logger=logger,
                            episode_idx=int(episode_idx),
                            episode_id=episode_id,
                            position_history=position_history,
                        )
                        if not ok:
                            recovery_stats["execution_broken_count"] += 1
                            break
                    host.reset(
                        episode.object_category,
                        goal_image=goal_image,
                        goal_mask=goal_mask,
                    )

            if rgb_frames is not None:
                fps_writer = imageio.get_writer(str(episode_dir / "fps.mp4"), fps=4)
                for image in rgb_frames:
                    fps_writer.append_data(image)
                fps_writer.close()
            if topdown_frames is not None:
                topdown_writer = imageio.get_writer(str(episode_dir / "metric.mp4"), fps=4)
                for topdown in topdown_frames:
                    topdown_writer.append_data(topdown)
                topdown_writer.close()

            ep_metrics = habitat_env.get_metrics()
            trigger_precision = (
                float(gate_stats["true_trigger_count"]) / float(max(1, gate_stats["trigger_count"]))
            )
            trigger_recall = (
                float(gate_stats["true_trigger_count"]) / float(max(1, gate_stats["failure_opportunity_count"]))
            )
            false_trigger_rate = (
                float(gate_stats["false_trigger_count"]) / float(max(1, gate_stats["trigger_count"]))
            )
            memory_node_count_final = int(len(memory.nodes)) if memory is not None else 0
            memory_edge_count_final = int(len(memory.edges)) if memory is not None else 0
            memory_token_final = float(memory.estimate_tokens()) if memory is not None else 0.0
            memory_samples = int(memory_stats["memory_samples"])
            avg_retained_nodes = (
                float(memory_stats["retained_nodes_sum"]) / float(max(1, memory_samples))
                if memory_mode != "disabled"
                else 0.0
            )
            avg_retained_branching_points = (
                float(memory_stats["retained_branching_points_sum"]) / float(max(1, memory_samples))
                if memory_mode != "disabled"
                else 0.0
            )
            avg_recovery_useful_edges = (
                float(memory_stats["recovery_useful_edges_sum"]) / float(max(1, memory_samples))
                if memory_mode != "disabled"
                else 0.0
            )
            token_budget_safe = max(1, int(args.token_budget))
            skeleton_retention_rate = (
                float(memory_node_count_final) / float(max(1, int(memory_stats["memory_entry_count"])))
                if memory_mode != "disabled"
                else 0.0
            )
            retained_nodes_per_budget = (
                float(memory_node_count_final) / float(token_budget_safe)
                if memory_mode != "disabled"
                else 0.0
            )
            retained_branching_points_per_budget = (
                float(avg_retained_branching_points) / float(token_budget_safe)
                if memory_mode != "disabled"
                else 0.0
            )
            recovery_useful_edges_per_budget = (
                float(avg_recovery_useful_edges) / float(token_budget_safe)
                if memory_mode != "disabled"
                else 0.0
            )
            avg_retained_tokens_per_recovery_call = (
                float(memory_stats["max_retained_memory_tokens"]) / float(max(1, recovery_stats["recovery_attempts"]))
                if memory_mode != "disabled"
                else 0.0
            )
            episode_row = {
                "episode_idx": episode_idx,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "success": ep_metrics["success"],
                "spl": ep_metrics["spl"],
                "soft_spl": ep_metrics["soft_spl"],
                "distance_to_goal": ep_metrics["distance_to_goal"],
                "object_goal": episode.object_category,
                "seed": args.seed,
                "monitor_only": int(args.monitor_only),
                "recovery_mode": str(args.recovery_mode),
                "gate_mode": str(args.gate_mode),
                "gate_failure_window": int(args.gate_failure_window),
                "gate_progress_epsilon": float(args.gate_progress_epsilon),
                "gate_replan_event_count": int(gate_stats["replan_event_count"]),
                "trigger_count": int(gate_stats["trigger_count"]),
                "true_trigger_count": int(gate_stats["true_trigger_count"]),
                "false_trigger_count": int(gate_stats["false_trigger_count"]),
                "failure_opportunity_count": int(gate_stats["failure_opportunity_count"]),
                "trigger_overlap_count": int(gate_stats["trigger_overlap_count"]),
                "trigger_precision": float(trigger_precision),
                "trigger_recall": float(trigger_recall),
                "false_trigger_rate": float(false_trigger_rate),
                "recovery_attempts": int(recovery_stats["recovery_attempts"]),
                "recovery_success_count": int(recovery_stats["recovery_success_count"]),
                "recovery_failure_count": int(recovery_stats["recovery_failure_count"]),
                "selection_illegal_count": int(recovery_stats["selection_illegal_count"]),
                "execution_broken_count": int(recovery_stats["execution_broken_count"]),
                "closed_loop_fail_count": int(recovery_stats["closed_loop_fail_count"]),
                "memory_mode": str(memory_mode),
                "memory_cell_size": float(args.memory_cell_size),
                "memory_max_nodes": int(args.memory_max_nodes),
                "fifo_history_size": int(args.fifo_history_size),
                "token_budget": int(args.token_budget),
                "call_budget": -1 if effective_call_budget is None else int(effective_call_budget),
                "memory_compression_strategy": str(args.memory_compression_strategy),
                "memory_entry_count": int(memory_stats["memory_entry_count"]),
                "memory_samples": int(memory_stats["memory_samples"]),
                "memory_compress_events": int(memory_stats["memory_compress_events"]),
                "memory_node_count_final": int(memory_node_count_final),
                "memory_edge_count_final": int(memory_edge_count_final),
                "memory_node_peak": int(memory_stats["memory_node_peak"]),
                "memory_edge_peak": int(memory_stats["memory_edge_peak"]),
                "memory_token_final": float(memory_token_final),
                "max_retained_memory_tokens": float(memory_stats["max_retained_memory_tokens"]),
                "avg_retained_tokens_per_recovery_call": float(avg_retained_tokens_per_recovery_call),
                "avg_retained_nodes": float(avg_retained_nodes),
                "avg_retained_branching_points": float(avg_retained_branching_points),
                "avg_recovery_useful_edges": float(avg_recovery_useful_edges),
                "skeleton_retention_rate": float(skeleton_retention_rate),
                "retained_nodes_per_budget": float(retained_nodes_per_budget),
                "retained_branching_points_per_budget": float(retained_branching_points_per_budget),
                "recovery_useful_edges_per_budget": float(recovery_useful_edges_per_budget),
                "budget_skip_call_count": int(budget_stats["call_budget_skip_count"]),
                "budget_hard_cap_reached": int(budget_manager.hard_cap_reached()),
                "budget_last_skip_reason": str(budget_manager.last_skip_reason),
                "budget_remaining_ratio_final": float(budget_manager.remaining_budget_ratio()),
                "selection_illegal_rate": (
                    float(recovery_stats["selection_illegal_count"]) / float(max(1, recovery_stats["recovery_attempts"]))
                ),
                "execution_broken_rate": (
                    float(recovery_stats["execution_broken_count"]) / float(max(1, recovery_stats["recovery_attempts"]))
                ),
                "closed_loop_fail_rate": (
                    float(recovery_stats["closed_loop_fail_count"]) / float(max(1, recovery_stats["recovery_attempts"]))
                ),
            }
            append_metric_row(episode_row, csv_path)
            episode_rows_written.append(episode_row)
            seen_scenes.add(scene_id)
            if logger is not None:
                logger.log_episode(episode_row)
    finally:
        if logger is not None:
            logger.close()
        habitat_env.close()

    if episode_rows_written:
        sr = float(sum(float(r["success"]) for r in episode_rows_written)) / float(len(episode_rows_written))
        spl = float(sum(float(r["spl"]) for r in episode_rows_written)) / float(len(episode_rows_written))
        calls_per_episode = (
            float(sum(float(r["recovery_attempts"]) for r in episode_rows_written)) / float(len(episode_rows_written))
        )
        stalled_eps = sum(1 for r in episode_rows_written if float(r["failure_opportunity_count"]) > 0.0)
        success_stalled_eps = sum(
            1
            for r in episode_rows_written
            if float(r["failure_opportunity_count"]) > 0.0 and float(r["success"]) > 0.5
        )
        recovered_stalled_eps = sum(
            1
            for r in episode_rows_written
            if float(r["failure_opportunity_count"]) > 0.0 and float(r["recovery_success_count"]) > 0.0
        )
        recovered_eps = sum(1 for r in episode_rows_written if float(r["recovery_success_count"]) > 0.0)
        refail_after_recovery_eps = sum(
            1
            for r in episode_rows_written
            if float(r["recovery_success_count"]) > 0.0 and float(r["closed_loop_fail_count"]) > 0.0
        )
        summary_payload = {
            "config_path": str(HM3D_CONFIG_PATH),
            "overrides": [f"habitat.dataset.split={args.split}"],
            "episodes": int(args.eval_episodes),
            "max_steps": 0,
            "seed": int(args.seed),
            "policy_effective": str(args.planner_replan_policy),
            "gate_mode": str(args.gate_mode),
            "recovery_mode": str(args.recovery_mode),
            "ablation_flags": {
                "memory_mode": str(args.memory_mode),
                "monitor_only": int(args.monitor_only),
                "export_telemetry": int(args.export_telemetry),
            },
            "budget_definition": {
                "call_budget_scope": "per_episode",
                "call_budget": -1 if effective_call_budget is None else int(effective_call_budget),
                "call_budget_applies_to": "plugin_side_recovery_attempts",
                "latency_budget_scope": "per_episode",
                "latency_budget_ms": -1.0,
                "token_budget_scope": "per_step_memory_token_estimate",
                "token_budget": int(args.token_budget),
            },
            "metrics": {
                "SR": float(sr),
                "SPL": float(spl),
                "SR_stag": (
                    float(success_stalled_eps) / float(stalled_eps)
                    if stalled_eps > 0
                    else 0.0
                ),
                "stalled_episodes": int(stalled_eps),
                "success_stalled_episodes": int(success_stalled_eps),
                "RSR": (
                    float(recovered_stalled_eps) / float(max(1, stalled_eps))
                    if stalled_eps > 0
                    else 0.0
                ),
                "RFR": (
                    float(refail_after_recovery_eps) / float(max(1, recovered_eps))
                    if recovered_eps > 0
                    else 0.0
                ),
                "Calls_per_episode": float(calls_per_episode),
                "BudgetSkips_per_episode": (
                    float(sum(float(r["budget_skip_call_count"]) for r in episode_rows_written))
                    / float(len(episode_rows_written))
                ),
                "Latency_per_episode_ms": 0.0,
                "unique_scenes": int(len(seen_scenes)),
            },
            "artifacts": {
                "csv_path": str(csv_path),
                "telemetry_dir": str(output_dir / "telemetry") if telemetry_enabled else "",
            },
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"- summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
