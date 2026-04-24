"""Recoverable skeleton memory under bounded context budget."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ada_semnav.env_runner import Position3D


@dataclass
class SkeletonNode:
    id: str
    scene_id: str
    cell_x: int
    cell_z: int
    world_x: float
    world_y: float
    world_z: float
    visit_count: int = 0
    last_visit_step: int = -1
    semantic_hint: str = ""
    is_recovery_anchor: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkeletonEdge:
    id: str
    from_node: str
    to_node: str
    action: str
    visit_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    last_visit_step: int = -1
    semantic_hint: str = ""
    is_unexplored: bool = True
    is_low_yield: bool = False
    blocked: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class RecoverableSkeletonMemory:
    """Graph-like memory that retains high recovery-value structure only."""

    def __init__(
        self,
        *,
        cell_size: float = 0.75,
        max_nodes: int = 50,
        quantize_mode: str = "round",
        adaptive_cell_split: bool = False,
        cell_split_distance_ratio: float = 0.5,
        move_delta_alias_split: bool = False,
        move_delta_alias_split_ratio: float = 0.5,
        move_delta_alias_split_recovery_only: bool = False,
    ) -> None:
        self.cell_size = max(0.05, float(cell_size))
        # Allow low-memory stress tests (e.g., max_nodes=3) for budget ablations.
        self.max_nodes = max(1, int(max_nodes))
        mode = str(quantize_mode).strip().lower()
        self.quantize_mode = mode if mode in {"round", "floor"} else "round"
        self.adaptive_cell_split = bool(adaptive_cell_split)
        self.cell_split_distance_ratio = max(0.1, float(cell_split_distance_ratio))
        self.move_delta_alias_split = bool(move_delta_alias_split)
        self.move_delta_alias_split_ratio = max(0.1, float(move_delta_alias_split_ratio))
        self.move_delta_alias_split_recovery_only = bool(move_delta_alias_split_recovery_only)
        self.nodes: dict[str, SkeletonNode] = {}
        self.edges: dict[str, SkeletonEdge] = {}
        self._node_seq = 0
        self._edge_seq = 0
        self._node_by_cell: dict[tuple[str, int, int], list[str]] = {}
        self._edge_by_action: dict[tuple[str, str], str] = {}
        self.last_node_id: str | None = None
        self.move_delta_alias_split_count = 0
        self.move_delta_alias_split_base_count = 0
        self.move_delta_alias_split_recovery_count = 0

    def _new_node_id(self) -> str:
        nid = f"n{self._node_seq:05d}"
        self._node_seq += 1
        return nid

    def _new_edge_id(self) -> str:
        eid = f"e{self._edge_seq:06d}"
        self._edge_seq += 1
        return eid

    def _quantize(self, scene_id: str, position: Position3D) -> tuple[str, int, int]:
        if self.quantize_mode == "floor":
            qx = int(position.x / self.cell_size)
            qz = int(position.z / self.cell_size)
        else:
            qx = int(round(position.x / self.cell_size))
            qz = int(round(position.z / self.cell_size))
        return (
            scene_id,
            qx,
            qz,
        )

    @staticmethod
    def _distance_to_node(position: Position3D, node: SkeletonNode) -> float:
        return math.sqrt(
            (float(position.x) - float(node.world_x)) ** 2
            + (float(position.y) - float(node.world_y)) ** 2
            + (float(position.z) - float(node.world_z)) ** 2
        )

    def _create_node(
        self,
        *,
        key: tuple[str, int, int],
        scene_id: str,
        position: Position3D,
        step_idx: int,
        semantic_hint: str,
    ) -> str:
        nid = self._new_node_id()
        self.nodes[nid] = SkeletonNode(
            id=nid,
            scene_id=scene_id,
            cell_x=key[1],
            cell_z=key[2],
            world_x=float(position.x),
            world_y=float(position.y),
            world_z=float(position.z),
            visit_count=1,
            last_visit_step=int(step_idx),
            semantic_hint=semantic_hint,
        )
        self._node_by_cell.setdefault(key, []).append(nid)
        return nid

    def ensure_node(
        self,
        *,
        scene_id: str,
        position: Position3D,
        step_idx: int,
        semantic_hint: str = "",
        move_delta_alias_threshold: float | None = None,
    ) -> str:
        key = self._quantize(scene_id, position)
        candidate_ids = [nid for nid in self._node_by_cell.get(key, []) if nid in self.nodes]
        if not candidate_ids:
            return self._create_node(
                key=key,
                scene_id=scene_id,
                position=position,
                step_idx=step_idx,
                semantic_hint=semantic_hint,
            )

        best_nid: str | None = None
        best_dist = float("inf")
        for nid in candidate_ids:
            node = self.nodes[nid]
            dist = self._distance_to_node(position, node)
            if dist < best_dist:
                best_dist = dist
                best_nid = nid

        if best_nid is None:
            return self._create_node(
                key=key,
                scene_id=scene_id,
                position=position,
                step_idx=step_idx,
                semantic_hint=semantic_hint,
            )

        split_threshold = self.cell_size * self.cell_split_distance_ratio
        if self.adaptive_cell_split and best_dist > split_threshold:
            return self._create_node(
                key=key,
                scene_id=scene_id,
                position=position,
                step_idx=step_idx,
                semantic_hint=semantic_hint,
            )
        if move_delta_alias_threshold is not None and best_dist > float(move_delta_alias_threshold):
            self.move_delta_alias_split_count += 1
            return self._create_node(
                key=key,
                scene_id=scene_id,
                position=position,
                step_idx=step_idx,
                semantic_hint=semantic_hint,
            )

        node = self.nodes[best_nid]
        prev = max(0, int(node.visit_count))
        node.visit_count = prev + 1
        node.last_visit_step = int(step_idx)
        # Keep a smoothed centroid.
        node.world_x = (node.world_x * prev + float(position.x)) / float(prev + 1)
        node.world_y = (node.world_y * prev + float(position.y)) / float(prev + 1)
        node.world_z = (node.world_z * prev + float(position.z)) / float(prev + 1)
        if semantic_hint:
            node.semantic_hint = semantic_hint
        return best_nid

    def ensure_edge(self, *, from_node: str, action: str) -> str:
        k = (from_node, action)
        if k in self._edge_by_action:
            return self._edge_by_action[k]
        eid = self._new_edge_id()
        self.edges[eid] = SkeletonEdge(
            id=eid,
            from_node=from_node,
            to_node=from_node,
            action=action,
        )
        self._edge_by_action[k] = eid
        return eid

    def observe(
        self,
        *,
        scene_id: str,
        position: Position3D,
        step_idx: int,
        action: str | None,
        collision: bool,
        source: str = "base",
        semantic_hint: str = "",
        prev_position: Position3D | None = None,
        allow_move_delta_alias_split: bool | None = None,
    ) -> str:
        move_delta_alias_threshold: float | None = None
        split_count_before = int(self.move_delta_alias_split_count)
        move_delta_alias_enabled = (
            self.move_delta_alias_split
            if allow_move_delta_alias_split is None
            else bool(allow_move_delta_alias_split)
        )
        if (
            move_delta_alias_enabled
            and (not self.move_delta_alias_split_recovery_only or source == "recovery")
            and action == "move_forward"
            and not collision
            and prev_position is not None
        ):
            move_delta = math.sqrt(
                (float(position.x) - float(prev_position.x)) ** 2
                + (float(position.y) - float(prev_position.y)) ** 2
                + (float(position.z) - float(prev_position.z)) ** 2
            )
            if move_delta > 1e-4:
                move_delta_alias_threshold = move_delta * self.move_delta_alias_split_ratio

        node_id = self.ensure_node(
            scene_id=scene_id,
            position=position,
            step_idx=step_idx,
            semantic_hint=semantic_hint,
            move_delta_alias_threshold=move_delta_alias_threshold,
        )
        split_delta = max(0, int(self.move_delta_alias_split_count) - split_count_before)
        if split_delta > 0:
            if source == "recovery":
                self.move_delta_alias_split_recovery_count += split_delta
            else:
                self.move_delta_alias_split_base_count += split_delta
        if source == "recovery" and node_id in self.nodes:
            self.nodes[node_id].is_recovery_anchor = True

        if self.last_node_id is not None and action is not None and self.last_node_id in self.nodes:
            eid = self.ensure_edge(from_node=self.last_node_id, action=action)
            edge = self.edges[eid]
            edge.to_node = self.last_node_id if collision else node_id
            edge.visit_count += 1
            edge.last_visit_step = int(step_idx)
            edge.is_unexplored = False
            if semantic_hint:
                edge.semantic_hint = semantic_hint
            if collision:
                edge.fail_count += 1
            else:
                edge.success_count += 1
            # Mark low-yield self-loops (typically repeated turn-in-place behavior)
            # so recovery can avoid wasting retries on non-progress edges.
            self_loop_count = int(edge.meta.get("self_loop_count", 0))
            if not collision and edge.to_node == edge.from_node:
                self_loop_count += 1
            elif not collision:
                self_loop_count = 0
            edge.meta["self_loop_count"] = self_loop_count
            edge.is_low_yield = bool(
                (edge.fail_count >= 2 and edge.success_count == 0) or self_loop_count >= 2
            )
            edge.blocked = bool(edge.fail_count >= 2 and edge.success_count == 0)

        self.last_node_id = node_id
        return node_id

    def node_degree(self, node_id: str) -> int:
        out_deg = sum(1 for e in self.edges.values() if e.from_node == node_id)
        in_deg = sum(1 for e in self.edges.values() if e.to_node == node_id)
        return out_deg + in_deg

    def backtrack_target(self, recent_nodes: list[str], lookback: int = 5, skip_current: bool = False) -> str | None:
        if not recent_nodes:
            return self.last_node_id
        window = recent_nodes[-max(1, int(lookback)) :]
        search_window = window[:-1] if skip_current and len(window) > 1 else window
        for nid in reversed(search_window):
            if nid in self.nodes and self.node_degree(nid) > 2:
                return nid
        for nid in reversed(search_window):
            if nid in self.nodes:
                return nid
        return self.last_node_id

    def mark_recovery_anchor(self, node_id: str | None) -> None:
        if node_id is None or node_id not in self.nodes:
            return
        self.nodes[node_id].is_recovery_anchor = True

    def candidate_edges(self, current_node: str, actions: list[str]) -> list[SkeletonEdge]:
        return self.query_edges(current_node=current_node, actions=actions, create_missing=True)

    def query_edges(
        self,
        *,
        current_node: str,
        actions: list[str],
        create_missing: bool = True,
    ) -> list[SkeletonEdge]:
        if current_node not in self.nodes:
            return []
        result: list[SkeletonEdge] = []
        for action in actions:
            eid = self._edge_by_action.get((current_node, action))
            if eid is None:
                if not create_missing:
                    continue
                eid = self.ensure_edge(from_node=current_node, action=action)
            result.append(self.edges[eid])
        result.sort(key=lambda e: (1 if e.blocked else 0, e.fail_count, e.visit_count, e.id))
        return result

    def _prune_removed_nodes(self, removed: list[str]) -> list[str]:
        if not removed:
            return []
        remove_set = set(removed)
        stale_edges = [eid for eid, e in self.edges.items() if e.from_node in remove_set or e.to_node in remove_set]
        for eid in stale_edges:
            edge = self.edges.pop(eid, None)
            if edge is None:
                continue
            key = (edge.from_node, edge.action)
            if self._edge_by_action.get(key) == eid:
                self._edge_by_action.pop(key, None)

        self._node_by_cell = {}
        for n in self.nodes.values():
            key = (n.scene_id, n.cell_x, n.cell_z)
            self._node_by_cell.setdefault(key, []).append(n.id)
        if self.last_node_id in remove_set:
            self.last_node_id = None
        return removed

    def compress(self, *, strategy: str = "recoverability") -> list[str]:
        if len(self.nodes) <= self.max_nodes:
            return []
        strategy = str(strategy).strip().lower() or "recoverability"
        if strategy == "normal":
            # Normal compression baseline: purely recency/frequency based trimming,
            # intentionally agnostic to recovery anchors and high-failure edges.
            candidates = sorted(self.nodes.values(), key=lambda n: (n.visit_count, n.last_visit_step, n.id))
            removed: list[str] = []
            while len(self.nodes) > self.max_nodes and candidates:
                n = candidates.pop(0)
                if n.id == self.last_node_id:
                    continue
                if n.id not in self.nodes:
                    continue
                removed.append(n.id)
                self.nodes.pop(n.id, None)
            return self._prune_removed_nodes(removed)

        keep: set[str] = set()
        if self.last_node_id is not None:
            keep.add(self.last_node_id)
        keep.update(nid for nid, n in self.nodes.items() if n.is_recovery_anchor)
        for edge in self.edges.values():
            if edge.fail_count >= 2 or edge.success_count >= 3:
                keep.add(edge.from_node)
                keep.add(edge.to_node)

        # Enforce a hard budget for the keep-set itself.
        if len(keep) > self.max_nodes:
            support: dict[str, int] = {}
            for edge in self.edges.values():
                w = 0
                if edge.fail_count >= 2:
                    w = 3
                elif edge.success_count >= 3:
                    w = 1
                if w <= 0:
                    continue
                support[edge.from_node] = support.get(edge.from_node, 0) + w
                support[edge.to_node] = support.get(edge.to_node, 0) + w

            ranked_keep = sorted(
                keep,
                key=lambda nid: (
                    1 if nid == self.last_node_id else 0,
                    support.get(nid, 0),
                    1 if self.nodes.get(nid) and self.nodes[nid].is_recovery_anchor else 0,
                    self.nodes[nid].last_visit_step if nid in self.nodes else -1,
                    nid,
                ),
                reverse=True,
            )
            keep = set(ranked_keep[: self.max_nodes])

        # Remove low-value nodes first: old + low visit + not in keep.
        candidates = [n for n in self.nodes.values() if n.id not in keep]
        candidates.sort(key=lambda n: (n.visit_count, n.last_visit_step, n.id))
        removed: list[str] = []
        while len(self.nodes) > self.max_nodes and candidates:
            n = candidates.pop(0)
            if n.id not in self.nodes:
                continue
            removed.append(n.id)
            self.nodes.pop(n.id, None)
        return self._prune_removed_nodes(removed)

    def estimate_tokens(self) -> int:
        # Budget accounting follows a compact recovery-query projection
        # instead of raw dataclass dumps. This mirrors the information
        # needed by constrained recovery (branch/edge legality/failure).
        nodes = sorted(self.nodes.values(), key=lambda n: n.id)
        edges = sorted(self.edges.values(), key=lambda e: e.id)
        payload = {
            "cur": self.last_node_id or "",
            "nodes": [
                {
                    "i": n.id,
                    "c": [int(n.cell_x), int(n.cell_z)],
                    "v": int(n.visit_count),
                    "t": int(n.last_visit_step),
                    "a": int(bool(n.is_recovery_anchor)),
                    "h": (n.semantic_hint or "")[:32],
                }
                for n in nodes
            ],
            "edges": [
                {
                    "i": e.id,
                    "f": e.from_node,
                    "t": e.to_node,
                    "a": e.action,
                    "v": int(e.visit_count),
                    "s": int(e.success_count),
                    "x": int(e.fail_count),
                    "b": int(bool(e.blocked)),
                    "l": int(bool(e.is_low_yield)),
                    "u": int(bool(e.is_unexplored)),
                    "h": (e.semantic_hint or "")[:32],
                }
                for e in edges
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return max(1, len(text) // 4)

    def node_signals_at(self, scene_id: str, position: Position3D) -> dict[str, float]:
        """Return composite ranking signals for the memory cell nearest to *position*.

        Returns a dict with keys:
          visit_count, degree, unexplored_edges, blocked_edges
        All values default to 0 when no node is found.
        """
        key = self._quantize(scene_id, position)
        candidate_ids = [nid for nid in self._node_by_cell.get(key, []) if nid in self.nodes]
        if not candidate_ids:
            return {"visit_count": 0.0, "degree": 0.0, "unexplored_edges": 0.0, "blocked_edges": 0.0}
        best_nid = min(candidate_ids, key=lambda nid: self._distance_to_node(position, self.nodes[nid]))
        node = self.nodes[best_nid]
        degree = 0
        unexplored = 0
        blocked = 0
        for e in self.edges.values():
            if e.from_node == best_nid:
                degree += 1
                if e.is_unexplored:
                    unexplored += 1
                if e.blocked:
                    blocked += 1
            elif e.to_node == best_nid:
                degree += 1
        return {
            "visit_count": float(node.visit_count),
            "degree": float(degree),
            "unexplored_edges": float(unexplored),
            "blocked_edges": float(blocked),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }
