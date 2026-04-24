"""Closed-loop edge-constrained recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ada_semnav.llm_interface import LLMInterface
from ada_semnav.skeleton_memory import RecoverableSkeletonMemory, SkeletonEdge


def _normalize_semantic_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


@dataclass
class RecoveryResult:
    success: bool
    selected_edge_id: str | None
    selected_action: str | None
    retries: int
    reason: str
    blocked_edge_ids: list[str]
    tried_edge_ids: list[str]
    backtrack_from_node: str | None
    backtrack_to_node: str | None
    llm_used: bool
    backtrack_reached_node: str | None = None
    backtrack_anchor_reached: bool = False
    backtrack_prefix_steps: int = 0
    planning_node: str | None = None
    candidate_count_initial: int = 0
    candidate_count_final: int = 0
    candidate_query_mode: str = "current_node"
    candidate_source_nodes: list[str] = field(default_factory=list)
    candidate_source_kinds: list[str] = field(default_factory=list)
    retrieved_subgraph_node_count: int = 0
    retrieved_subgraph_edge_count: int = 0
    retrieved_branch_types: list[str] = field(default_factory=list)
    selected_edge_from_node: str | None = None
    selected_edge_source_kind: str | None = None
    selected_edge_branch_type: str | None = None
    semantic_sources_considered_count: int = 0
    semantic_sources_retained_count: int = 0
    semantic_sources_rejected_low_grounding_count: int = 0
    semantic_sources_retained_failsafe_count: int = 0
    semantic_failsafe_local_repeat_window_active: int = 0
    semantic_failsafe_local_repeat_window_unique_nodes: int = 0
    semantic_failsafe_local_repeat_window_current_hits: int = 0
    grounded_only_nonlocal_admission_enabled: int = 0
    grounded_only_nonlocal_filtered_count: int = 0
    non_grounded_nonlocal_bonus_scale: float = 1.0
    non_grounded_nonlocal_scaled_edge_count: int = 0
    semantic_exit_fail_support_penalty: float = 0.0
    semantic_sources_penalized_fail_support_count: int = 0
    semantic_candidate_pool_size: int = 0
    semantic_candidate_pool_competing_count: int = 0
    semantic_top1_source_node: str | None = None
    semantic_top1_grounding_count: int = 0
    semantic_top1_node_semantic_match: int = 0
    semantic_top1_usable_count: int = 0
    semantic_top1_unexplored_usable_count: int = 0
    semantic_top1_unexplored_usable_ratio: float = 0.0
    semantic_top1_fail_support: int = 0
    semantic_top1_value_proxy: float = 0.0
    semantic_best_novelty_source_node: str | None = None
    semantic_best_novelty_grounding_count: int = 0
    semantic_best_novelty_node_semantic_match: int = 0
    semantic_best_novelty_usable_count: int = 0
    semantic_best_novelty_unexplored_usable_count: int = 0
    semantic_best_novelty_unexplored_usable_ratio: float = 0.0
    semantic_best_novelty_fail_support: int = 0
    semantic_best_novelty_value_proxy: float = 0.0
    semantic_top1_is_best_novelty: int = 0
    semantic_top1_best_novelty_ratio_gap: float = 0.0
    semantic_pool_repair_attempted: int = 0
    semantic_pool_repair_pool_size: int = 0
    semantic_pool_repair_semantic_exit_present: int = 0
    semantic_pool_repair_debug: str | None = None


class EdgeConstrainedRecovery:
    """Deterministic backtrack + constrained edge selection."""

    def __init__(
        self,
        *,
        llm: LLMInterface | None = None,
        backend: str = "heuristic",
        candidate_query_mode: str = "current_node",
        candidate_query_max_sources: int = 3,
        semantic_exit_min_edge_hint_matches: int = 0,
        semantic_exit_min_edge_hint_ratio: float = 0.0,
        semantic_exit_require_edge_hint_for_node_fallback: bool = False,
        semantic_exit_require_node_hint_match: bool = False,
        semantic_exit_enable_node_consistency_failsafe: bool = False,
        semantic_exit_failsafe_prefer_margin: float = -1.0,
        semantic_exit_failsafe_require_repeat_window: int = 0,
        semantic_exit_failsafe_repeat_unique_max: int = 0,
        semantic_exit_failsafe_repeat_min_current_hits: int = 0,
        grounded_only_nonlocal_admission: bool = False,
        non_grounded_nonlocal_bonus_scale: float = 1.0,
        semantic_exit_fail_support_penalty: float = 0.0,
        a_unexplored: float = 2.0,
        b_low_visit: float = 1.0,
        c_semantic: float = 2.0,
        d_failed: float = 3.0,
        e_retry: float = 2.0,
        f_low_yield: float = 1.5,
        g_forward_bonus: float = 0.75,
    ) -> None:
        self.llm = llm
        self.backend = backend
        mode = str(candidate_query_mode).strip().lower()
        self.candidate_query_mode = (
            mode
            if mode
            in {
                "current_node",
                "anchor_frontier",
                "skeleton_frontier",
                "query_aware_subgraph",
                "query_aware_edge_pool",
            }
            else "current_node"
        )
        self.candidate_query_max_sources = max(1, int(candidate_query_max_sources))
        self.semantic_exit_min_edge_hint_matches = max(0, int(semantic_exit_min_edge_hint_matches))
        self.semantic_exit_min_edge_hint_ratio = max(0.0, float(semantic_exit_min_edge_hint_ratio))
        self.semantic_exit_require_edge_hint_for_node_fallback = bool(
            semantic_exit_require_edge_hint_for_node_fallback
        )
        self.semantic_exit_require_node_hint_match = bool(semantic_exit_require_node_hint_match)
        self.semantic_exit_enable_node_consistency_failsafe = bool(
            semantic_exit_enable_node_consistency_failsafe
        )
        self.semantic_exit_failsafe_prefer_margin = float(semantic_exit_failsafe_prefer_margin)
        self.semantic_exit_failsafe_require_repeat_window = max(0, int(semantic_exit_failsafe_require_repeat_window))
        self.semantic_exit_failsafe_repeat_unique_max = max(0, int(semantic_exit_failsafe_repeat_unique_max))
        self.semantic_exit_failsafe_repeat_min_current_hits = max(
            0, int(semantic_exit_failsafe_repeat_min_current_hits)
        )
        self.grounded_only_nonlocal_admission = bool(grounded_only_nonlocal_admission)
        self.non_grounded_nonlocal_bonus_scale = max(0.0, min(1.0, float(non_grounded_nonlocal_bonus_scale)))
        self.semantic_exit_fail_support_penalty = max(0.0, float(semantic_exit_fail_support_penalty))
        self.a_unexplored = float(a_unexplored)
        self.b_low_visit = float(b_low_visit)
        self.c_semantic = float(c_semantic)
        self.d_failed = float(d_failed)
        self.e_retry = float(e_retry)
        self.f_low_yield = float(f_low_yield)
        self.g_forward_bonus = float(g_forward_bonus)

    def _failsafe_local_repeat_window_stats(
        self,
        *,
        recent_nodes: list[str],
        current_node: str,
    ) -> tuple[int, int, int]:
        """Return (active, unique_nodes, current_hits) for optional fail-safe gating."""
        window = int(self.semantic_exit_failsafe_require_repeat_window)
        if window <= 0:
            # Default behavior keeps fail-safe A semantics unchanged.
            return 1, 0, 0

        tail = recent_nodes[-window:] if len(recent_nodes) >= window else []
        if not tail:
            return 0, 0, 0

        unique_nodes = len(set(tail))
        current_hits = sum(1 for node_id in tail if node_id == current_node)
        unique_ok = (
            self.semantic_exit_failsafe_repeat_unique_max <= 0
            or unique_nodes <= self.semantic_exit_failsafe_repeat_unique_max
        )
        current_ok = (
            self.semantic_exit_failsafe_repeat_min_current_hits <= 0
            or current_hits >= self.semantic_exit_failsafe_repeat_min_current_hits
        )
        active = int(unique_ok and current_ok)
        return active, unique_nodes, current_hits

    def _score_edge(
        self,
        edge: SkeletonEdge,
        target_hint: str,
        action_support: dict[str, float] | None = None,
        edge_support: dict[str, float] | None = None,
    ) -> float:
        target_hint_norm = _normalize_semantic_text(target_hint)
        unexplored = 1.0 if edge.is_unexplored else 0.0
        low_visit = 1.0 if edge.visit_count <= 1 else 0.0
        semantic_match = 0.0
        if target_hint_norm and edge.semantic_hint:
            semantic_match = 1.0 if target_hint_norm in _normalize_semantic_text(edge.semantic_hint) else 0.0
        failed_before = 1.0 if edge.fail_count > 0 else 0.0
        recent_retry = 1.0 if edge.blocked else 0.0
        low_yield = 1.0 if edge.is_low_yield else 0.0
        forward_bonus = 1.0 if edge.action == "move_forward" else 0.0
        base_score = (
            self.a_unexplored * unexplored
            + self.b_low_visit * low_visit
            + self.c_semantic * semantic_match
            - self.d_failed * failed_before
            - self.e_retry * recent_retry
            - self.f_low_yield * low_yield
            + self.g_forward_bonus * forward_bonus
        )
        support_bonus = 0.0 if action_support is None else float(action_support.get(edge.action, 0.0))
        edge_bonus = 0.0 if edge_support is None else float(edge_support.get(edge.id, 0.0))
        return base_score + support_bonus + edge_bonus

    def _heuristic_select(
        self,
        candidates: list[SkeletonEdge],
        target_hint: str,
        action_support: dict[str, float] | None = None,
        edge_support: dict[str, float] | None = None,
    ) -> SkeletonEdge:
        if not candidates:
            raise ValueError("No candidate edges.")
        return max(
            candidates,
            key=lambda e: (
                self._score_edge(e, target_hint, action_support, edge_support),
                0.0 if edge_support is None else float(edge_support.get(e.id, 0.0)),
                0.0 if action_support is None else float(action_support.get(e.action, 0.0)),
                -e.visit_count,
                e.id,
            ),
        )

    def _llm_select(
        self,
        *,
        backtrack_node: str,
        candidates: list[SkeletonEdge],
        target_hint: str,
    ) -> tuple[SkeletonEdge | None, bool]:
        if self.llm is None or self.backend != "openai":
            return None, False
        lines = []
        for e in candidates:
            lines.append(
                f"- edge_id={e.id}; action={e.action}; unexplored={int(e.is_unexplored)}; "
                f"low_visit={int(e.visit_count <= 1)}; failed_before={int(e.fail_count > 0)}; "
                f"blocked={int(e.blocked)}; semantic_hint={e.semantic_hint or 'none'}"
            )
        prompt = (
            "Select exactly one edge_id from LEGAL_EDGES. Return JSON only.\n"
            'Format: {"edge_id":"..."}\n'
            f"TARGET_HINT: {target_hint or 'none'}\n"
            f"BACKTRACK_NODE: {backtrack_node}\n"
            "LEGAL_EDGES:\n"
            + "\n".join(lines)
            )
        payload, _ = self.llm.call_json(purpose="recovery_edge_selection", prompt=prompt, max_tokens=64)
        if payload is None:
            return None, True
        edge_id = str(payload.get("edge_id", ""))
        for e in candidates:
            if e.id == edge_id:
                return e, True
        return None, True

    def _candidate_query_sources(
        self,
        *,
        memory: RecoverableSkeletonMemory,
        current_node: str | None,
        backtrack_to_node: str | None,
        recent_nodes: list[str],
        target_hint: str,
    ) -> list[tuple[str, str]]:
        current_valid = current_node is not None and current_node in memory.nodes
        anchor_valid = backtrack_to_node is not None and backtrack_to_node in memory.nodes
        if self.candidate_query_mode == "skeleton_frontier":
            frontier: list[tuple[tuple[float, ...], str, str]] = []
            target_hint_norm = _normalize_semantic_text(target_hint)
            for node in memory.nodes.values():
                if node.id == current_node:
                    continue
                incident_fail_support = 0
                blocked_support = 0
                degree = memory.node_degree(node.id)
                for edge in memory.edges.values():
                    if edge.from_node != node.id and edge.to_node != node.id:
                        continue
                    if edge.fail_count > 0:
                        incident_fail_support += 1
                    if edge.blocked:
                        blocked_support += 1
                semantic_match = (
                    1.0 if target_hint_norm and target_hint_norm in _normalize_semantic_text(node.semantic_hint) else 0.0
                )
                if not node.is_recovery_anchor and degree <= 2 and incident_fail_support <= 0 and semantic_match <= 0.0:
                    continue
                source_kind = "anchor_frontier" if backtrack_to_node == node.id else "skeleton_frontier"
                score = (
                    1.0 if source_kind == "anchor_frontier" else 0.0,
                    semantic_match,
                    1.0 if node.is_recovery_anchor else 0.0,
                    float(blocked_support),
                    float(incident_fail_support),
                    float(degree),
                    float(node.last_visit_step),
                    float(node.visit_count),
                )
                frontier.append((score, node.id, source_kind))
            frontier.sort(reverse=True)
            query_sources = [(node_id, source_kind) for _score, node_id, source_kind in frontier[: self.candidate_query_max_sources]]
            if query_sources:
                return query_sources
            if current_valid and current_node is not None:
                return [(current_node, "current_node")]
            return []
        if self.candidate_query_mode in {"query_aware_subgraph", "query_aware_edge_pool"}:
            return []
        if self.candidate_query_mode == "anchor_frontier":
            anchor_node = backtrack_to_node if anchor_valid else None
            if anchor_node == current_node:
                recovery_anchors = [
                    node
                    for node in memory.nodes.values()
                    if node.id != current_node and bool(node.is_recovery_anchor)
                ]
                if recovery_anchors:
                    recovery_anchors.sort(key=lambda node: (node.last_visit_step, node.visit_count, node.id), reverse=True)
                    anchor_node = recovery_anchors[0].id
                else:
                    fallback_anchor = memory.backtrack_target(
                        recent_nodes=recent_nodes,
                        lookback=max(8, len(recent_nodes)),
                        skip_current=True,
                    )
                    if fallback_anchor is not None and fallback_anchor in memory.nodes and fallback_anchor != current_node:
                        anchor_node = fallback_anchor
            if anchor_node is not None:
                source_kind = "anchor_frontier" if anchor_node != current_node else "current_node"
                return [(anchor_node, source_kind)]
            if current_valid and current_node is not None:
                return [(current_node, "current_node")]
            return []
        if current_valid and current_node is not None:
            return [(current_node, "current_node")]
        if anchor_valid and backtrack_to_node is not None:
            return [(backtrack_to_node, "anchor_frontier")]
        return []

    def _resolve_anchor_query_node(
        self,
        *,
        memory: RecoverableSkeletonMemory,
        current_node: str | None,
        backtrack_to_node: str | None,
        recent_nodes: list[str],
    ) -> str | None:
        anchor_valid = backtrack_to_node is not None and backtrack_to_node in memory.nodes
        anchor_node = backtrack_to_node if anchor_valid else None
        if anchor_node == current_node:
            recovery_anchors = [
                node for node in memory.nodes.values() if node.id != current_node and bool(node.is_recovery_anchor)
            ]
            if recovery_anchors:
                recovery_anchors.sort(key=lambda node: (node.last_visit_step, node.visit_count, node.id), reverse=True)
                anchor_node = recovery_anchors[0].id
            else:
                fallback_anchor = memory.backtrack_target(
                    recent_nodes=recent_nodes,
                    lookback=max(8, len(recent_nodes)),
                    skip_current=True,
                )
                if fallback_anchor is not None and fallback_anchor in memory.nodes and fallback_anchor != current_node:
                    anchor_node = fallback_anchor
                else:
                    for node_id in reversed(recent_nodes[:-1]):
                        if node_id in memory.nodes and node_id != current_node:
                            anchor_node = node_id
                            break
        return anchor_node

    def _summarize_semantic_candidate_pool(
        self,
        *,
        memory: RecoverableSkeletonMemory,
        current_node: str,
        target_hint: str,
        actions: list[str],
    ) -> dict[str, Any]:
        target_hint_norm = _normalize_semantic_text(target_hint)
        summary: dict[str, Any] = {
            "semantic_candidate_pool_size": 0,
            "semantic_candidate_pool_competing_count": 0,
            "semantic_top1_source_node": "",
            "semantic_top1_grounding_count": 0,
            "semantic_top1_node_semantic_match": 0,
            "semantic_top1_usable_count": 0,
            "semantic_top1_unexplored_usable_count": 0,
            "semantic_top1_unexplored_usable_ratio": 0.0,
            "semantic_top1_fail_support": 0,
            "semantic_top1_value_proxy": 0.0,
            "semantic_best_novelty_source_node": "",
            "semantic_best_novelty_grounding_count": 0,
            "semantic_best_novelty_node_semantic_match": 0,
            "semantic_best_novelty_usable_count": 0,
            "semantic_best_novelty_unexplored_usable_count": 0,
            "semantic_best_novelty_unexplored_usable_ratio": 0.0,
            "semantic_best_novelty_fail_support": 0,
            "semantic_best_novelty_value_proxy": 0.0,
            "semantic_top1_is_best_novelty": 0,
            "semantic_top1_best_novelty_ratio_gap": 0.0,
            "semantic_pool_repair_attempted": 0,
            "semantic_pool_repair_pool_size": 0,
            "semantic_pool_repair_semantic_exit_present": 0,
            "semantic_pool_repair_debug": "",
        }
        if not target_hint_norm:
            return summary

        candidate_profiles: list[dict[str, Any]] = []
        for node in memory.nodes.values():
            if node.id == current_node:
                continue
            observed_edges = memory.query_edges(current_node=node.id, actions=actions, create_missing=False)
            if not observed_edges:
                continue
            usable_alternatives = [
                edge
                for edge in observed_edges
                if not edge.blocked and not edge.is_low_yield and edge.fail_count <= 0
            ]
            semantic_edge_match_count = sum(
                1 for edge in observed_edges if target_hint_norm in _normalize_semantic_text(edge.semantic_hint)
            )
            node_semantic_match = bool(target_hint_norm in _normalize_semantic_text(node.semantic_hint))
            if not (semantic_edge_match_count > 0 or node_semantic_match):
                continue
            semantic_edge_match_ratio = float(semantic_edge_match_count) / float(max(1, len(observed_edges)))
            edge_grounding_ok = bool(
                semantic_edge_match_count >= self.semantic_exit_min_edge_hint_matches
                and semantic_edge_match_ratio >= self.semantic_exit_min_edge_hint_ratio
            )
            if self.semantic_exit_require_node_hint_match:
                edge_grounding_ok = bool(edge_grounding_ok and node_semantic_match)
            node_fallback_ok = bool(
                not self.semantic_exit_require_edge_hint_for_node_fallback
                and self.semantic_exit_min_edge_hint_matches <= 0
                and self.semantic_exit_min_edge_hint_ratio <= 0.0
                and node_semantic_match
            )
            if not (edge_grounding_ok or node_fallback_ok):
                continue
            fail_support = sum(
                1
                for edge in memory.edges.values()
                if (edge.from_node == node.id or edge.to_node == node.id)
                and (edge.blocked or edge.is_low_yield or edge.fail_count > 0)
            )
            unexplored_usable_count = sum(1 for edge in usable_alternatives if edge.is_unexplored)
            unexplored_usable_ratio = float(unexplored_usable_count) / float(max(1, len(usable_alternatives)))
            semantic_value_proxy = float(len(usable_alternatives)) - (
                self.semantic_exit_fail_support_penalty * float(fail_support)
            )
            candidate_profiles.append(
                {
                    "node_id": node.id,
                    "semantic_edge_match_count": int(semantic_edge_match_count),
                    "node_semantic_match": int(node_semantic_match),
                    "usable_count": int(len(usable_alternatives)),
                    "unexplored_usable_count": int(unexplored_usable_count),
                    "unexplored_usable_ratio": float(unexplored_usable_ratio),
                    "fail_support": int(fail_support),
                    "semantic_value_proxy": float(semantic_value_proxy),
                    "node_degree": int(memory.node_degree(node.id)),
                    "last_visit_step": int(node.last_visit_step),
                }
            )

        if not candidate_profiles:
            return summary

        candidate_profiles.sort(
            key=lambda profile: (
                float(profile["semantic_edge_match_count"]),
                float(profile["node_semantic_match"]),
                float(profile["semantic_value_proxy"]),
                float(profile["node_degree"]),
                float(profile["last_visit_step"]),
                str(profile["node_id"]),
            ),
            reverse=True,
        )
        top1_profile = candidate_profiles[0]
        best_novelty_profile = max(
            candidate_profiles,
            key=lambda profile: (
                float(profile["unexplored_usable_ratio"]),
                int(profile["unexplored_usable_count"]),
                int(profile["usable_count"]),
                int(profile["semantic_edge_match_count"]),
                int(profile["node_semantic_match"]),
                int(profile["node_degree"]),
                int(profile["last_visit_step"]),
                str(profile["node_id"]),
            ),
        )
        summary.update(
            {
                "semantic_candidate_pool_size": int(len(candidate_profiles)),
                "semantic_candidate_pool_competing_count": int(max(0, len(candidate_profiles) - 1)),
                "semantic_top1_source_node": str(top1_profile["node_id"]),
                "semantic_top1_grounding_count": int(top1_profile["semantic_edge_match_count"]),
                "semantic_top1_node_semantic_match": int(top1_profile["node_semantic_match"]),
                "semantic_top1_usable_count": int(top1_profile["usable_count"]),
                "semantic_top1_unexplored_usable_count": int(top1_profile["unexplored_usable_count"]),
                "semantic_top1_unexplored_usable_ratio": float(top1_profile["unexplored_usable_ratio"]),
                "semantic_top1_fail_support": int(top1_profile["fail_support"]),
                "semantic_top1_value_proxy": float(top1_profile["semantic_value_proxy"]),
                "semantic_best_novelty_source_node": str(best_novelty_profile["node_id"]),
                "semantic_best_novelty_grounding_count": int(best_novelty_profile["semantic_edge_match_count"]),
                "semantic_best_novelty_node_semantic_match": int(best_novelty_profile["node_semantic_match"]),
                "semantic_best_novelty_usable_count": int(best_novelty_profile["usable_count"]),
                "semantic_best_novelty_unexplored_usable_count": int(
                    best_novelty_profile["unexplored_usable_count"]
                ),
                "semantic_best_novelty_unexplored_usable_ratio": float(
                    best_novelty_profile["unexplored_usable_ratio"]
                ),
                "semantic_best_novelty_fail_support": int(best_novelty_profile["fail_support"]),
                "semantic_best_novelty_value_proxy": float(best_novelty_profile["semantic_value_proxy"]),
                "semantic_top1_is_best_novelty": int(
                    str(top1_profile["node_id"]) == str(best_novelty_profile["node_id"])
                ),
                "semantic_top1_best_novelty_ratio_gap": float(
                    best_novelty_profile["unexplored_usable_ratio"]
                )
                - float(top1_profile["unexplored_usable_ratio"]),
            }
        )
        return summary

    def _build_query_aware_subgraph(
        self,
        *,
        memory: RecoverableSkeletonMemory,
        current_node: str,
        backtrack_to_node: str | None,
        recent_nodes: list[str],
        target_hint: str,
        actions: list[str],
    ) -> tuple[
        list[tuple[str, str]],
        dict[str, float],
        dict[str, str],
        dict[str, float],
        dict[str, str],
        int,
        list[str],
        dict[str, int],
    ]:
        target_hint_norm = _normalize_semantic_text(target_hint)
        sources: list[tuple[str, str]] = []
        seen_nodes: set[str] = set()
        branch_types: list[str] = []
        semantic_source_stats: dict[str, Any] = {
            "considered": 0,
            "retained": 0,
            "rejected_low_grounding": 0,
            "retained_failsafe": 0,
            "failsafe_local_repeat_window_active": 0,
            "failsafe_local_repeat_window_unique_nodes": 0,
            "failsafe_local_repeat_window_current_hits": 0,
            "grounded_only_nonlocal_admission_enabled": int(self.grounded_only_nonlocal_admission),
            "grounded_only_nonlocal_filtered": 0,
            "non_grounded_nonlocal_scaled_edges": 0,
            "semantic_fail_support_penalized": 0,
            "semantic_candidate_pool_size": 0,
            "semantic_candidate_pool_competing_count": 0,
            "semantic_top1_source_node": "",
            "semantic_top1_grounding_count": 0,
            "semantic_top1_node_semantic_match": 0,
            "semantic_top1_usable_count": 0,
            "semantic_top1_unexplored_usable_count": 0,
            "semantic_top1_unexplored_usable_ratio": 0.0,
            "semantic_top1_fail_support": 0,
            "semantic_top1_value_proxy": 0.0,
            "semantic_best_novelty_source_node": "",
            "semantic_best_novelty_grounding_count": 0,
            "semantic_best_novelty_node_semantic_match": 0,
            "semantic_best_novelty_usable_count": 0,
            "semantic_best_novelty_unexplored_usable_count": 0,
            "semantic_best_novelty_unexplored_usable_ratio": 0.0,
            "semantic_best_novelty_fail_support": 0,
            "semantic_best_novelty_value_proxy": 0.0,
            "semantic_top1_is_best_novelty": 0,
            "semantic_top1_best_novelty_ratio_gap": 0.0,
        }
        (
            failsafe_local_repeat_window_active,
            failsafe_local_repeat_window_unique_nodes,
            failsafe_local_repeat_window_current_hits,
        ) = self._failsafe_local_repeat_window_stats(recent_nodes=recent_nodes, current_node=current_node)
        semantic_source_stats["failsafe_local_repeat_window_active"] = failsafe_local_repeat_window_active
        semantic_source_stats["failsafe_local_repeat_window_unique_nodes"] = (
            failsafe_local_repeat_window_unique_nodes
        )
        semantic_source_stats["failsafe_local_repeat_window_current_hits"] = (
            failsafe_local_repeat_window_current_hits
        )

        def add_source(node_id: str | None, branch_type: str) -> None:
            if node_id is None or node_id == current_node or node_id not in memory.nodes or node_id in seen_nodes:
                return
            seen_nodes.add(node_id)
            sources.append((node_id, branch_type))
            branch_types.append(branch_type)

        anchor_node = self._resolve_anchor_query_node(
            memory=memory,
            current_node=current_node,
            backtrack_to_node=backtrack_to_node,
            recent_nodes=recent_nodes,
        )
        add_source(anchor_node, "anchor_branch")

        def semantic_candidate_profile(node_id: str) -> dict[str, Any]:
            node = memory.nodes[node_id]
            observed_edges = memory.query_edges(current_node=node_id, actions=actions, create_missing=False)
            usable_alternatives = [
                edge
                for edge in observed_edges
                if not edge.blocked and not edge.is_low_yield and edge.fail_count <= 0
            ]
            fail_support = sum(
                1
                for edge in memory.edges.values()
                if (edge.from_node == node_id or edge.to_node == node_id)
                and (edge.blocked or edge.is_low_yield or edge.fail_count > 0)
            )
            semantic_edge_match_count = sum(
                1
                for edge in observed_edges
                if target_hint_norm and target_hint_norm in _normalize_semantic_text(edge.semantic_hint)
            )
            node_semantic_match = int(
                bool(target_hint_norm and target_hint_norm in _normalize_semantic_text(node.semantic_hint))
            )
            unexplored_usable_count = sum(1 for edge in usable_alternatives if edge.is_unexplored)
            unexplored_usable_ratio = float(unexplored_usable_count) / float(max(1, len(usable_alternatives)))
            semantic_value_proxy = float(len(usable_alternatives)) - (
                self.semantic_exit_fail_support_penalty * float(fail_support)
            )
            return {
                "node_id": node_id,
                "semantic_edge_match_count": int(semantic_edge_match_count),
                "node_semantic_match": int(node_semantic_match),
                "usable_count": int(len(usable_alternatives)),
                "unexplored_usable_count": int(unexplored_usable_count),
                "unexplored_usable_ratio": float(unexplored_usable_ratio),
                "fail_support": int(fail_support),
                "semantic_value_proxy": float(semantic_value_proxy),
                "node_degree": int(memory.node_degree(node_id)),
                "last_visit_step": int(node.last_visit_step),
            }

        failure_candidates: list[tuple[tuple[float, ...], str]] = []
        semantic_candidates: list[tuple[tuple[float, ...], str]] = []
        semantic_failsafe_candidates: list[tuple[tuple[float, ...], str]] = []
        for node in memory.nodes.values():
            if node.id == current_node:
                continue
            observed_edges = memory.query_edges(current_node=node.id, actions=actions, create_missing=False)
            if not observed_edges:
                continue
            usable_alternatives = [
                edge
                for edge in observed_edges
                if not edge.blocked and not edge.is_low_yield and edge.fail_count <= 0
            ]
            fail_support = 0
            if usable_alternatives:
                fail_support = sum(
                    1
                    for edge in memory.edges.values()
                    if (edge.from_node == node.id or edge.to_node == node.id)
                    and (edge.blocked or edge.is_low_yield or edge.fail_count > 0)
                )
                if fail_support > 0:
                    failure_candidates.append(
                        (
                            (
                                float(fail_support),
                                float(len(usable_alternatives)),
                                float(memory.node_degree(node.id)),
                                float(node.last_visit_step),
                            ),
                            node.id,
                        )
                    )
            semantic_edge_match_count = 0
            node_semantic_match = False
            if target_hint_norm:
                semantic_edge_match_count = sum(
                    1
                    for edge in observed_edges
                    if target_hint_norm in _normalize_semantic_text(edge.semantic_hint)
                )
                node_semantic_match = target_hint_norm in _normalize_semantic_text(node.semantic_hint)
                has_any_semantic_signal = bool(semantic_edge_match_count > 0 or node_semantic_match)
                if has_any_semantic_signal:
                    semantic_source_stats["considered"] += 1
                    if self.semantic_exit_fail_support_penalty > 0.0 and fail_support > 0:
                        semantic_source_stats["semantic_fail_support_penalized"] += 1
                    semantic_edge_match_ratio = float(semantic_edge_match_count) / float(max(1, len(observed_edges)))
                    edge_grounding_ok = bool(
                        semantic_edge_match_count >= self.semantic_exit_min_edge_hint_matches
                        and semantic_edge_match_ratio >= self.semantic_exit_min_edge_hint_ratio
                    )
                    if self.semantic_exit_require_node_hint_match:
                        edge_grounding_ok = bool(edge_grounding_ok and node_semantic_match)
                    # Legacy fallback allows node-level semantic match without
                    # edge-level grounding only when explicit edge grounding
                    # gates are not requested.
                    node_fallback_ok = bool(
                        not self.semantic_exit_require_edge_hint_for_node_fallback
                        and self.semantic_exit_min_edge_hint_matches <= 0
                        and self.semantic_exit_min_edge_hint_ratio <= 0.0
                        and node_semantic_match
                    )
                    if edge_grounding_ok or node_fallback_ok:
                        unexplored_usable_count = sum(1 for edge in usable_alternatives if edge.is_unexplored)
                        unexplored_usable_ratio = float(unexplored_usable_count) / float(
                            max(1, len(usable_alternatives))
                        )
                        semantic_value_proxy = float(len(usable_alternatives)) - (
                            self.semantic_exit_fail_support_penalty * float(fail_support)
                        )
                        semantic_sort_key = (
                            float(semantic_edge_match_count),
                            1.0 if node_semantic_match else 0.0,
                            semantic_value_proxy,
                            float(memory.node_degree(node.id)),
                            float(node.last_visit_step),
                        )
                        semantic_candidates.append(
                            (
                                semantic_sort_key,
                                node.id,
                            )
                        )
                    else:
                        semantic_source_stats["rejected_low_grounding"] += 1
                        if (
                            self.semantic_exit_enable_node_consistency_failsafe
                            and failsafe_local_repeat_window_active > 0
                            and self.semantic_exit_require_node_hint_match
                            and node_semantic_match
                        ):
                            # Conservative fail-safe source: only when strict
                            # node-consistency gate rejects all semantic_exit
                            # candidates in the current recovery call.
                            semantic_failsafe_candidates.append(
                                (
                                    (
                                        float(semantic_edge_match_count),
                                        float(len(usable_alternatives))
                                        - (self.semantic_exit_fail_support_penalty * float(fail_support)),
                                        float(memory.node_degree(node.id)),
                                        float(node.last_visit_step),
                                    ),
                                    node.id,
                                )
                            )

        failure_candidates.sort(reverse=True)
        semantic_candidates.sort(reverse=True)
        semantic_failsafe_candidates.sort(reverse=True)
        if semantic_candidates:
            semantic_source_stats["semantic_candidate_pool_size"] = len(semantic_candidates)
            semantic_source_stats["semantic_candidate_pool_competing_count"] = max(
                0, len(semantic_candidates) - 1
            )
            ranked_profiles = [semantic_candidate_profile(node_id) for _, node_id in semantic_candidates]
            top1_profile = ranked_profiles[0]
            best_novelty_profile = max(
                ranked_profiles,
                key=lambda profile: (
                    float(profile["unexplored_usable_ratio"]),
                    int(profile["unexplored_usable_count"]),
                    int(profile["usable_count"]),
                    int(profile["semantic_edge_match_count"]),
                    int(profile["node_semantic_match"]),
                    int(profile["node_degree"]),
                    int(profile["last_visit_step"]),
                    str(profile["node_id"]),
                ),
            )
            semantic_source_stats["semantic_top1_source_node"] = str(top1_profile["node_id"])
            semantic_source_stats["semantic_top1_grounding_count"] = int(top1_profile["semantic_edge_match_count"])
            semantic_source_stats["semantic_top1_node_semantic_match"] = int(top1_profile["node_semantic_match"])
            semantic_source_stats["semantic_top1_usable_count"] = int(top1_profile["usable_count"])
            semantic_source_stats["semantic_top1_unexplored_usable_count"] = int(
                top1_profile["unexplored_usable_count"]
            )
            semantic_source_stats["semantic_top1_unexplored_usable_ratio"] = float(
                top1_profile["unexplored_usable_ratio"]
            )
            semantic_source_stats["semantic_top1_fail_support"] = int(top1_profile["fail_support"])
            semantic_source_stats["semantic_top1_value_proxy"] = float(top1_profile["semantic_value_proxy"])
            semantic_source_stats["semantic_best_novelty_source_node"] = str(best_novelty_profile["node_id"])
            semantic_source_stats["semantic_best_novelty_grounding_count"] = int(
                best_novelty_profile["semantic_edge_match_count"]
            )
            semantic_source_stats["semantic_best_novelty_node_semantic_match"] = int(
                best_novelty_profile["node_semantic_match"]
            )
            semantic_source_stats["semantic_best_novelty_usable_count"] = int(best_novelty_profile["usable_count"])
            semantic_source_stats["semantic_best_novelty_unexplored_usable_count"] = int(
                best_novelty_profile["unexplored_usable_count"]
            )
            semantic_source_stats["semantic_best_novelty_unexplored_usable_ratio"] = float(
                best_novelty_profile["unexplored_usable_ratio"]
            )
            semantic_source_stats["semantic_best_novelty_fail_support"] = int(best_novelty_profile["fail_support"])
            semantic_source_stats["semantic_best_novelty_value_proxy"] = float(
                best_novelty_profile["semantic_value_proxy"]
            )
            semantic_source_stats["semantic_top1_is_best_novelty"] = int(
                str(top1_profile["node_id"]) == str(best_novelty_profile["node_id"])
            )
            semantic_source_stats["semantic_top1_best_novelty_ratio_gap"] = float(
                best_novelty_profile["unexplored_usable_ratio"]
            ) - float(top1_profile["unexplored_usable_ratio"])
        add_source(failure_candidates[0][1] if failure_candidates else None, "failure_detour")
        add_source(semantic_candidates[0][1] if semantic_candidates else None, "semantic_exit")
        if not semantic_candidates:
            add_source(
                semantic_failsafe_candidates[0][1] if semantic_failsafe_candidates else None,
                "semantic_exit_failsafe",
            )
        if self.grounded_only_nonlocal_admission:
            grounded_sources: list[tuple[str, str]] = []
            grounded_branch_types: list[str] = []
            for node_id, branch_type in sources:
                if branch_type in {"semantic_exit", "semantic_exit_failsafe"}:
                    grounded_sources.append((node_id, branch_type))
                    grounded_branch_types.append(branch_type)
                else:
                    semantic_source_stats["grounded_only_nonlocal_filtered"] += 1
            sources = grounded_sources
            branch_types = grounded_branch_types

        if not sources:
            return [], {}, {}, {}, {}, 0, [], semantic_source_stats

        limited_sources = sources[: self.candidate_query_max_sources]
        action_support: dict[str, float] = {}
        action_branch_bonus: dict[str, dict[str, float]] = {}
        edge_support: dict[str, float] = {}
        edge_branch_type: dict[str, str] = {}
        retained_sources: list[tuple[str, str]] = []
        retained_branch_types: list[str] = []
        retrieved_edge_count = 0
        for node_id, branch_type in limited_sources:
            node = memory.nodes.get(node_id)
            if node is None:
                continue
            local_action_support: dict[str, float] = {}
            local_action_branch_bonus: dict[str, dict[str, float]] = {}
            local_edge_support: dict[str, float] = {}
            local_edge_branch_type: dict[str, str] = {}
            local_positive_edges: list[SkeletonEdge] = []
            local_retrieved_edge_count = 0
            observed_edges = memory.query_edges(current_node=node_id, actions=actions, create_missing=False)
            for edge in observed_edges:
                if edge.blocked:
                    continue
                local_retrieved_edge_count += 1
                edge_semantic_match = bool(
                    target_hint_norm and target_hint_norm in _normalize_semantic_text(edge.semantic_hint)
                )
                node_semantic_match = bool(
                    target_hint_norm and target_hint_norm in _normalize_semantic_text(node.semantic_hint)
                )
                semantic_match = bool(edge_semantic_match or node_semantic_match)
                bonus = 0.0
                if branch_type == "anchor_branch":
                    bonus += 0.8
                    if edge.is_unexplored:
                        bonus += 0.4
                    if edge.visit_count <= 1:
                        bonus += 0.2
                elif branch_type == "failure_detour":
                    bonus += 1.0
                    if edge.fail_count <= 0 and not edge.is_low_yield:
                        bonus += 1.0
                    if edge.is_unexplored:
                        bonus += 0.4
                elif branch_type == "semantic_exit":
                    if self.candidate_query_mode == "query_aware_edge_pool":
                        # In edge-pool mode, semantic retrieval should expose
                        # semantically grounded exits, not blindly boost turn
                        # actions from a semantically tagged node.
                        if edge_semantic_match:
                            bonus += 1.8
                            if edge.visit_count <= 1:
                                bonus += 0.2
                        elif (
                            not self.semantic_exit_require_edge_hint_for_node_fallback
                            and self.semantic_exit_min_edge_hint_matches <= 0
                            and self.semantic_exit_min_edge_hint_ratio <= 0.0
                            and node_semantic_match
                            and edge.action == "move_forward"
                            and edge.fail_count <= 0
                            and not edge.is_low_yield
                        ):
                            bonus += 0.8
                            if edge.is_unexplored:
                                bonus += 0.4
                            if edge.visit_count <= 1:
                                bonus += 0.2
                    else:
                        bonus += 0.8
                        if semantic_match:
                            bonus += 1.4
                        if edge.visit_count <= 1:
                            bonus += 0.2
                elif branch_type == "semantic_exit_failsafe":
                    if self.candidate_query_mode == "query_aware_edge_pool":
                        # Fail-safe branch remains conservative: prioritize
                        # edge-level semantic evidence; allow bounded node-level
                        # move_forward fallback when strict gate has no survivor.
                        if edge_semantic_match:
                            bonus += 1.5
                            if edge.visit_count <= 1:
                                bonus += 0.2
                        elif (
                            node_semantic_match
                            and edge.action == "move_forward"
                            and edge.fail_count <= 0
                            and not edge.is_low_yield
                        ):
                            bonus += 0.65
                            if edge.is_unexplored:
                                bonus += 0.35
                            if edge.visit_count <= 1:
                                bonus += 0.2
                    else:
                        bonus += 0.6
                        if semantic_match:
                            bonus += 1.2
                        if edge.visit_count <= 1:
                            bonus += 0.2
                if (
                    branch_type in {"anchor_branch", "failure_detour"}
                    and self.non_grounded_nonlocal_bonus_scale < 1.0
                    and bonus > 0.0
                ):
                    bonus *= self.non_grounded_nonlocal_bonus_scale
                    semantic_source_stats["non_grounded_nonlocal_scaled_edges"] += 1
                if edge.fail_count > 0:
                    bonus -= 0.6
                if edge.is_low_yield:
                    bonus -= 0.8
                if bonus <= 0.0:
                    continue
                local_positive_edges.append(edge)
                local_action_support[edge.action] = local_action_support.get(edge.action, 0.0) + bonus
                branch_bonus = local_action_branch_bonus.setdefault(edge.action, {})
                branch_bonus[branch_type] = branch_bonus.get(branch_type, 0.0) + bonus
                local_edge_support[edge.id] = float(bonus)
                local_edge_branch_type[edge.id] = branch_type

            if (
                self.candidate_query_mode == "query_aware_edge_pool"
                and len(limited_sources) == 1
                and branch_type in {"semantic_exit", "semantic_exit_failsafe"}
                and local_positive_edges
                and all(edge.to_node == current_node for edge in local_positive_edges)
            ):
                # A lone semantic source whose only retained positive edges just
                # replay back into the current planning node is stale fallback
                # evidence in the audited v3 failure pattern. Drop the source
                # and let the existing no-candidate fallback handle the event.
                continue

            retained_sources.append((node_id, branch_type))
            retained_branch_types.append(branch_type)
            if branch_type in {"semantic_exit", "semantic_exit_failsafe"}:
                semantic_source_stats["retained"] += 1
            if branch_type == "semantic_exit_failsafe":
                semantic_source_stats["retained_failsafe"] += 1
            retrieved_edge_count += local_retrieved_edge_count
            for action, bonus in local_action_support.items():
                action_support[action] = action_support.get(action, 0.0) + bonus
            for action, branch_bonus in local_action_branch_bonus.items():
                merged_branch_bonus = action_branch_bonus.setdefault(action, {})
                for branch_name, bonus in branch_bonus.items():
                    merged_branch_bonus[branch_name] = merged_branch_bonus.get(branch_name, 0.0) + bonus
            edge_support.update(local_edge_support)
            edge_branch_type.update(local_edge_branch_type)

        if not retained_sources:
            return [], {}, {}, {}, {}, 0, [], semantic_source_stats

        action_branch_type: dict[str, str] = {}
        for action, branch_bonus in action_branch_bonus.items():
            if not branch_bonus:
                continue
            action_branch_type[action] = max(branch_bonus.items(), key=lambda item: (item[1], item[0]))[0]

        return (
            retained_sources,
            action_support,
            action_branch_type,
            edge_support,
            edge_branch_type,
            retrieved_edge_count,
            retained_branch_types,
            semantic_source_stats,
        )

    def run(
        self,
        *,
        memory: RecoverableSkeletonMemory,
        current_node: str | None,
        recent_nodes: list[str],
        actions: list[str],
        target_hint: str,
        max_retries: int,
        collision_fn: Callable[[str, str], bool],
        backtrack_from_node: str | None = None,
        backtrack_to_node: str | None = None,
    ) -> RecoveryResult:
        if current_node is None:
            return RecoveryResult(
                success=False,
                selected_edge_id=None,
                selected_action=None,
                retries=0,
                reason="no_current_node",
                blocked_edge_ids=[],
                tried_edge_ids=[],
                backtrack_from_node=None,
                backtrack_to_node=None,
                llm_used=False,
                planning_node=None,
            )

        backtrack_from = backtrack_from_node or current_node
        backtrack_to = backtrack_to_node or memory.backtrack_target(recent_nodes=recent_nodes, lookback=5) or current_node
        memory.mark_recovery_anchor(backtrack_to)
        planning_node = current_node
        query_sources = self._candidate_query_sources(
            memory=memory,
            current_node=planning_node,
            backtrack_to_node=backtrack_to,
            recent_nodes=recent_nodes,
            target_hint=target_hint,
        )
        action_support: dict[str, float] | None = None
        action_branch_type: dict[str, str] = {}
        edge_support: dict[str, float] | None = None
        edge_branch_type: dict[str, str] = {}
        retrieved_subgraph_edge_count = 0
        retrieved_branch_types: list[str] = []
        semantic_source_stats: dict[str, Any] = {
            "considered": 0,
            "retained": 0,
            "rejected_low_grounding": 0,
            "retained_failsafe": 0,
            "failsafe_local_repeat_window_active": 0,
            "failsafe_local_repeat_window_unique_nodes": 0,
            "failsafe_local_repeat_window_current_hits": 0,
            "grounded_only_nonlocal_admission_enabled": int(self.grounded_only_nonlocal_admission),
            "grounded_only_nonlocal_filtered": 0,
            "non_grounded_nonlocal_scaled_edges": 0,
            "semantic_fail_support_penalized": 0,
            "semantic_candidate_pool_size": 0,
            "semantic_candidate_pool_competing_count": 0,
            "semantic_top1_source_node": "",
            "semantic_top1_grounding_count": 0,
            "semantic_top1_node_semantic_match": 0,
            "semantic_top1_usable_count": 0,
            "semantic_top1_unexplored_usable_count": 0,
            "semantic_top1_unexplored_usable_ratio": 0.0,
            "semantic_top1_fail_support": 0,
            "semantic_top1_value_proxy": 0.0,
            "semantic_best_novelty_source_node": "",
            "semantic_best_novelty_grounding_count": 0,
            "semantic_best_novelty_node_semantic_match": 0,
            "semantic_best_novelty_usable_count": 0,
            "semantic_best_novelty_unexplored_usable_count": 0,
            "semantic_best_novelty_unexplored_usable_ratio": 0.0,
            "semantic_best_novelty_fail_support": 0,
            "semantic_best_novelty_value_proxy": 0.0,
            "semantic_top1_is_best_novelty": 0,
            "semantic_top1_best_novelty_ratio_gap": 0.0,
            "semantic_pool_repair_attempted": 0,
            "semantic_pool_repair_pool_size": 0,
            "semantic_pool_repair_semantic_exit_present": 0,
            "semantic_pool_repair_debug": "",
        }
        if self.candidate_query_mode in {"query_aware_subgraph", "query_aware_edge_pool"}:
            (
                query_sources,
                action_support,
                action_branch_type,
                edge_support,
                edge_branch_type,
                retrieved_subgraph_edge_count,
                retrieved_branch_types,
                semantic_source_stats,
            ) = self._build_query_aware_subgraph(
                memory=memory,
                current_node=planning_node,
                backtrack_to_node=backtrack_to,
                recent_nodes=recent_nodes,
                target_hint=target_hint,
                actions=actions,
            )
            if (
                int(semantic_source_stats.get("considered", 0)) > 0
                and int(semantic_source_stats.get("semantic_candidate_pool_size", 0)) <= 0
            ):
                semantic_source_stats["semantic_pool_repair_attempted"] = 1
                semantic_source_stats["semantic_pool_repair_semantic_exit_present"] = int(
                    any(source_kind == "semantic_exit" for _, source_kind in query_sources)
                )
                # Phase-A telemetry repair: preserve the existing ranking/execution
                # behavior, but recompute the semantic pool summary when the
                # returned stats remain at defaults despite semantic admission.
                repair_summary = self._summarize_semantic_candidate_pool(
                    memory=memory,
                    current_node=planning_node,
                    target_hint=target_hint,
                    actions=actions,
                )
                semantic_source_stats.update(repair_summary)
                semantic_source_stats["semantic_pool_repair_pool_size"] = int(
                    repair_summary.get("semantic_candidate_pool_size", 0)
                )
                if int(repair_summary.get("semantic_candidate_pool_size", 0)) <= 0:
                    target_hint_norm = _normalize_semantic_text(target_hint)
                    semantic_debug_rows: list[str] = []
                    for node_id, source_kind in query_sources:
                        if source_kind not in {"semantic_exit", "semantic_exit_failsafe"}:
                            continue
                        node = memory.nodes.get(node_id)
                        if node is None:
                            continue
                        observed_edges = memory.query_edges(
                            current_node=node_id,
                            actions=actions,
                            create_missing=False,
                        )
                        usable_count = sum(
                            1
                            for edge in observed_edges
                            if not edge.blocked and not edge.is_low_yield and edge.fail_count <= 0
                        )
                        edge_match_count = sum(
                            1
                            for edge in observed_edges
                            if target_hint_norm and target_hint_norm in _normalize_semantic_text(edge.semantic_hint)
                        )
                        node_match = int(
                            bool(target_hint_norm and target_hint_norm in _normalize_semantic_text(node.semantic_hint))
                        )
                        semantic_debug_rows.append(
                            f"{node_id}:{source_kind}:edge={edge_match_count}:node={node_match}:usable={usable_count}"
                        )
                    semantic_source_stats["semantic_pool_repair_debug"] = (
                        f"target={target_hint_norm};considered={int(semantic_source_stats.get('considered', 0))};"
                        f"retained={int(semantic_source_stats.get('retained', 0))};"
                        f"sources={'|'.join(semantic_debug_rows)}"
                    )
        candidate_source_nodes = [node_id for node_id, _ in query_sources]
        candidate_source_kinds = [source_kind for _, source_kind in query_sources]
        source_kind_by_node = {node_id: source_kind for node_id, source_kind in query_sources}
        retrieved_subgraph_node_count = len(candidate_source_nodes)
        semantic_source_result_fields = {
            "semantic_sources_considered_count": int(semantic_source_stats.get("considered", 0)),
            "semantic_sources_retained_count": int(semantic_source_stats.get("retained", 0)),
            "semantic_sources_rejected_low_grounding_count": int(
                semantic_source_stats.get("rejected_low_grounding", 0)
            ),
            "semantic_sources_retained_failsafe_count": int(semantic_source_stats.get("retained_failsafe", 0)),
            "semantic_failsafe_local_repeat_window_active": int(
                semantic_source_stats.get("failsafe_local_repeat_window_active", 0)
            ),
            "semantic_failsafe_local_repeat_window_unique_nodes": int(
                semantic_source_stats.get("failsafe_local_repeat_window_unique_nodes", 0)
            ),
            "semantic_failsafe_local_repeat_window_current_hits": int(
                semantic_source_stats.get("failsafe_local_repeat_window_current_hits", 0)
            ),
            "grounded_only_nonlocal_admission_enabled": int(
                semantic_source_stats.get("grounded_only_nonlocal_admission_enabled", 0)
            ),
            "grounded_only_nonlocal_filtered_count": int(
                semantic_source_stats.get("grounded_only_nonlocal_filtered", 0)
            ),
            "non_grounded_nonlocal_bonus_scale": float(self.non_grounded_nonlocal_bonus_scale),
            "non_grounded_nonlocal_scaled_edge_count": int(
                semantic_source_stats.get("non_grounded_nonlocal_scaled_edges", 0)
            ),
            "semantic_exit_fail_support_penalty": float(self.semantic_exit_fail_support_penalty),
            "semantic_sources_penalized_fail_support_count": int(
                semantic_source_stats.get("semantic_fail_support_penalized", 0)
            ),
            "semantic_candidate_pool_size": int(semantic_source_stats.get("semantic_candidate_pool_size", 0)),
            "semantic_candidate_pool_competing_count": int(
                semantic_source_stats.get("semantic_candidate_pool_competing_count", 0)
            ),
            "semantic_top1_source_node": str(semantic_source_stats.get("semantic_top1_source_node", "")) or None,
            "semantic_top1_grounding_count": int(semantic_source_stats.get("semantic_top1_grounding_count", 0)),
            "semantic_top1_node_semantic_match": int(
                semantic_source_stats.get("semantic_top1_node_semantic_match", 0)
            ),
            "semantic_top1_usable_count": int(semantic_source_stats.get("semantic_top1_usable_count", 0)),
            "semantic_top1_unexplored_usable_count": int(
                semantic_source_stats.get("semantic_top1_unexplored_usable_count", 0)
            ),
            "semantic_top1_unexplored_usable_ratio": float(
                semantic_source_stats.get("semantic_top1_unexplored_usable_ratio", 0.0)
            ),
            "semantic_top1_fail_support": int(semantic_source_stats.get("semantic_top1_fail_support", 0)),
            "semantic_top1_value_proxy": float(semantic_source_stats.get("semantic_top1_value_proxy", 0.0)),
            "semantic_best_novelty_source_node": str(
                semantic_source_stats.get("semantic_best_novelty_source_node", "")
            )
            or None,
            "semantic_best_novelty_grounding_count": int(
                semantic_source_stats.get("semantic_best_novelty_grounding_count", 0)
            ),
            "semantic_best_novelty_node_semantic_match": int(
                semantic_source_stats.get("semantic_best_novelty_node_semantic_match", 0)
            ),
            "semantic_best_novelty_usable_count": int(
                semantic_source_stats.get("semantic_best_novelty_usable_count", 0)
            ),
            "semantic_best_novelty_unexplored_usable_count": int(
                semantic_source_stats.get("semantic_best_novelty_unexplored_usable_count", 0)
            ),
            "semantic_best_novelty_unexplored_usable_ratio": float(
                semantic_source_stats.get("semantic_best_novelty_unexplored_usable_ratio", 0.0)
            ),
            "semantic_best_novelty_fail_support": int(
                semantic_source_stats.get("semantic_best_novelty_fail_support", 0)
            ),
            "semantic_best_novelty_value_proxy": float(
                semantic_source_stats.get("semantic_best_novelty_value_proxy", 0.0)
            ),
            "semantic_top1_is_best_novelty": int(semantic_source_stats.get("semantic_top1_is_best_novelty", 0)),
            "semantic_top1_best_novelty_ratio_gap": float(
                semantic_source_stats.get("semantic_top1_best_novelty_ratio_gap", 0.0)
            ),
            "semantic_pool_repair_attempted": int(semantic_source_stats.get("semantic_pool_repair_attempted", 0)),
            "semantic_pool_repair_pool_size": int(semantic_source_stats.get("semantic_pool_repair_pool_size", 0)),
            "semantic_pool_repair_semantic_exit_present": int(
                semantic_source_stats.get("semantic_pool_repair_semantic_exit_present", 0)
            ),
            "semantic_pool_repair_debug": str(semantic_source_stats.get("semantic_pool_repair_debug", "")) or None,
        }

        blocked: set[str] = set()
        tried: list[str] = []
        retries = 0
        llm_used = False
        candidate_count_initial = 0
        candidate_count_final = 0
        while retries <= max(0, int(max_retries)):
            candidates: list[SkeletonEdge] = []
            seen: set[str] = set()
            if self.candidate_query_mode == "query_aware_subgraph":
                for edge in memory.candidate_edges(planning_node, actions):
                    if edge.id in seen or edge.id in blocked or edge.blocked:
                        continue
                    candidates.append(edge)
                    seen.add(edge.id)
            else:
                for source_node, _source_kind in query_sources:
                    for edge in memory.candidate_edges(source_node, actions):
                        if edge.id in seen or edge.id in blocked or edge.blocked:
                            continue
                        candidates.append(edge)
                        seen.add(edge.id)
            candidate_count_final = len(candidates)
            if candidate_count_initial <= 0:
                candidate_count_initial = candidate_count_final
            if not candidates:
                return RecoveryResult(
                    success=False,
                    selected_edge_id=None,
                    selected_action=None,
                    retries=retries,
                    reason="no_legal_edges",
                    blocked_edge_ids=sorted(blocked),
                    tried_edge_ids=tried,
                    backtrack_from_node=backtrack_from,
                    backtrack_to_node=backtrack_to,
                    llm_used=llm_used,
                    backtrack_reached_node=current_node,
                    backtrack_anchor_reached=current_node == backtrack_to,
                    backtrack_prefix_steps=0,
                    planning_node=planning_node,
                    candidate_count_initial=candidate_count_initial,
                    candidate_count_final=candidate_count_final,
                    candidate_query_mode=self.candidate_query_mode,
                    candidate_source_nodes=list(candidate_source_nodes),
                    candidate_source_kinds=list(candidate_source_kinds),
                    retrieved_subgraph_node_count=retrieved_subgraph_node_count,
                    retrieved_subgraph_edge_count=retrieved_subgraph_edge_count,
                    retrieved_branch_types=list(dict.fromkeys(retrieved_branch_types)),
                    **semantic_source_result_fields,
                )

            chosen: SkeletonEdge | None
            chosen, used_llm = self._llm_select(
                backtrack_node=backtrack_to,
                candidates=candidates,
                target_hint=target_hint,
            )
            llm_used = llm_used or used_llm
            if chosen is None:
                if self.candidate_query_mode == "query_aware_edge_pool":
                    chosen = self._heuristic_select(candidates, target_hint, None, edge_support)
                    if self.semantic_exit_failsafe_prefer_margin >= 0.0:
                        failsafe_candidates = [
                            edge
                            for edge in candidates
                            if edge_branch_type.get(edge.id, "") == "semantic_exit_failsafe"
                        ]
                        if failsafe_candidates:
                            chosen_score = self._score_edge(chosen, target_hint, None, edge_support)
                            failsafe_best = max(
                                failsafe_candidates,
                                key=lambda edge: self._score_edge(edge, target_hint, None, edge_support),
                            )
                            failsafe_score = self._score_edge(failsafe_best, target_hint, None, edge_support)
                            # Optional tie/margin preference that allows the
                            # bounded fail-safe branch to be selected when it is
                            # close enough to the best non-failsafe candidate.
                            if failsafe_score + self.semantic_exit_failsafe_prefer_margin >= chosen_score:
                                chosen = failsafe_best
                else:
                    chosen = self._heuristic_select(candidates, target_hint, action_support)
            tried.append(chosen.id)
            collided = collision_fn(chosen.action, chosen.id)
            if collided:
                blocked.add(chosen.id)
                retries += 1
                continue

            return RecoveryResult(
                success=True,
                selected_edge_id=chosen.id,
                selected_action=chosen.action,
                retries=retries,
                reason="recovery_success",
                blocked_edge_ids=sorted(blocked),
                tried_edge_ids=tried,
                backtrack_from_node=backtrack_from,
                backtrack_to_node=backtrack_to,
                llm_used=llm_used,
                backtrack_reached_node=current_node,
                backtrack_anchor_reached=current_node == backtrack_to,
                backtrack_prefix_steps=0,
                planning_node=planning_node,
                candidate_count_initial=candidate_count_initial,
                candidate_count_final=candidate_count_final,
                candidate_query_mode=self.candidate_query_mode,
                candidate_source_nodes=list(candidate_source_nodes),
                candidate_source_kinds=list(candidate_source_kinds),
                retrieved_subgraph_node_count=retrieved_subgraph_node_count,
                retrieved_subgraph_edge_count=retrieved_subgraph_edge_count,
                retrieved_branch_types=list(dict.fromkeys(retrieved_branch_types)),
                selected_edge_from_node=chosen.from_node,
                selected_edge_source_kind=(
                    action_branch_type.get(chosen.action)
                    if self.candidate_query_mode == "query_aware_subgraph"
                    else source_kind_by_node.get(chosen.from_node)
                ),
                selected_edge_branch_type=(
                    action_branch_type.get(chosen.action)
                    if self.candidate_query_mode == "query_aware_subgraph"
                    else edge_branch_type.get(chosen.id, "")
                ),
                **semantic_source_result_fields,
            )

        return RecoveryResult(
            success=False,
            selected_edge_id=None,
            selected_action=None,
            retries=retries,
            reason="max_retries_exceeded",
            blocked_edge_ids=sorted(blocked),
            tried_edge_ids=tried,
            backtrack_from_node=backtrack_from,
            backtrack_to_node=backtrack_to,
            llm_used=llm_used,
            backtrack_reached_node=current_node,
            backtrack_anchor_reached=current_node == backtrack_to,
            backtrack_prefix_steps=0,
            planning_node=planning_node,
            candidate_count_initial=candidate_count_initial,
            candidate_count_final=candidate_count_final,
            candidate_query_mode=self.candidate_query_mode,
            candidate_source_nodes=list(candidate_source_nodes),
            candidate_source_kinds=list(candidate_source_kinds),
            retrieved_subgraph_node_count=retrieved_subgraph_node_count,
            retrieved_subgraph_edge_count=retrieved_subgraph_edge_count,
            retrieved_branch_types=list(dict.fromkeys(retrieved_branch_types)),
            **semantic_source_result_fields,
        )
