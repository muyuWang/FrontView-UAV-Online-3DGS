"""Deterministic hard-budget decisions for progressive mapping."""

from typing import Dict, Iterable, List, Set

from .types import GaussianTreeNode, ProjectiveAnchor


class BudgetManager:
    def __init__(self, config: Dict[str, object]):
        self.config = config

    def anchor_prune_candidates(
        self, anchors: Iterable[ProjectiveAnchor], current_frame: int
    ) -> List[int]:
        anchors = list(anchors)
        excess = max(0, len(anchors) - int(self.config["max_projective_anchors"]))
        grace_frames = int(self.config["projective_prune_grace_frames"])
        stale_frames = int(self.config["projective_prune_stale_frames"])

        def prune_rank(anchor: ProjectiveAnchor):
            age = max(0, current_frame - anchor.reference_frame_id)
            unseen = max(0, current_frame - anchor.last_seen_frame)
            stale = unseen >= stale_frames
            protected = age < grace_frames and not stale
            reliable_near = (
                anchor.reference_depth_valid
                and 0.0 < anchor.reference_depth_prior <= float(
                    self.config["near_promotion_max_depth_m"]
                )
            )
            # Stale anchors go first, then mature low-quality tracks. Young
            # tracks are only a hard-budget fallback after the other buckets.
            bucket = 0 if stale else 2 if protected else 1
            return (
                bucket,
                int(reliable_near),
                anchor.observation_count,
                anchor.static_confidence,
                -anchor.posterior_entropy,
                anchor.last_seen_frame,
                anchor.anchor_id,
            )

        ranked = sorted(
            anchors,
            key=prune_rank,
        )
        return [anchor.anchor_id for anchor in ranked[:excess]]

    def can_promote(self, num_metric_roots: int) -> bool:
        return num_metric_roots < int(self.config["max_metric_roots"])

    def can_refine(self, num_surface_gaussians: int, required: int = 4) -> bool:
        return num_surface_gaussians + required <= int(self.config["max_surface_gaussians"])

    @staticmethod
    def refinement_priority(node: GaussianTreeNode) -> float:
        return node.projected_radius_ema * node.residual_ema * node.confidence

    def surface_archive_candidates(
        self,
        roots: Iterable[GaussianTreeNode],
        current_frame: int,
        visible_root_ids: Set[int],
        surface_count: int,
        total_active_count: int,
    ) -> List[int]:
        surface_excess = max(0, surface_count - int(self.config["max_surface_gaussians"]))
        total_excess = max(0, total_active_count - int(self.config["max_active_gaussians"]))
        needed = max(surface_excess, total_excess)
        candidates = []
        for root in roots:
            unseen = current_frame - root.last_seen_frame
            timed_out = unseen >= int(self.config["archive_after_unseen_frames"])
            if root.node_id in visible_root_ids and not needed:
                continue
            if timed_out or needed:
                visibility_bonus = 1000000.0 if root.node_id in visible_root_ids else 0.0
                score = unseen - visibility_bonus - root.projected_radius_ema
                candidates.append((score, root))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = []
        released = 0
        for _, root in candidates:
            selected.append(root.node_id)
            # One A proxy replaces the fine children, so only the net release
            # contributes to the total-active budget.
            released += max(1, len(root.children_ids) - 1)
            if needed and released >= needed:
                break
        return selected
