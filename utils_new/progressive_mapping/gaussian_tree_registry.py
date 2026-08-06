"""Stable node identities and parent-child state for the progressive hierarchy."""

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .types import GaussianTreeNode, NodeState, ProjectiveAnchor


class GaussianTreeRegistry:
    """Own hierarchy metadata independently from mutable Gaussian tensor rows."""

    def __init__(self):
        self.nodes: Dict[int, GaussianTreeNode] = {}
        self._root_ids: List[int] = []
        self._next_node_id = 0

    def create_metric_root(
        self,
        anchor: ProjectiveAnchor,
        world_center: torch.Tensor,
        rotation: torch.Tensor,
        scale: torch.Tensor,
        frame_id: int,
    ) -> GaussianTreeNode:
        node_id = self._next_node_id
        self._next_node_id += 1
        radius = torch.max(scale[:2])
        node = GaussianTreeNode(
            node_id=node_id,
            state=NodeState.METRIC,
            parent_id=None,
            children_ids=[],
            source_anchor_ids=[anchor.anchor_id],
            world_center=world_center.detach().clone(),
            world_bbox_min=(world_center - radius).detach().clone(),
            world_bbox_max=(world_center + radius).detach().clone(),
            metric_gaussian_id=None,
            surface_gaussian_ids=[],
            archive_handle=None,
            observation_count=anchor.observation_count,
            last_seen_frame=frame_id,
            residual_ema=anchor.best_error_ema,
            projected_radius_ema=0.0,
            confidence=float(torch.softmax(anchor.mode_log_weights, dim=0).max().item()),
            root_rotation=rotation.detach().clone(),
            root_scale=scale.detach().clone(),
            root_color=anchor.mean_color.detach().clone(),
            appearance_grid=anchor.appearance_grid.detach().clone(),
            descriptor=anchor.descriptor.detach().clone(),
            support_keyframes=[frame_id],
        )
        self.nodes[node_id] = node
        self._root_ids.append(node_id)
        return node

    def find_merge_candidate(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
        descriptor: torch.Tensor,
        radius_factor: float,
        feature_threshold: float,
    ) -> Optional[GaussianTreeNode]:
        nodes = self.root_nodes(states=(NodeState.METRIC,))
        if not nodes:
            return None

        centers = torch.stack(
            [node.world_center.to(device=center.device, dtype=center.dtype) for node in nodes]
        )
        root_scales = torch.stack(
            [node.root_scale[:2].to(device=scale.device, dtype=scale.dtype) for node in nodes]
        )
        descriptors = torch.stack(
            [
                node.descriptor.to(
                    device=descriptor.device, dtype=descriptor.dtype
                )
                for node in nodes
            ]
        )
        distances = torch.linalg.norm(centers - center.reshape(1, 3), dim=1)
        candidate_scale = torch.maximum(
            root_scales.amax(dim=1), scale[:2].amax().expand(len(nodes))
        )
        similarities = F.cosine_similarity(
            descriptors, descriptor.reshape(1, -1).expand_as(descriptors), dim=1
        )
        valid = (distances < radius_factor * candidate_scale) & (
            similarities >= feature_threshold
        )
        if not bool(valid.any()):
            return None
        best_index = torch.argmin(distances.masked_fill(~valid, float("inf")))
        return nodes[int(best_index.item())]

    def merge_anchor_support(
        self, node: GaussianTreeNode, anchor: ProjectiveAnchor, center: torch.Tensor, scale: torch.Tensor
    ) -> None:
        old_count = max(1, node.observation_count)
        new_count = max(1, anchor.observation_count)
        alpha = min(0.25, new_count / float(old_count + new_count))
        node.world_center = (
            (1.0 - alpha) * node.world_center
            + alpha * center.detach().to(node.world_center.device)
        )
        node.root_scale = (
            (1.0 - alpha) * node.root_scale
            + alpha * scale.detach().to(node.root_scale.device)
        )
        node.root_color = (1.0 - alpha) * node.root_color + alpha * anchor.mean_color.detach().to(node.root_color.device)
        node.appearance_grid = (
            (1.0 - alpha) * node.appearance_grid
            + alpha * anchor.appearance_grid.detach().to(node.appearance_grid.device)
        )
        descriptor = (1.0 - alpha) * node.descriptor + alpha * anchor.descriptor.detach().to(node.descriptor.device)
        node.descriptor = descriptor / torch.clamp(torch.linalg.norm(descriptor), min=1.0e-8)
        node.observation_count += anchor.observation_count
        node.last_seen_frame = max(node.last_seen_frame, anchor.last_seen_frame)
        node.confidence = max(node.confidence, float(torch.softmax(anchor.mode_log_weights, dim=0).max().item()))
        if anchor.anchor_id not in node.source_anchor_ids:
            node.source_anchor_ids.append(anchor.anchor_id)

    def refine(self, root_id: int, child_centers: torch.Tensor, child_scale: torch.Tensor) -> List[int]:
        root = self.nodes[root_id]
        if root.state != NodeState.METRIC:
            raise ValueError("Only METRIC roots can be refined")
        child_ids = []
        for center in child_centers:
            child_id = self._next_node_id
            self._next_node_id += 1
            radius = torch.max(child_scale[:2])
            child = GaussianTreeNode(
                node_id=child_id,
                state=NodeState.SURFACE,
                parent_id=root_id,
                children_ids=[],
                source_anchor_ids=list(root.source_anchor_ids),
                world_center=center.detach().clone(),
                world_bbox_min=(center - radius).detach().clone(),
                world_bbox_max=(center + radius).detach().clone(),
                metric_gaussian_id=None,
                surface_gaussian_ids=[child_id],
                archive_handle=None,
                observation_count=root.observation_count,
                last_seen_frame=root.last_seen_frame,
                residual_ema=root.residual_ema,
                projected_radius_ema=root.projected_radius_ema,
                confidence=root.confidence,
                root_rotation=root.root_rotation.detach().clone(),
                root_scale=child_scale.detach().clone(),
                root_color=root.root_color.detach().clone(),
                appearance_grid=None,
                descriptor=root.descriptor.detach().clone(),
                support_keyframes=list(root.support_keyframes),
            )
            self.nodes[child_id] = child
            child_ids.append(child_id)
        root.state = NodeState.SURFACE
        root.children_ids = child_ids
        root.surface_gaussian_ids = list(child_ids)
        root.metric_gaussian_id = None
        return child_ids

    def mark_archived(self, root_id: int, archive_handle: str) -> None:
        root = self.nodes[root_id]
        if root.state != NodeState.SURFACE:
            raise ValueError("Only SURFACE roots can be archived")
        root.state = NodeState.ARCHIVED
        root.archive_handle = archive_handle
        root.surface_gaussian_ids = []
        for child_id in root.children_ids:
            self.nodes[child_id].state = NodeState.ARCHIVED

    def mark_reactivated(self, root_id: int) -> None:
        root = self.nodes[root_id]
        if root.state != NodeState.ARCHIVED:
            raise ValueError("Only ARCHIVED roots can be reactivated")
        root.state = NodeState.SURFACE
        root.surface_gaussian_ids = list(root.children_ids)
        for child_id in root.children_ids:
            self.nodes[child_id].state = NodeState.SURFACE

    def root_nodes(self, states: Optional[Iterable[NodeState]] = None) -> List[GaussianTreeNode]:
        state_set = None if states is None else set(states)
        return [
            self.nodes[node_id]
            for node_id in self._root_ids
            if state_set is None or self.nodes[node_id].state in state_set
        ]

    def count(self, state: NodeState) -> int:
        if state == NodeState.SURFACE:
            return sum(len(node.children_ids) for node in self.root_nodes((NodeState.SURFACE,)))
        return len(self.root_nodes((state,)))
