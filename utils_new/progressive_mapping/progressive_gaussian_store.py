"""Stable root-to-group mapping for metric, surface, and archive proxy splats."""

from typing import Dict, Iterable, List, Optional, Set

import torch


RAW_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def clone_raw_params(params: Dict[str, torch.Tensor], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    result = {}
    for key in RAW_KEYS:
        if key not in params:
            raise ValueError("Missing Gaussian parameter '{}'".format(key))
        value = params[key].detach().clone()
        result[key] = value if device is None else value.to(device)
    return result


class ProgressiveGaussianStore:
    """Use one GaussianModel group per root so transitions remove complete Adam state."""

    def __init__(self, gaussian_model=None):
        self.gaussian_model = gaussian_model
        self.active_metric: Dict[int, Dict[str, torch.Tensor]] = {}
        self.active_surface: Dict[int, Dict[str, torch.Tensor]] = {}
        self.archive_proxies: Dict[int, Dict[str, torch.Tensor]] = {}
        self.root_snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
        self.group_ids: Dict[int, int] = {}
        self.child_rows: Dict[int, Dict[int, int]] = {}
        self._next_virtual_group = 0

    @property
    def device(self) -> torch.device:
        if self.gaussian_model is not None:
            return torch.device(self.gaussian_model.device)
        for collection in (self.active_metric, self.active_surface):
            if collection:
                return next(iter(collection.values()))["means"].device
        return torch.device("cpu")

    def _add_backend_group(self, root_id: int, params: Dict[str, torch.Tensor], optimize: bool) -> int:
        if self.gaussian_model is None:
            group_id = self._next_virtual_group
            self._next_virtual_group += 1
        else:
            group_id = self.gaussian_model.add_progressive_group(params, optimize=optimize, level=0)
        self.group_ids[root_id] = group_id
        return group_id

    def _remove_backend_group(self, root_id: int) -> None:
        group_id = self.group_ids.pop(root_id, None)
        if group_id is not None and self.gaussian_model is not None:
            self.gaussian_model.remove_progressive_group(group_id, level=0)

    def _sync(self, root_id: int, fallback: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.gaussian_model is None:
            return clone_raw_params(fallback)
        return self.gaussian_model.export_progressive_group(self.group_ids[root_id])

    def add_metric(self, root_id: int, params: Dict[str, torch.Tensor]) -> int:
        if root_id in self.group_ids:
            raise ValueError("Root {} already has an active group".format(root_id))
        stored = clone_raw_params(params, self.device)
        self.active_metric[root_id] = stored
        return self._add_backend_group(root_id, stored, optimize=True)

    def remove_metric(self, root_id: int, preserve_snapshot: bool = True) -> Dict[str, torch.Tensor]:
        if root_id not in self.active_metric:
            raise KeyError(root_id)
        params = self._sync(root_id, self.active_metric[root_id])
        if preserve_snapshot:
            self.root_snapshots[root_id] = clone_raw_params(params, torch.device("cpu"))
        self._remove_backend_group(root_id)
        del self.active_metric[root_id]
        return params

    def update_metric(self, root_id: int, params: Dict[str, torch.Tensor]) -> None:
        """Apply a low-rate merge update without changing the stable group identity."""
        if root_id not in self.active_metric:
            raise KeyError(root_id)
        updated = clone_raw_params(params, self.device)
        self.active_metric[root_id] = updated
        if self.gaussian_model is not None:
            self.gaussian_model.update_progressive_group(self.group_ids[root_id], updated)

    def add_surface(
        self, root_id: int, child_node_ids: List[int], params: Dict[str, torch.Tensor]
    ) -> int:
        if params["means"].shape[0] != len(child_node_ids):
            raise ValueError("Each surface child must map to one tensor row")
        stored = clone_raw_params(params, self.device)
        self.active_surface[root_id] = stored
        self.child_rows[root_id] = {child_id: row for row, child_id in enumerate(child_node_ids)}
        return self._add_backend_group(root_id, stored, optimize=True)

    def remove_surface(self, root_id: int) -> Dict[str, torch.Tensor]:
        if root_id not in self.active_surface:
            raise KeyError(root_id)
        params = self._sync(root_id, self.active_surface[root_id])
        self._remove_backend_group(root_id)
        del self.active_surface[root_id]
        self.child_rows.pop(root_id, None)
        return params

    def update_surface(self, root_id: int, params: Dict[str, torch.Tensor]) -> None:
        """Update a surface group in place while preserving its optimizer state."""
        if root_id not in self.active_surface:
            raise KeyError(root_id)
        updated = clone_raw_params(params, self.device)
        self.active_surface[root_id] = updated
        if self.gaussian_model is not None:
            self.gaussian_model.update_progressive_group(self.group_ids[root_id], updated)

    def refine(
        self, root_id: int, child_node_ids: List[int], child_params: Dict[str, torch.Tensor]
    ) -> int:
        self.remove_metric(root_id, preserve_snapshot=True)
        return self.add_surface(root_id, child_node_ids, child_params)

    def set_archive_proxy(self, root_id: int, params: Dict[str, torch.Tensor]) -> None:
        self.archive_proxies[root_id] = clone_raw_params(params, torch.device("cpu"))

    def pop_archive_proxy(self, root_id: int) -> Optional[Dict[str, torch.Tensor]]:
        return self.archive_proxies.pop(root_id, None)

    def backend_group_ids(self) -> Set[int]:
        return set(self.group_ids.values())

    def merge_active_into_baseline(self) -> tuple:
        """Compact active M/S rows into the managed map before final refinement."""
        if self.gaussian_model is None or not self.group_ids:
            return 0, 0
        merged = self.gaussian_model.merge_progressive_groups_into_baseline(level=0)
        self.active_metric.clear()
        self.active_surface.clear()
        self.group_ids.clear()
        self.child_rows.clear()
        return merged

    @property
    def num_metric(self) -> int:
        return len(self.active_metric)

    @property
    def num_surface(self) -> int:
        return sum(params["means"].shape[0] for params in self.active_surface.values())

    @property
    def num_archive_proxies(self) -> int:
        return sum(params["means"].shape[0] for params in self.archive_proxies.values())

    def archive_external_splats(
        self,
        device: torch.device,
        dtype: torch.dtype,
        root_ids: Optional[Set[int]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        proxies = [
            params for root_id, params in self.archive_proxies.items()
            if root_ids is None or root_id in root_ids
        ]
        if not proxies:
            return None
        raw = {
            key: torch.cat([params[key] for params in proxies], dim=0).to(
                device=device, dtype=dtype
            )
            for key in RAW_KEYS
        }
        return {
            "means": raw["means"].detach(),
            "scales": torch.exp(raw["scales"]).detach(),
            "quats": raw["quats"].detach(),
            "opacities": torch.sigmoid(raw["opacities"]).detach(),
            "shs": torch.cat((raw["sh0"], raw["shN"]), dim=1).detach(),
        }

    def active_raw_params(self) -> List[Dict[str, torch.Tensor]]:
        result = []
        for root_id, fallback in self.active_metric.items():
            result.append(self._sync(root_id, fallback))
        for root_id, fallback in self.active_surface.items():
            result.append(self._sync(root_id, fallback))
        return result


def concatenate_raw_params(param_sets: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    sets = list(param_sets)
    if not sets:
        return {}
    return {key: torch.cat([params[key].detach().cpu() for params in sets], dim=0) for key in RAW_KEYS}
