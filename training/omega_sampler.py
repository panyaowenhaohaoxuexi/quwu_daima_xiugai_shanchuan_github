"""Detached, deterministic local Omega sampler for counterfactual route labels."""

from collections import deque

import torch


class OmegaSampler:
    def __init__(self, regions_per_image=6, min_area=16, max_area=256, seed=0, edge_threshold=0.5):
        if not 4 <= regions_per_image <= 8:
            raise ValueError("regions_per_image must be in [4, 8]")
        if not 0 < min_area <= max_area:
            raise ValueError("invalid Omega area range")
        self.regions_per_image = int(regions_per_image)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.seed = int(seed)
        self.edge_threshold = float(edge_threshold)

    @staticmethod
    def _local_density_scale(image, y, x):
        """A detached seed-local density tolerance for connected growth."""
        radius = 2
        y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
        return image[y0:y1, x0:x1].std().clamp_min(0.02)

    def _grow_region(self, image, image_edge, accepted, y, x):
        """Grow one connected component without crossing an edge barrier."""
        height, width = image.shape
        scale = self._local_density_scale(image, y, x)
        seed_density = image[y, x]
        region = torch.zeros_like(accepted)
        visited = torch.zeros_like(accepted)
        queue = deque([(y, x)])
        visited[y, x] = True
        target_area = min(self.max_area, max(self.min_area, int(round(self.min_area * 1.5))))

        while queue and int(region.sum()) < target_area:
            cy, cx = queue.popleft()
            if accepted[cy, cx] or image_edge[cy, cx] > self.edge_threshold:
                continue
            if (image[cy, cx] - seed_density).abs() > (1.5 * scale + 0.01):
                continue
            region[cy, cx] = True
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        if int(region.sum()) < self.min_area:
            return None, scale
        return region, scale

    def sample(self, density_gt, tir_edge, generator=None):
        density = density_gt.detach()
        edge = tir_edge.detach()
        if density.ndim != 4 or density.shape[1] != 1 or edge.shape != density.shape:
            raise ValueError("density_gt and tir_edge must match [B,1,H,W]")
        local_generator = generator or torch.Generator(device=density.device)
        if generator is None:
            local_generator.manual_seed(self.seed)
        supports, weights, owners, classes, statuses = [], [], [], [], []
        for batch_index in range(density.shape[0]):
            image = density[batch_index, 0]
            image_edge = edge[batch_index, 0]
            density_flat = bool(image.std().detach() < 1e-6)
            tir_flat = bool(image_edge.std().detach() < 1e-6)
            quantiles = torch.quantile(image.flatten(), torch.tensor([1 / 3, 2 / 3], device=image.device))
            accepted = torch.zeros_like(image, dtype=torch.bool)
            local_count = 0
            requested_classes = ("low", "middle", "high")
            used_quantile_fallback = False
            for attempt in range(self.regions_per_image * 16):
                if local_count >= self.regions_per_image:
                    break
                klass = requested_classes[local_count % len(requested_classes)]
                if klass == "low":
                    candidates = torch.where(image <= quantiles[0])
                elif klass == "middle":
                    candidates = torch.where((image > quantiles[0]) & (image <= quantiles[1]))
                else:
                    candidates = torch.where(image > quantiles[1])
                if candidates[0].numel() == 0:
                    # A missing quantile never turns into a fake target: choose
                    # a remaining valid seed but retain the requested class for
                    # coverage diagnostics.
                    candidates = torch.where((~accepted) & (image_edge <= self.edge_threshold))
                    used_quantile_fallback = True
                if candidates[0].numel() == 0:
                    break
                choice = int(torch.randint(candidates[0].numel(), (), generator=local_generator, device=image.device))
                y, x = int(candidates[0][choice]), int(candidates[1][choice])
                region, scale = self._grow_region(image, image_edge, accepted, y, x)
                if region is None or (region & accepted).any():
                    continue
                accepted |= region
                support = region.float().unsqueeze(0)
                similarity = torch.exp(-0.5 * ((image - image[y, x]) / scale).square())
                edge_affinity = torch.exp(-image_edge / max(self.edge_threshold, 1e-6))
                weight = (support[0] * similarity * edge_affinity).unsqueeze(0)
                supports.append(support)
                weights.append(weight)
                owners.append(batch_index)
                classes.append(klass)
                local_count += 1
            statuses.append({
                "requested_count": self.regions_per_image,
                "actual_count": local_count,
                "fallback_reason": (
                    "flat_density_and_tir" if density_flat and tir_flat else
                    "flat_density" if density_flat else
                    "flat_tir" if tir_flat else
                    "empty_quantile_fallback" if used_quantile_fallback else
                    "insufficient_nonoverlapping_regions" if local_count < self.regions_per_image else None
                ),
            })
        height, width = density.shape[-2:]
        if supports:
            support = torch.stack(supports).detach()
            weight = torch.stack(weights).detach()
            owner = torch.tensor(owners, device=density.device, dtype=torch.long)
        else:
            support = density.new_zeros((0, 1, height, width)).detach()
            weight = density.new_zeros((0, 1, height, width)).detach()
            owner = torch.empty(0, device=density.device, dtype=torch.long)
        return {
            "omega_support": support,
            "omega_weight": weight,
            "owner_index": owner,
            "quantile_class": classes,
            "status": statuses,
        }
