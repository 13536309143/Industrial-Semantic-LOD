"""Deterministic orientation and local crops; no labels or CAD names used."""
import numpy as np


def normalize(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid point cloud')
    centered = points - (points.max(0) + points.min(0)) * .5
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius <= 1e-10:
        raise ValueError('Degenerate point cloud')
    return (centered / radius).astype(np.float32)


def canonical_points(points):
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    eigenvalues, basis = np.linalg.eigh(centered.T @ centered / max(len(points), 1))
    basis = basis[:, np.argsort(eigenvalues)[::-1]]
    for axis in range(2):
        coordinates = centered @ basis[:, axis]
        pivot = int(np.argmax(np.abs(coordinates)))
        if coordinates[pivot] < 0:
            basis[:, axis] *= -1
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    return normalize(centered @ basis)


def proposals(points, faces, max_crops=2):
    world, canonical = normalize(points), canonical_points(points)
    clouds, kinds = [world, canonical], [0, 1]
    lengths = np.sort(np.ptp(canonical, axis=0))[::-1]
    elongated = lengths[0] / max(lengths[1], 1e-8) >= 4
    if faces >= 10000 or elongated:
        # Overlapping halves preserve local shape. They never independently label the full part.
        axis = int(np.argmax(np.ptp(canonical, axis=0)))
        order = np.argsort(canonical[:, axis], kind='stable')
        size = max(32, int(len(order) * .65))
        for crop in [order[:size], order[-size:]][:max_crops]:
            chosen = canonical[crop]
            # Stable resampling, independent of node order/name.
            indices = np.arange(len(points)) % len(chosen)
            clouds.append(normalize(chosen[indices]))
            kinds.append(2)
    return clouds, kinds
