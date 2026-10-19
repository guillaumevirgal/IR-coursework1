"""Small NumPy-only nearest-neighbour queries for coursework scripts."""

import numpy as np


class NumpyNearestNeighbors:
    """Nearest-neighbour subset matching the cKDTree calls used in this repo.

    Queries are chunked so pairwise distances do not require a large temporary
    array. Supports nearest-neighbour queries and radius queries.
    """

    def __init__(self, data, query_chunk_size=8):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2 or len(self.data) == 0:
            raise ValueError("Neighbour data must be a non-empty 2D array")
        self.query_chunk_size = max(1, int(query_chunk_size))
        self._data_norms = np.einsum("nd,nd->n", self.data, self.data)

    def _distances(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        for start in range(0, len(points), self.query_chunk_size):
            block = points[start:start + self.query_chunk_size]
            block_norms = np.einsum("qd,qd->q", block, block)
            squared = block_norms[:, None] + self._data_norms[None, :] - 2.0 * block @ self.data.T
            yield start, np.maximum(squared, 0.0)

    def query(self, points, k=1):
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        k = int(k)
        if k < 1:
            raise ValueError("k must be at least 1")
        actual_k = min(k, len(self.data))
        distances = np.empty((len(points), actual_k), dtype=float)
        indices = np.empty((len(points), actual_k), dtype=int)
        for start, squared in self._distances(points):
            stop = start + len(squared)
            if actual_k == 1:
                idx = np.argmin(squared, axis=1)[:, None]
            else:
                idx = np.argpartition(squared, actual_k - 1, axis=1)[:, :actual_k]
                order = np.take_along_axis(squared, idx, axis=1).argsort(axis=1)
                idx = np.take_along_axis(idx, order, axis=1)
            indices[start:stop] = idx
            distances[start:stop] = np.sqrt(np.take_along_axis(squared, idx, axis=1))
        if scalar:
            return distances[0, 0], indices[0, 0]
        if k == 1:
            return distances[:, 0], indices[:, 0]
        return distances, indices

    def query_ball_point(self, points, radius):
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        radii = np.broadcast_to(np.asarray(radius, dtype=float), (len(points),))
        results = [None] * len(points)
        for start, squared in self._distances(points):
            for row, distances_squared in enumerate(squared):
                limit = radii[start + row] ** 2
                results[start + row] = np.flatnonzero(distances_squared <= limit).tolist()
        return results[0] if scalar else results
