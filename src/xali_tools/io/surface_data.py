"""
Unified surface data structure for all mesh formats.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SurfaceData:
    """
    Unified surface data structure.

    Attributes:
        positions: Flat array of vertex positions [x0, y0, z0, x1, y1, z1, ...]
        indices: Flat array of triangle indices [i0, j0, k0, i1, j1, k1, ...] or None
        properties: Dictionary of vertex/face properties
        name: Surface name
    """
    positions: np.ndarray
    indices: Optional[np.ndarray] = None
    properties: Dict[str, np.ndarray] = field(default_factory=dict)
    name: str = "surface"

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.positions) // 3

    @property
    def n_triangles(self) -> int:
        """Number of triangles."""
        if self.indices is None:
            return 0
        return len(self.indices) // 3

    def get_positions_matrix(self) -> np.ndarray:
        """Return positions as (n, 3) matrix."""
        return self.positions.reshape(-1, 3)

    def get_indices_matrix(self) -> Optional[np.ndarray]:
        """Return indices as (n, 3) matrix."""
        if self.indices is None:
            return None
        return self.indices.reshape(-1, 3)

