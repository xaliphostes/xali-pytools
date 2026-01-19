"""
Unified surface data structure for all mesh formats.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field

from xali_tools.core import Serie


@dataclass
class SurfaceData:
    """
    Unified surface data structure.

    Attributes:
        positions: Flat array of vertex positions [x0, y0, z0, x1, y1, z1, ...]
        indices: Flat array of triangle indices [i0, j0, k0, i1, j1, k1, ...] or None
        properties: Dictionary of vertex/face properties (numpy arrays)
        property_sizes: Dictionary of item sizes for each property (1=scalar, 3=vector, etc.)
        name: Surface name
    """
    positions: np.ndarray
    indices: Optional[np.ndarray] = None
    properties: Dict[str, np.ndarray] = field(default_factory=dict)
    property_sizes: Dict[str, int] = field(default_factory=dict)
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

    def get_property(self, name: str) -> Serie:
        """
        Get a property as a Serie view.

        The returned Serie is a view of the underlying data,
        so modifications will affect the original array.

        Args:
            name: Property name.

        Returns:
            Serie view of the property.

        Raises:
            KeyError: If property not found.
        """
        if name not in self.properties:
            raise KeyError(f"Property '{name}' not found. Available: {list(self.properties.keys())}")

        item_size = self.property_sizes.get(name, 1)
        return Serie.as_view(self.properties[name], item_size, name)

    def set_property(
        self,
        name: str,
        data: np.ndarray,
        item_size: int = None
    ) -> None:
        """
        Set a property with its item size.

        Args:
            name: Property name.
            data: Property data as numpy array.
            item_size: Number of components per item. If None, inferred from shape.
        """
        data = np.asarray(data, dtype=np.float64)

        # Infer item_size from shape if not provided
        if item_size is None:
            item_size = data.shape[1] if data.ndim == 2 else 1

        self.properties[name] = data
        self.property_sizes[name] = item_size

    def property_names(self) -> list:
        """Return list of property names."""
        return list(self.properties.keys())

