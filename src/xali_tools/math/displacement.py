"""
Displacement field operations for triangulated surfaces.

A displacement vector has 3 components: [ux, uy, uz]

This module provides a class to store multiple displacement properties
at mesh vertices and perform weighted combinations.
"""

import numpy as np
from typing import List, Union
from dataclasses import dataclass, field


@dataclass
class DisplacementProperties:
    """
    Store n displacement properties for m vertices of a triangulated surface.

    Each displacement property is a flattened array of size 3*m containing
    the 3 displacement components [ux, uy, uz] for each vertex.

    Attributes:
        n_vertices: Number of vertices (m)
        displacements: List of displacement arrays, each of shape (3*m,)
        names: Optional names for each displacement property

    Example:
        ```python
        # Create storage for 100 vertices
        dp = DisplacementProperties(n_vertices=100)

        # Add displacement properties
        dp.add(disp1, name="remote")
        dp.add(disp2, name="fault_1")
        dp.add(disp3, name="fault_2")

        # Weighted sum: 1.0*remote + 0.5*fault_1 + 0.3*fault_2
        combined = dp.weighted_sum([1.0, 0.5, 0.3])
        ```
    """
    n_vertices: int
    displacements: List[np.ndarray] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate after initialization."""
        if self.n_vertices <= 0:
            raise ValueError("n_vertices must be positive")

    @property
    def n_displacements(self) -> int:
        """Number of displacement properties stored."""
        return len(self.displacements)

    @property
    def expected_size(self) -> int:
        """Expected size of each displacement array (3 * n_vertices)."""
        return 3 * self.n_vertices

    def add(self, displacement: np.ndarray, name: str = "") -> None:
        """
        Add a displacement property.

        Args:
            displacement: Flattened displacement array of size 3*m containing
                    [ux, uy, uz] for each vertex.
            name: Optional name for this displacement property.

        Raises:
            ValueError: If displacement array has wrong size.
        """
        displacement = np.asarray(displacement, dtype=np.float64).flatten()

        if displacement.size != self.expected_size:
            raise ValueError(
                f"Displacement array size {displacement.size} does not match "
                f"expected size {self.expected_size} (3 * {self.n_vertices})"
            )

        self.displacements.append(displacement)
        self.names.append(name if name else f"displacement_{self.n_displacements}")

    def get(self, index_or_name: Union[int, str]) -> np.ndarray:
        """
        Get a displacement property by index or name.

        Args:
            index_or_name: Integer index or string name.

        Returns:
            Displacement array of size 3*m.
        """
        if isinstance(index_or_name, str):
            try:
                idx = self.names.index(index_or_name)
            except ValueError:
                raise KeyError(f"Displacement '{index_or_name}' not found. Available: {self.names}")
            return self.displacements[idx]
        return self.displacements[index_or_name]

    def weighted_sum(self, weights: List[float]) -> np.ndarray:
        """
        Compute weighted sum of all displacement properties.

        result = Σ (weights[i] * displacements[i])

        Args:
            weights: List of weights, one per displacement property.

        Returns:
            Combined displacement array of size 3*m.

        Raises:
            ValueError: If number of weights doesn't match number of displacements.
        """
        if len(weights) != self.n_displacements:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of displacements ({self.n_displacements})"
            )

        if self.n_displacements == 0:
            raise ValueError("No displacement properties to combine")

        result = np.zeros(self.expected_size, dtype=np.float64)

        for w, displacement in zip(weights, self.displacements):
            result += w * displacement

        return result

    def weighted_sum_by_name(self, weights: dict) -> np.ndarray:
        """
        Compute weighted sum using a dictionary of name -> weight.

        Args:
            weights: Dictionary mapping displacement names to weights.
                     Displacements not in the dictionary get weight 0.

        Returns:
            Combined displacement array of size 3*m.

        Example:
            combined = dp.weighted_sum_by_name({
                "remote": 1.0,
                "fault_1": 0.5,
                "fault_2": 0.3
            })
        """
        weight_list = [weights.get(name, 0.0) for name in self.names]
        return self.weighted_sum(weight_list)

    def to_matrix(self, index_or_name: Union[int, str] = None,
                  displacement_array: np.ndarray = None) -> np.ndarray:
        """
        Reshape a displacement array to (m, 3) matrix for easier manipulation.

        Args:
            index_or_name: Get displacement by index or name (optional).
            displacement_array: Use this array directly (optional).
                         One of index_or_name or displacement_array must be provided.

        Returns:
            Array of shape (m, 3) where each row is [ux, uy, uz] for a vertex.
        """
        if displacement_array is not None:
            arr = np.asarray(displacement_array)
        elif index_or_name is not None:
            arr = self.get(index_or_name)
        else:
            raise ValueError("Provide either index_or_name or displacement_array")

        return arr.reshape(self.n_vertices, 3)

    def component(self, index_or_name: Union[int, str],
                  component: Union[int, str]) -> np.ndarray:
        """
        Extract a single displacement component for all vertices.

        Args:
            index_or_name: Displacement index or name.
            component: Component index (0-2) or name ('x', 'y', 'z').

        Returns:
            Array of size m with the component value at each vertex.
        """
        component_map = {'x': 0, 'y': 1, 'z': 2}

        if isinstance(component, str):
            component = component_map.get(component.lower())
            if component is None:
                raise ValueError(f"Unknown component. Use: {list(component_map.keys())}")

        matrix = self.to_matrix(index_or_name)
        return matrix[:, component]

    def magnitude(self, index_or_name: Union[int, str] = None,
                  displacement_array: np.ndarray = None) -> np.ndarray:
        """
        Compute magnitude of displacement at each vertex.

        Args:
            index_or_name: Get displacement by index or name (optional).
            displacement_array: Use this array directly (optional).

        Returns:
            Array of size m with magnitude at each vertex.
        """
        matrix = self.to_matrix(index_or_name, displacement_array)
        return np.sqrt(np.sum(matrix**2, axis=1))

    def __repr__(self) -> str:
        return (f"DisplacementProperties(n_vertices={self.n_vertices}, "
                f"n_displacements={self.n_displacements}, names={self.names})")
