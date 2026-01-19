"""
Stress tensor operations for triangulated surfaces.

A stress tensor is symmetric and has 6 independent components:
[σxx, σxy, σxz, σyy, σyz, σzz]

This module provides a class to store multiple stress properties
at mesh vertices and perform weighted combinations.
"""

import numpy as np
from typing import List, Union
from dataclasses import dataclass, field


@dataclass
class StressProperties:
    """
    Store n stress properties for m vertices of a triangulated surface.

    Each stress property is a flattened array of size 6*m containing
    the 6 stress components [σxx, σxy, σxz, σyy, σyz, σzz] for each vertex.

    Attributes:
        n_vertices: Number of vertices (m)
        stresses: List of stress arrays, each of shape (6*m,)
        names: Optional names for each stress property

    Example:
        ```python
        # Create storage for 100 vertices
        sp = StressProperties(n_vertices=100)

        # Add stress properties
        sp.add(stress1, name="remote")
        sp.add(stress2, name="fault_1")
        sp.add(stress3, name="fault_2")

        # Weighted sum: 1.0*remote + 0.5*fault_1 + 0.3*fault_2
        combined = sp.weighted_sum([1.0, 0.5, 0.3])
        ```
    """
    n_vertices: int
    stresses: List[np.ndarray] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate after initialization."""
        if self.n_vertices <= 0:
            raise ValueError("n_vertices must be positive")

    @property
    def n_stresses(self) -> int:
        """Number of stress properties stored."""
        return len(self.stresses)

    @property
    def expected_size(self) -> int:
        """Expected size of each stress array (6 * n_vertices)."""
        return 6 * self.n_vertices

    def add(self, stress: np.ndarray, name: str = "") -> None:
        """
        Add a stress property.

        Args:
            stress: Flattened stress array of size 6*m containing
                    [σxx, σxy, σxz, σyy, σyz, σzz] for each vertex.
            name: Optional name for this stress property.

        Raises:
            ValueError: If stress array has wrong size.
        """
        stress = np.asarray(stress, dtype=np.float64).flatten()

        if stress.size != self.expected_size:
            raise ValueError(
                f"Stress array size {stress.size} does not match "
                f"expected size {self.expected_size} (6 * {self.n_vertices})"
            )

        self.stresses.append(stress)
        self.names.append(name if name else f"stress_{self.n_stresses}")

    def get(self, index_or_name: Union[int, str]) -> np.ndarray:
        """
        Get a stress property by index or name.

        Args:
            index_or_name: Integer index or string name.

        Returns:
            Stress array of size 6*m.
        """
        if isinstance(index_or_name, str):
            try:
                idx = self.names.index(index_or_name)
            except ValueError:
                raise KeyError(f"Stress '{index_or_name}' not found. Available: {self.names}")
            return self.stresses[idx]
        return self.stresses[index_or_name]

    def weighted_sum(self, weights: List[float]) -> np.ndarray:
        """
        Compute weighted sum of all stress properties.

        result = Σ (weights[i] * stresses[i])

        Args:
            weights: List of weights, one per stress property.

        Returns:
            Combined stress array of size 6*m.

        Raises:
            ValueError: If number of weights doesn't match number of stresses.
        """
        if len(weights) != self.n_stresses:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of stresses ({self.n_stresses})"
            )

        if self.n_stresses == 0:
            raise ValueError("No stress properties to combine")

        result = np.zeros(self.expected_size, dtype=np.float64)

        for w, stress in zip(weights, self.stresses):
            result += w * stress

        return result

    def weighted_sum_by_name(self, weights: dict) -> np.ndarray:
        """
        Compute weighted sum using a dictionary of name -> weight.

        Args:
            weights: Dictionary mapping stress names to weights.
                     Stresses not in the dictionary get weight 0.

        Returns:
            Combined stress array of size 6*m.

        Example:
            combined = sp.weighted_sum_by_name({
                "remote": 1.0,
                "fault_1": 0.5,
                "fault_2": 0.3
            })
        """
        weight_list = [weights.get(name, 0.0) for name in self.names]
        return self.weighted_sum(weight_list)

    def to_matrix(self, index_or_name: Union[int, str] = None,
                  stress_array: np.ndarray = None) -> np.ndarray:
        """
        Reshape a stress array to (m, 6) matrix for easier manipulation.

        Args:
            index_or_name: Get stress by index or name (optional).
            stress_array: Use this array directly (optional).
                         One of index_or_name or stress_array must be provided.

        Returns:
            Array of shape (m, 6) where each row is
            [σxx, σxy, σxz, σyy, σyz, σzz] for a vertex.
        """
        if stress_array is not None:
            arr = np.asarray(stress_array)
        elif index_or_name is not None:
            arr = self.get(index_or_name)
        else:
            raise ValueError("Provide either index_or_name or stress_array")

        return arr.reshape(self.n_vertices, 6)

    def component(self, index_or_name: Union[int, str],
                  component: Union[int, str]) -> np.ndarray:
        """
        Extract a single stress component for all vertices.

        Args:
            index_or_name: Stress index or name.
            component: Component index (0-5) or name
                      ('xx', 'yy', 'zz', 'xy', 'xz', 'yz').

        Returns:
            Array of size m with the component value at each vertex.
        """
        component_map = {'xx': 0, 'xy': 1, 'xz': 2, 'yy': 3, 'yz': 4, 'zz': 5}

        if isinstance(component, str):
            component = component_map.get(component.lower())
            if component is None:
                raise ValueError(f"Unknown component. Use: {list(component_map.keys())}")

        matrix = self.to_matrix(index_or_name)
        return matrix[:, component]

    def to_tensors(self, index_or_name: Union[int, str] = None,
                   stress_array: np.ndarray = None) -> np.ndarray:
        """
        Convert stress array to 3x3 symmetric tensor matrices.

        Args:
            index_or_name: Get stress by index or name (optional).
            stress_array: Use this array directly (optional).

        Returns:
            Array of shape (m, 3, 3) containing symmetric stress tensors.
        """
        matrix = self.to_matrix(index_or_name, stress_array)

        # Build symmetric 3x3 tensors
        # Order: [σxx, σxy, σxz, σyy, σyz, σzz]
        tensors = np.zeros((self.n_vertices, 3, 3), dtype=np.float64)

        tensors[:, 0, 0] = matrix[:, 0]  # σxx
        tensors[:, 0, 1] = matrix[:, 1]  # σxy
        tensors[:, 0, 2] = matrix[:, 2]  # σxz
        tensors[:, 1, 0] = matrix[:, 1]  # σxy (symmetric)
        tensors[:, 1, 1] = matrix[:, 3]  # σyy
        tensors[:, 1, 2] = matrix[:, 4]  # σyz
        tensors[:, 2, 0] = matrix[:, 2]  # σxz (symmetric)
        tensors[:, 2, 1] = matrix[:, 4]  # σyz (symmetric)
        tensors[:, 2, 2] = matrix[:, 5]  # σzz

        return tensors

    def principal_values(self, index_or_name: Union[int, str] = None,
                         stress_array: np.ndarray = None) -> np.ndarray:
        """
        Compute principal stress values (eigenvalues) for all vertices.

        Principal stresses are ordered: σ1 >= σ2 >= σ3

        Args:
            index_or_name: Get stress by index or name (optional).
            stress_array: Use this array directly (optional).

        Returns:
            Flattened array of size 3*m containing [σ1, σ2, σ3] for each vertex.
        """
        tensors = self.to_tensors(index_or_name, stress_array)

        # Compute eigenvalues for all tensors
        eigenvalues = np.linalg.eigvalsh(tensors)

        # Sort in descending order (σ1 >= σ2 >= σ3)
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]

        return eigenvalues.flatten()

    def principal_directions(self, index_or_name: Union[int, str] = None,
                             stress_array: np.ndarray = None) -> np.ndarray:
        """
        Compute principal stress directions (eigenvectors) for all vertices.

        Directions are ordered corresponding to σ1 >= σ2 >= σ3.
        Each direction is a unit vector [nx, ny, nz].

        Args:
            index_or_name: Get stress by index or name (optional).
            stress_array: Use this array directly (optional).

        Returns:
            Flattened array of size 9*m containing [n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z]
            for each vertex (3 directions × 3 components).
        """
        tensors = self.to_tensors(index_or_name, stress_array)

        # Compute eigenvalues and eigenvectors for all tensors
        eigenvalues, eigenvectors = np.linalg.eigh(tensors)

        # Sort by eigenvalues in descending order (σ1 >= σ2 >= σ3)
        sort_idx = np.argsort(eigenvalues, axis=1)[:, ::-1]

        # Reorder eigenvectors according to sorted eigenvalues
        # eigenvectors has shape (m, 3, 3) where eigenvectors[i, :, j] is the j-th eigenvector
        directions = np.zeros((self.n_vertices, 3, 3), dtype=np.float64)
        for i in range(self.n_vertices):
            directions[i] = eigenvectors[i, :, sort_idx[i]].T

        # directions[i] is now (3, 3) with rows being the principal directions
        return directions.reshape(self.n_vertices, 9).flatten()

    def principal_values_matrix(self, index_or_name: Union[int, str] = None,
                                stress_array: np.ndarray = None) -> np.ndarray:
        """
        Compute principal stress values as (m, 3) matrix.

        Args:
            index_or_name: Get stress by index or name (optional).
            stress_array: Use this array directly (optional).

        Returns:
            Array of shape (m, 3) with columns [σ1, σ2, σ3].
        """
        values = self.principal_values(index_or_name, stress_array)
        return values.reshape(self.n_vertices, 3)

    def principal_directions_matrix(self, index_or_name: Union[int, str] = None,
                                    stress_array: np.ndarray = None) -> np.ndarray:
        """
        Compute principal stress directions as (m, 3, 3) matrix.

        Args:
            index_or_name: Get stress by index or name (optional).
            stress_array: Use this array directly (optional).

        Returns:
            Array of shape (m, 3, 3) where result[i, j, :] is the j-th principal
            direction (unit vector) at vertex i.
        """
        directions = self.principal_directions(index_or_name, stress_array)
        return directions.reshape(self.n_vertices, 3, 3)

    def __repr__(self) -> str:
        return (f"StressProperties(n_vertices={self.n_vertices}, "
                f"n_stresses={self.n_stresses}, names={self.names})")
