"""
Slip analysis for triangulated surfaces.

Provides the SlipAnalyzer class for evaluating slip on fault surfaces
and the SlipResult dataclass for storing analysis results.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Union

from .traction import compute_triangle_normals, resolve_stress
from .coulomb import CoulombParameters, evaluate_coulomb, coulomb_slip_tendency


@dataclass
class SlipResult:
    """
    Results from slip analysis on a triangulated surface.

    Attributes:
        sigma_n: Normal stress on each triangle (tension positive)
        tau: Shear stress magnitude on each triangle
        slip_mask: Boolean array, True where slip occurs
        slip_tendency: Ratio of shear stress to slip threshold
        params: Coulomb parameters used for this analysis
    """

    sigma_n: np.ndarray
    tau: np.ndarray
    slip_mask: np.ndarray
    slip_tendency: np.ndarray
    params: CoulombParameters

    @property
    def n_triangles(self) -> int:
        """Number of triangles analyzed."""
        return len(self.sigma_n)

    @property
    def slip_count(self) -> int:
        """Number of triangles that slip."""
        return int(np.sum(self.slip_mask))

    @property
    def slip_ratio(self) -> float:
        """Fraction of triangles that slip (0 to 1)."""
        if self.n_triangles == 0:
            return 0.0
        return self.slip_count / self.n_triangles

    @property
    def binary_slip(self) -> int:
        """Binary indicator: 1 if any triangle slips, 0 otherwise."""
        return 1 if self.slip_count > 0 else 0

    @property
    def max_slip_tendency(self) -> float:
        """Maximum slip tendency across all triangles."""
        finite_tendency = self.slip_tendency[np.isfinite(self.slip_tendency)]
        if len(finite_tendency) == 0:
            return np.inf
        return float(np.max(finite_tendency))

    def to_attributes(self) -> Dict[str, tuple]:
        """
        Convert results to attribute format for integration with AttributeManager.

        Returns:
            Dictionary of {name: (data, item_size)} tuples
        """
        return {
            "sigma_n": (self.sigma_n, 1),
            "tau": (self.tau, 1),
            "slip": (self.slip_mask.astype(np.float64), 1),
            "slip_tendency": (self.slip_tendency, 1),
        }


class SlipAnalyzer:
    """
    Analyzer for evaluating slip on triangulated fault surfaces.

    The analyzer computes traction vectors from stress tensors applied to
    triangle surfaces, resolves them into normal and shear components,
    and evaluates the Coulomb failure criterion.

    Example:
        >>> from xali_tools.io import load_surface
        >>> from xali_tools.slip_envelope import SlipAnalyzer
        >>>
        >>> surface = load_surface("fault.ts")
        >>> analyzer = SlipAnalyzer(surface)
        >>> result = analyzer.analyze(
        ...     stress=stress_tensor,
        ...     friction=0.6,
        ...     cohesion=0.0,
        ...     pore_pressure=5.0
        ... )
        >>> print(f"Slip ratio: {result.slip_ratio:.2%}")
    """

    def __init__(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ):
        """
        Initialize the slip analyzer.

        Args:
            positions: Vertex positions, shape (n_vertices, 3) or flat
            indices: Triangle indices, shape (n_triangles, 3) or flat
            normals: Optional pre-computed normals, shape (n_triangles, 3).
                     If not provided, computed from mesh geometry.
        """
        # Handle flat arrays
        positions = np.asarray(positions)
        indices = np.asarray(indices)

        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        if indices.ndim == 1:
            indices = indices.reshape(-1, 3)

        self._positions = positions
        self._indices = indices
        self._normals = normals
        self._cached_normals = None

    @classmethod
    def from_surface(cls, surface) -> "SlipAnalyzer":
        """
        Create analyzer from a SurfaceData object.

        Args:
            surface: SurfaceData instance

        Returns:
            SlipAnalyzer instance
        """
        positions = surface.get_positions_matrix()
        indices = surface.get_indices_matrix()
        return cls(positions, indices)

    @property
    def normals(self) -> np.ndarray:
        """
        Get triangle normals (computed lazily and cached).

        Returns:
            Unit normal vectors, shape (n_triangles, 3)
        """
        if self._normals is not None:
            return self._normals

        if self._cached_normals is None:
            self._cached_normals = compute_triangle_normals(
                self._positions, self._indices
            )
        return self._cached_normals

    @property
    def n_triangles(self) -> int:
        """Number of triangles in the mesh."""
        return self._indices.shape[0]

    def analyze(
        self,
        stress: np.ndarray,
        friction: float = 0.6,
        cohesion: float = 0.0,
        pore_pressure: float = 0.0,
    ) -> SlipResult:
        """
        Analyze slip for given stress and Coulomb parameters.

        Args:
            stress: Stress tensor, shape (6,) for uniform stress,
                   or (n_triangles, 6) for per-triangle stress.
                   Components: [Sxx, Sxy, Sxz, Syy, Syz, Szz]
            friction: Friction coefficient (mu), default 0.6
            cohesion: Cohesion strength (c), default 0.0
            pore_pressure: Pore fluid pressure (Pp), default 0.0

        Returns:
            SlipResult with normal/shear stress, slip mask, and tendency
        """
        stress = np.asarray(stress)
        params = CoulombParameters(
            friction=friction, cohesion=cohesion, pore_pressure=pore_pressure
        )

        # Resolve stress into normal and shear components
        sigma_n, tau = resolve_stress(stress, self.normals)

        # Evaluate Coulomb criterion
        slip_mask = evaluate_coulomb(sigma_n, tau, params)
        tendency = coulomb_slip_tendency(sigma_n, tau, params)

        return SlipResult(
            sigma_n=sigma_n,
            tau=tau,
            slip_mask=slip_mask,
            slip_tendency=tendency,
            params=params,
        )

    def analyze_with_params(
        self, stress: np.ndarray, params: CoulombParameters
    ) -> SlipResult:
        """
        Analyze slip using a CoulombParameters object.

        Args:
            stress: Stress tensor, shape (6,) or (n_triangles, 6)
            params: CoulombParameters instance

        Returns:
            SlipResult with analysis results
        """
        return self.analyze(
            stress=stress,
            friction=params.friction,
            cohesion=params.cohesion,
            pore_pressure=params.pore_pressure,
        )
