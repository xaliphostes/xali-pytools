"""
Domain sweep analysis for slip envelope computation.

Provides the DomainSweepAnalyzer for performing 2D and 3D parameter sweeps
to map slip/no-slip boundaries in parameter space.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

from .parameters import DomainSpecification, ParameterType
from .slip_analysis import SlipAnalyzer, SlipResult


class OutputMetric(Enum):
    """
    Output metrics for domain sweep analysis.

    SLIP_RATIO: Fraction of triangles that slip (0 to 1)
    BINARY_SLIP: 1 if any triangle slips, 0 otherwise
    SLIP_COUNT: Number of triangles that slip
    MAX_SLIP_TENDENCY: Maximum slip tendency across triangles
    """

    SLIP_RATIO = "slip_ratio"
    BINARY_SLIP = "binary_slip"
    SLIP_COUNT = "slip_count"
    MAX_SLIP_TENDENCY = "max_slip_tendency"


@dataclass
class DomainResult:
    """
    Results from a domain sweep analysis.

    Stores the metric values computed at each grid point in parameter space.

    Attributes:
        values: Grid of metric values, shape matches domain specification
        axes: List of axis value arrays
        axis_names: List of parameter names for each axis
        metric: The output metric used
    """

    values: np.ndarray
    axes: List[np.ndarray]
    axis_names: List[str]
    metric: OutputMetric

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions (1, 2, or 3)."""
        return len(self.axes)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the values grid."""
        return self.values.shape

    def get_2d_contour_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data formatted for 2D contour plotting.

        Returns:
            Tuple of (X, Y, Z) meshgrid arrays suitable for matplotlib contourf

        Raises:
            ValueError: If domain is not 2D
        """
        if self.n_dimensions != 2:
            raise ValueError(f"Expected 2D domain, got {self.n_dimensions}D")

        X, Y = np.meshgrid(self.axes[0], self.axes[1], indexing="ij")
        return X, Y, self.values

    def get_slice(
        self, axis: int, index: int
    ) -> "DomainResult":
        """
        Extract a lower-dimensional slice from a 3D result.

        Args:
            axis: Axis to slice along (0, 1, or 2)
            index: Index along the sliced axis

        Returns:
            DomainResult with one fewer dimension

        Raises:
            ValueError: If domain is not 3D or axis is invalid
        """
        if self.n_dimensions != 3:
            raise ValueError(f"Slicing requires 3D domain, got {self.n_dimensions}D")
        if axis < 0 or axis > 2:
            raise ValueError(f"Invalid axis {axis}, must be 0, 1, or 2")

        # Slice the values array
        sliced_values = np.take(self.values, index, axis=axis)

        # Remove the sliced axis from axes and names
        remaining_axes = [a for i, a in enumerate(self.axes) if i != axis]
        remaining_names = [n for i, n in enumerate(self.axis_names) if i != axis]

        return DomainResult(
            values=sliced_values,
            axes=remaining_axes,
            axis_names=remaining_names,
            metric=self.metric,
        )

    def get_slice_at_value(
        self, axis: int, value: float
    ) -> "DomainResult":
        """
        Extract a slice at the nearest grid point to a given value.

        Args:
            axis: Axis to slice along (0, 1, or 2)
            value: Target value on the axis

        Returns:
            DomainResult with one fewer dimension
        """
        index = int(np.argmin(np.abs(self.axes[axis] - value)))
        return self.get_slice(axis, index)

    def get_isosurface_level(self, threshold: float = 0.5) -> float:
        """
        Find the iso-value for a given slip ratio threshold.

        For SLIP_RATIO or BINARY_SLIP metrics, returns the threshold.
        For other metrics, interpolates based on the data range.

        Args:
            threshold: Target threshold (0 to 1 for ratio metrics)

        Returns:
            Iso-value for contouring
        """
        if self.metric in (OutputMetric.SLIP_RATIO, OutputMetric.BINARY_SLIP):
            return threshold
        else:
            # For count/tendency, interpolate based on range
            min_val = np.nanmin(self.values)
            max_val = np.nanmax(self.values)
            return min_val + threshold * (max_val - min_val)


class DomainSweepAnalyzer:
    """
    Analyzer for sweeping parameter domains and computing slip envelopes.

    Performs slip analysis at each point in a parameter grid and aggregates
    results for visualization as iso-contours or iso-surfaces.

    Example:
        >>> from xali_tools.slip_envelope import (
        ...     DomainSweepAnalyzer, friction_pore_pressure_domain, OutputMetric
        ... )
        >>>
        >>> analyzer = DomainSweepAnalyzer(positions, indices)
        >>> domain = friction_pore_pressure_domain()
        >>> result = analyzer.sweep(domain, stress, OutputMetric.SLIP_RATIO)
        >>>
        >>> # Plot 2D contours
        >>> X, Y, Z = result.get_2d_contour_data()
        >>> plt.contourf(X, Y, Z, levels=20)
        >>> plt.contour(X, Y, Z, levels=[0.5], colors='red')  # 50% slip boundary
    """

    def __init__(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ):
        """
        Initialize the domain sweep analyzer.

        Args:
            positions: Vertex positions, shape (n_vertices, 3) or flat
            indices: Triangle indices, shape (n_triangles, 3) or flat
            normals: Optional pre-computed normals, shape (n_triangles, 3)
        """
        self._slip_analyzer = SlipAnalyzer(positions, indices, normals)

    @classmethod
    def from_surface(cls, surface) -> "DomainSweepAnalyzer":
        """
        Create analyzer from a SurfaceData object.

        Args:
            surface: SurfaceData instance

        Returns:
            DomainSweepAnalyzer instance
        """
        positions = surface.get_positions_matrix()
        indices = surface.get_indices_matrix()
        return cls(positions, indices)

    def _extract_metric(self, result: SlipResult, metric: OutputMetric) -> float:
        """Extract the requested metric from a SlipResult."""
        if metric == OutputMetric.SLIP_RATIO:
            return result.slip_ratio
        elif metric == OutputMetric.BINARY_SLIP:
            return float(result.binary_slip)
        elif metric == OutputMetric.SLIP_COUNT:
            return float(result.slip_count)
        elif metric == OutputMetric.MAX_SLIP_TENDENCY:
            return result.max_slip_tendency
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _get_stress_for_params(
        self,
        base_stress: np.ndarray,
        params: Dict[str, float],
        stress_modifier: Optional[Callable[[np.ndarray, Dict[str, float]], np.ndarray]],
    ) -> np.ndarray:
        """
        Get the stress tensor for given parameters.

        If a stress_modifier is provided, it will be called with the base stress
        and parameters to compute the modified stress.
        """
        if stress_modifier is None:
            return base_stress
        return stress_modifier(base_stress, params)

    def sweep(
        self,
        domain: DomainSpecification,
        stress: np.ndarray,
        metric: OutputMetric = OutputMetric.SLIP_RATIO,
        stress_modifier: Optional[
            Callable[[np.ndarray, Dict[str, float]], np.ndarray]
        ] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DomainResult:
        """
        Perform a parameter domain sweep.

        Args:
            domain: DomainSpecification defining the parameter grid
            stress: Base stress tensor, shape (6,) or (n_triangles, 6)
            metric: Output metric to compute at each grid point
            stress_modifier: Optional function to modify stress based on parameters.
                            Signature: (base_stress, params_dict) -> modified_stress
                            This allows varying stress with parameters like stress_ratio.
            progress_callback: Optional callback for progress updates.
                              Signature: (current, total) -> None

        Returns:
            DomainResult with metric values at each grid point
        """
        stress = np.asarray(stress)

        # Initialize output array
        values = np.zeros(domain.shape)
        total_points = domain.n_points
        current_point = 0

        # Iterate over all grid points
        for idx, params in domain.iter_grid_with_indices():
            # Get Coulomb parameters
            friction = params.get("friction", 0.6)
            cohesion = params.get("cohesion", 0.0)
            pore_pressure = params.get("pore_pressure", 0.0)

            # Get stress (possibly modified by stress-related parameters)
            current_stress = self._get_stress_for_params(stress, params, stress_modifier)

            # Perform slip analysis
            result = self._slip_analyzer.analyze(
                stress=current_stress,
                friction=friction,
                cohesion=cohesion,
                pore_pressure=pore_pressure,
            )

            # Extract and store metric
            values[idx] = self._extract_metric(result, metric)

            # Progress callback
            current_point += 1
            if progress_callback is not None:
                progress_callback(current_point, total_points)

        return DomainResult(
            values=values,
            axes=domain.axes,
            axis_names=domain.axis_names,
            metric=metric,
        )

    def sweep_with_stress_generator(
        self,
        domain: DomainSpecification,
        stress_generator: Callable[[Dict[str, float]], np.ndarray],
        metric: OutputMetric = OutputMetric.SLIP_RATIO,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DomainResult:
        """
        Perform a domain sweep with a custom stress generator.

        This allows full control over how stress is computed for each parameter
        combination, useful for Anderson/Principal stress parameterizations.

        Args:
            domain: DomainSpecification defining the parameter grid
            stress_generator: Function that takes parameters and returns stress tensor.
                             Signature: (params_dict) -> stress_array(6,) or (n_triangles, 6)
            metric: Output metric to compute at each grid point
            progress_callback: Optional callback for progress updates

        Returns:
            DomainResult with metric values at each grid point

        Example:
            >>> from xali_tools.geophysics import from_anderson
            >>>
            >>> def generate_stress(params):
            ...     ratio = params.get('stress_ratio', 1.0)
            ...     stress = from_anderson(Sh=10*ratio, SH=20, Sv=25)
            ...     return stress.components
            >>>
            >>> result = analyzer.sweep_with_stress_generator(
            ...     domain, generate_stress, OutputMetric.SLIP_RATIO
            ... )
        """
        # Initialize output array
        values = np.zeros(domain.shape)
        total_points = domain.n_points
        current_point = 0

        # Iterate over all grid points
        for idx, params in domain.iter_grid_with_indices():
            # Generate stress for these parameters
            stress = stress_generator(params)

            # Get Coulomb parameters
            friction = params.get("friction", 0.6)
            cohesion = params.get("cohesion", 0.0)
            pore_pressure = params.get("pore_pressure", 0.0)

            # Perform slip analysis
            result = self._slip_analyzer.analyze(
                stress=stress,
                friction=friction,
                cohesion=cohesion,
                pore_pressure=pore_pressure,
            )

            # Extract and store metric
            values[idx] = self._extract_metric(result, metric)

            # Progress callback
            current_point += 1
            if progress_callback is not None:
                progress_callback(current_point, total_points)

        return DomainResult(
            values=values,
            axes=domain.axes,
            axis_names=domain.axis_names,
            metric=metric,
        )
