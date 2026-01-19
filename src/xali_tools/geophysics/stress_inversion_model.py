"""
Stress inversion model for multiple points with multiple stress sources.

Each point has:
- n source stresses (e.g., from different faults or loading conditions)
- Observed data (joint normal, stylolite normal, SHmax direction + R, etc.)
- A data type that determines the cost function

The inversion finds optimal weights w = [w1, w2, ..., w_n] such that
the weighted stress at each point best fits the observations.

At each point: weighted_stress = Σ(w_i * stress_i)

For pressure inversion (e.g., dyke trajectories), the model extends to:
    weighted_stress = Σ(αᵢ × σ_tectonic_i) + R_v × σ_gravity + Σ(Pⱼ × σ_pressure_j)

Where:
- αᵢ: tectonic stress weights (can be negative)
- R_v: density ratio parameter (must be >= 0)
- Pⱼ: magma pressure parameters (must be >= 0)
"""

import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum

from .joint import cost_single_joint
from .stylolite import cost_single_stylolite
from .stress_data import cost_direction, cost_stress_ratio


class DataType(Enum):
    """Types of geological/geophysical data."""
    JOINT = "joint"                    # Joint/fracture normal -> aligned with S3
    STYLOLITE = "stylolite"            # Stylolite normal -> aligned with S1
    STRESS_DIRECTION = "stress_direction"  # Principal direction observation
    STRESS_RATIO = "stress_ratio"      # R = (S2-S3)/(S1-S3) observation
    STRESS_DIRECTION_AND_RATIO = "stress_direction_and_ratio"  # Both


@dataclass
class PointData:
    """Data for a single observation point."""
    stresses: np.ndarray          # Shape (n_sources, 6): source stresses at this point
    data_type: DataType           # Type of observation
    observed_data: Dict           # Observed data (depends on type)
    weight: float = 1.0           # Weight for this point in total cost

    def __post_init__(self):
        self.stresses = np.asarray(self.stresses, dtype=np.float64)
        if self.stresses.ndim == 1:
            self.stresses = self.stresses.reshape(1, -1)


@dataclass
class InversionModelResult:
    """Result of stress inversion model."""
    best_weights: np.ndarray       # Optimal weights for stress sources
    best_cost: float               # Minimum cost achieved
    n_iterations: int              # Number of iterations
    costs_per_point: np.ndarray    # Cost at each point with best weights
    weighted_stresses: np.ndarray  # Resulting stress at each point (n_points, 6)
    all_costs: np.ndarray = None   # Full cost history (optional)


class StressInversionModel:
    """
    Model for inverting stress from multiple observation points.

    Each point has multiple source stresses. The inversion finds optimal
    weights to combine these stresses to best fit the observations.

    Example:
        model = StressInversionModel(n_sources=6)

        # Add joint observation
        model.add_joint(
            stresses=source_stresses_at_point1,  # Shape (6, 6)
            normal=[0, 0, 1]
        )

        # Add stylolite observation
        model.add_stylolite(
            stresses=source_stresses_at_point2,
            normal=[1, 0, 0]
        )

        # Add stress data observation
        model.add_stress_direction_and_ratio(
            stresses=source_stresses_at_point3,
            direction=[1, 0, 0],
            R=0.5,
            principal_index=0
        )

        # Run inversion
        result = model.run(n_iterations=10000)
        print(result.best_weights)
    """

    def __init__(self, n_sources: int = 6):
        """
        Initialize the model.

        Args:
            n_sources: Number of stress sources (weights to find).
        """
        self.n_sources = n_sources
        self.points: List[PointData] = []

    def add_point(self, stresses: np.ndarray, data_type: DataType,
                  observed_data: Dict, weight: float = 1.0) -> None:
        """
        Add an observation point.

        Args:
            stresses: Array of shape (n_sources, 6) with source stresses.
            data_type: Type of observation data.
            observed_data: Dictionary with observation data.
            weight: Weight for this point in total cost.
        """
        stresses = np.asarray(stresses, dtype=np.float64)
        if stresses.shape[0] != self.n_sources:
            raise ValueError(
                f"Expected {self.n_sources} source stresses, got {stresses.shape[0]}"
            )
        if stresses.shape[1] != 6:
            raise ValueError(
                f"Each stress must have 6 components, got {stresses.shape[1]}"
            )

        self.points.append(PointData(
            stresses=stresses,
            data_type=data_type,
            observed_data=observed_data,
            weight=weight
        ))

    def add_joint(self, stresses: np.ndarray, normal: np.ndarray,
                  weight: float = 1.0) -> None:
        """
        Add a joint/fracture observation.

        Args:
            stresses: Array of shape (n_sources, 6) with source stresses.
            normal: Joint normal vector [nx, ny, nz].
            weight: Weight for this point.
        """
        self.add_point(
            stresses=stresses,
            data_type=DataType.JOINT,
            observed_data={"normal": np.asarray(normal, dtype=np.float64)},
            weight=weight
        )

    def add_stylolite(self, stresses: np.ndarray, normal: np.ndarray,
                      weight: float = 1.0) -> None:
        """
        Add a stylolite observation.

        Args:
            stresses: Array of shape (n_sources, 6) with source stresses.
            normal: Stylolite normal vector [nx, ny, nz].
            weight: Weight for this point.
        """
        self.add_point(
            stresses=stresses,
            data_type=DataType.STYLOLITE,
            observed_data={"normal": np.asarray(normal, dtype=np.float64)},
            weight=weight
        )

    def add_stress_direction(self, stresses: np.ndarray, direction: np.ndarray,
                             principal_index: int = 0, weight: float = 1.0) -> None:
        """
        Add a principal stress direction observation.

        Args:
            stresses: Array of shape (n_sources, 6) with source stresses.
            direction: Observed principal direction [nx, ny, nz].
            principal_index: Which principal stress (0=S1, 1=S2, 2=S3).
            weight: Weight for this point.
        """
        self.add_point(
            stresses=stresses,
            data_type=DataType.STRESS_DIRECTION,
            observed_data={
                "direction": np.asarray(direction, dtype=np.float64),
                "principal_index": principal_index
            },
            weight=weight
        )

    def add_stress_ratio(self, stresses: np.ndarray, R: float,
                         weight: float = 1.0) -> None:
        """
        Add a stress ratio observation.

        Args:
            stresses: Array of shape (n_sources, 6) with source stresses.
            R: Observed stress ratio (S2-S3)/(S1-S3).
            weight: Weight for this point.
        """
        self.add_point(
            stresses=stresses,
            data_type=DataType.STRESS_RATIO,
            observed_data={"R": float(R)},
            weight=weight
        )

    def add_stress_direction_and_ratio(self, stresses: np.ndarray,
                                       direction: np.ndarray, R: float,
                                       principal_index: int = 0,
                                       weight_direction: float = 0.5,
                                       weight_ratio: float = 0.5,
                                       weight: float = 1.0) -> None:
        """
        Add a combined direction + ratio observation.

        Args:
            stresses: Array of shape (n_sources, 6) with source stresses.
            direction: Observed principal direction [nx, ny, nz].
            R: Observed stress ratio (S2-S3)/(S1-S3).
            principal_index: Which principal stress (0=S1, 1=S2, 2=S3).
            weight_direction: Internal weight for direction vs ratio.
            weight_ratio: Internal weight for ratio vs direction.
            weight: Weight for this point.
        """
        self.add_point(
            stresses=stresses,
            data_type=DataType.STRESS_DIRECTION_AND_RATIO,
            observed_data={
                "direction": np.asarray(direction, dtype=np.float64),
                "R": float(R),
                "principal_index": principal_index,
                "weight_direction": weight_direction,
                "weight_ratio": weight_ratio
            },
            weight=weight
        )

    def compute_weighted_stress(self, weights: np.ndarray,
                                point: PointData) -> np.ndarray:
        """
        Compute weighted stress at a point.

        Args:
            weights: Array of n_sources weights.
            point: Point data with source stresses.

        Returns:
            Weighted stress (6 components).
        """
        return np.sum(weights[:, np.newaxis] * point.stresses, axis=0)

    def compute_point_cost(self, weighted_stress: np.ndarray,
                           point: PointData) -> float:
        """
        Compute cost for a single point given weighted stress.

        Args:
            weighted_stress: Combined stress (6 components).
            point: Point data with observation.

        Returns:
            Cost value.
        """
        data = point.observed_data
        dtype = point.data_type

        if dtype == DataType.JOINT:
            return cost_single_joint(data["normal"], weighted_stress)

        elif dtype == DataType.STYLOLITE:
            return cost_single_stylolite(data["normal"], weighted_stress)

        elif dtype == DataType.STRESS_DIRECTION:
            return cost_direction(
                data["direction"],
                weighted_stress,
                data["principal_index"]
            )

        elif dtype == DataType.STRESS_RATIO:
            return cost_stress_ratio(data["R"], weighted_stress)

        elif dtype == DataType.STRESS_DIRECTION_AND_RATIO:
            c_dir = cost_direction(
                data["direction"],
                weighted_stress,
                data["principal_index"]
            )
            c_ratio = cost_stress_ratio(data["R"], weighted_stress)
            w_dir = data["weight_direction"]
            w_ratio = data["weight_ratio"]
            total = w_dir + w_ratio
            return (w_dir * c_dir + w_ratio * c_ratio) / total

        else:
            raise ValueError(f"Unknown data type: {dtype}")

    def compute_total_cost(self, weights: np.ndarray) -> float:
        """
        Compute total cost for given weights.

        Args:
            weights: Array of n_sources weights.

        Returns:
            Weighted sum of costs across all points.
        """
        total_cost = 0.0
        total_weight = 0.0

        for point in self.points:
            weighted_stress = self.compute_weighted_stress(weights, point)
            cost = self.compute_point_cost(weighted_stress, point)
            total_cost += point.weight * cost
            total_weight += point.weight

        return total_cost / total_weight if total_weight > 0 else 0.0

    def run(self, n_iterations: int = 10000,
            weight_range: tuple = (-1, 1),
            seed: int = None,
            store_history: bool = False,
            progress_callback: Callable[[int, float], None] = None
            ) -> InversionModelResult:
        """
        Run Monte Carlo inversion.

        Args:
            n_iterations: Number of random weight combinations to try.
            weight_range: (min, max) range for weight values.
            seed: Random seed for reproducibility.
            store_history: If True, store all costs.
            progress_callback: Optional callback(iteration, best_cost).

        Returns:
            InversionModelResult with optimal weights and diagnostics.
        """
        if len(self.points) == 0:
            raise ValueError("No observation points added")

        if seed is not None:
            np.random.seed(seed)

        best_weights = None
        best_cost = float('inf')
        all_costs = [] if store_history else None

        for i in range(n_iterations):
            # Generate random weights (same as random stress components)
            weights = np.random.uniform(weight_range[0], weight_range[1],
                                        self.n_sources)

            cost = self.compute_total_cost(weights)

            if store_history:
                all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_weights = weights.copy()

            if progress_callback is not None and (i + 1) % 1000 == 0:
                progress_callback(i + 1, best_cost)

        # Compute per-point costs and weighted stresses for best weights
        n_points = len(self.points)
        costs_per_point = np.zeros(n_points)
        weighted_stresses = np.zeros((n_points, 6))

        for j, point in enumerate(self.points):
            ws = self.compute_weighted_stress(best_weights, point)
            weighted_stresses[j] = ws
            costs_per_point[j] = self.compute_point_cost(ws, point)

        return InversionModelResult(
            best_weights=best_weights,
            best_cost=best_cost,
            n_iterations=n_iterations,
            costs_per_point=costs_per_point,
            weighted_stresses=weighted_stresses,
            all_costs=np.array(all_costs) if store_history else None
        )

    def __repr__(self) -> str:
        type_counts = {}
        for p in self.points:
            t = p.data_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        return (f"StressInversionModel(n_sources={self.n_sources}, "
                f"n_points={len(self.points)}, types={type_counts})")


# =============================================================================
# Pressure Inversion Model
# =============================================================================

@dataclass
class PressurePointData:
    """
    Data for a single observation point with pressure sources.

    The combined stress at this point is:
        σ = Σ(αᵢ × σ_tectonic_i) + Σ(R_v × σ_gravity_j + Pⱼ × σ_pressure_j)
    """
    tectonic_stresses: np.ndarray      # Shape (n_tectonic, 6)
    gravity_stresses: Optional[np.ndarray]   # Shape (n_pressure, 6) or None
    pressure_stresses: Optional[np.ndarray]  # Shape (n_pressure, 6) or None
    data_type: DataType
    observed_data: Dict
    weight: float = 1.0

    def __post_init__(self):
        self.tectonic_stresses = np.asarray(self.tectonic_stresses, dtype=np.float64)
        if self.tectonic_stresses.ndim == 1:
            self.tectonic_stresses = self.tectonic_stresses.reshape(1, -1)

        if self.gravity_stresses is not None:
            self.gravity_stresses = np.asarray(self.gravity_stresses, dtype=np.float64)
            if self.gravity_stresses.ndim == 1:
                self.gravity_stresses = self.gravity_stresses.reshape(1, -1)

        if self.pressure_stresses is not None:
            self.pressure_stresses = np.asarray(self.pressure_stresses, dtype=np.float64)
            if self.pressure_stresses.ndim == 1:
                self.pressure_stresses = self.pressure_stresses.reshape(1, -1)


@dataclass
class PressureInversionResult:
    """Result of pressure inversion model."""
    tectonic_weights: np.ndarray   # Optimal weights for tectonic sources (n_tectonic,)
    density_ratio: Optional[float]  # R_v density ratio (None if no gravity)
    pressures: Optional[np.ndarray]  # Optimal pressures (n_pressure,) or None
    best_cost: float               # Minimum cost achieved
    n_iterations: int              # Number of iterations
    costs_per_point: np.ndarray    # Cost at each point with best parameters
    weighted_stresses: np.ndarray  # Resulting stress at each point (n_points, 6)
    all_costs: Optional[np.ndarray] = None  # Full cost history (optional)

    @property
    def all_parameters(self) -> np.ndarray:
        """Return all parameters as a single array."""
        params = list(self.tectonic_weights)
        if self.density_ratio is not None:
            params.append(self.density_ratio)
        if self.pressures is not None:
            params.extend(self.pressures)
        return np.array(params)


class PressureInversionModel:
    """
    Model for inverting stress with pressure sources (e.g., for dyke trajectories).

    The combined stress at each point is:
        σ = Σ(αᵢ × σ_tectonic_i) + Σ(R_v × σ_gravity_j + Pⱼ × σ_pressure_j)

    Each pressure source j has an associated gravity stress field.

    Parameters to invert:
    - αᵢ: tectonic stress weights (can be negative, range specified)
    - R_v: density ratio (must be >= 0)
    - Pⱼ: magma pressures (can be negative)

    Example:
        model = PressureInversionModel(
            n_tectonic=6,
            include_gravity=True,
            n_pressure=2
        )

        # Add dyke observation (dyke normal aligned with S3)
        model.add_joint(
            tectonic_stresses=tectonic_at_point,   # (6, 6)
            gravity_stresses=gravity_at_point,     # (2, 6) - one per pressure source
            pressure_stresses=pressure_at_point,   # (2, 6)
            normal=dyke_normal
        )

        result = model.run(n_iterations=50000)
        print(f"Tectonic weights: {result.tectonic_weights}")
        print(f"Density ratio R_v: {result.density_ratio}")
        print(f"Pressures: {result.pressures}")
    """

    def __init__(self, n_tectonic: int = 6, include_gravity: bool = False,
                 n_pressure: int = 0):
        """
        Initialize the pressure inversion model.

        Args:
            n_tectonic: Number of tectonic stress sources.
            include_gravity: Whether to include gravity/lithostatic stress.
            n_pressure: Number of pressure sources (magma chambers).
        """
        self.n_tectonic = n_tectonic
        self.include_gravity = include_gravity
        self.n_pressure = n_pressure
        self.points: List[PressurePointData] = []

    @property
    def n_parameters(self) -> int:
        """Total number of parameters to invert."""
        return self.n_tectonic + (1 if self.include_gravity else 0) + self.n_pressure

    def add_point(self, tectonic_stresses: np.ndarray,
                  gravity_stresses: Optional[np.ndarray],
                  pressure_stresses: Optional[np.ndarray],
                  data_type: DataType, observed_data: Dict,
                  weight: float = 1.0) -> None:
        """
        Add an observation point.

        Args:
            tectonic_stresses: Array of shape (n_tectonic, 6) with tectonic stresses.
            gravity_stresses: Array of shape (n_pressure, 6) with gravity stresses, or None.
            pressure_stresses: Array of shape (n_pressure, 6) with pressure stresses, or None.
            data_type: Type of observation data.
            observed_data: Dictionary with observation data.
            weight: Weight for this point in total cost.
        """
        tectonic_stresses = np.asarray(tectonic_stresses, dtype=np.float64)
        if tectonic_stresses.shape[0] != self.n_tectonic:
            raise ValueError(
                f"Expected {self.n_tectonic} tectonic stresses, got {tectonic_stresses.shape[0]}"
            )

        if self.n_pressure > 0:
            if pressure_stresses is None:
                raise ValueError("Model has pressure sources but pressure_stresses is None")
            pressure_stresses = np.asarray(pressure_stresses, dtype=np.float64)
            if pressure_stresses.shape[0] != self.n_pressure:
                raise ValueError(
                    f"Expected {self.n_pressure} pressure stresses, got {pressure_stresses.shape[0]}"
                )

            if self.include_gravity:
                if gravity_stresses is None:
                    raise ValueError("Model includes gravity but gravity_stresses is None")
                gravity_stresses = np.asarray(gravity_stresses, dtype=np.float64)
                if gravity_stresses.shape[0] != self.n_pressure:
                    raise ValueError(
                        f"gravity_stresses must have same count as pressure_stresses "
                        f"({self.n_pressure}), got {gravity_stresses.shape[0]}"
                    )

        self.points.append(PressurePointData(
            tectonic_stresses=tectonic_stresses,
            gravity_stresses=gravity_stresses,
            pressure_stresses=pressure_stresses,
            data_type=data_type,
            observed_data=observed_data,
            weight=weight
        ))

    def add_joint(self, tectonic_stresses: np.ndarray,
                  normal: np.ndarray,
                  gravity_stresses: Optional[np.ndarray] = None,
                  pressure_stresses: Optional[np.ndarray] = None,
                  weight: float = 1.0) -> None:
        """
        Add a joint/fracture/dyke observation.

        Args:
            tectonic_stresses: Array of shape (n_tectonic, 6).
            normal: Joint/dyke normal vector [nx, ny, nz].
            gravity_stresses: Array of shape (n_pressure, 6) or None.
            pressure_stresses: Array of shape (n_pressure, 6) or None.
            weight: Weight for this point.
        """
        self.add_point(
            tectonic_stresses=tectonic_stresses,
            gravity_stresses=gravity_stresses,
            pressure_stresses=pressure_stresses,
            data_type=DataType.JOINT,
            observed_data={"normal": np.asarray(normal, dtype=np.float64)},
            weight=weight
        )

    def add_stylolite(self, tectonic_stresses: np.ndarray,
                      normal: np.ndarray,
                      gravity_stresses: Optional[np.ndarray] = None,
                      pressure_stresses: Optional[np.ndarray] = None,
                      weight: float = 1.0) -> None:
        """
        Add a stylolite observation.

        Args:
            tectonic_stresses: Array of shape (n_tectonic, 6).
            normal: Stylolite normal vector [nx, ny, nz].
            gravity_stresses: Array of shape (n_pressure, 6) or None.
            pressure_stresses: Array of shape (n_pressure, 6) or None.
            weight: Weight for this point.
        """
        self.add_point(
            tectonic_stresses=tectonic_stresses,
            gravity_stresses=gravity_stresses,
            pressure_stresses=pressure_stresses,
            data_type=DataType.STYLOLITE,
            observed_data={"normal": np.asarray(normal, dtype=np.float64)},
            weight=weight
        )

    def add_stress_direction_and_ratio(self, tectonic_stresses: np.ndarray,
                                       direction: np.ndarray, R: float,
                                       principal_index: int = 0,
                                       gravity_stresses: Optional[np.ndarray] = None,
                                       pressure_stresses: Optional[np.ndarray] = None,
                                       weight_direction: float = 0.5,
                                       weight_ratio: float = 0.5,
                                       weight: float = 1.0) -> None:
        """
        Add a combined direction + ratio observation.

        Args:
            tectonic_stresses: Array of shape (n_tectonic, 6).
            direction: Observed principal direction [nx, ny, nz].
            R: Observed stress ratio (S2-S3)/(S1-S3).
            principal_index: Which principal stress (0=S1, 1=S2, 2=S3).
            gravity_stresses: Array of shape (n_pressure, 6) or None.
            pressure_stresses: Array of shape (n_pressure, 6) or None.
            weight_direction: Internal weight for direction cost.
            weight_ratio: Internal weight for ratio cost.
            weight: Weight for this point.
        """
        self.add_point(
            tectonic_stresses=tectonic_stresses,
            gravity_stresses=gravity_stresses,
            pressure_stresses=pressure_stresses,
            data_type=DataType.STRESS_DIRECTION_AND_RATIO,
            observed_data={
                "direction": np.asarray(direction, dtype=np.float64),
                "R": float(R),
                "principal_index": principal_index,
                "weight_direction": weight_direction,
                "weight_ratio": weight_ratio
            },
            weight=weight
        )

    def compute_weighted_stress(self, tectonic_weights: np.ndarray,
                                density_ratio: float,
                                pressures: np.ndarray,
                                point: PressurePointData) -> np.ndarray:
        """
        Compute combined stress at a point.

        σ = Σ(αᵢ × σ_tectonic_i) + Σ(R_v × σ_gravity_j + Pⱼ × σ_pressure_j)

        Args:
            tectonic_weights: Array of n_tectonic weights.
            density_ratio: R_v density ratio.
            pressures: Array of n_pressure pressures.
            point: Point data.

        Returns:
            Combined stress (6 components).
        """
        # Tectonic contribution
        stress = np.sum(tectonic_weights[:, np.newaxis] * point.tectonic_stresses, axis=0)

        # Gravity + Pressure contributions (paired)
        if point.pressure_stresses is not None and len(pressures) > 0:
            # Pressure contribution: Σ(Pⱼ × σ_pressure_j)
            stress = stress + np.sum(pressures[:, np.newaxis] * point.pressure_stresses, axis=0)

            # Gravity contribution: R_v × Σ(σ_gravity_j)
            if point.gravity_stresses is not None:
                stress = stress + density_ratio * np.sum(point.gravity_stresses, axis=0)

        return stress

    def compute_point_cost(self, weighted_stress: np.ndarray,
                           point: PressurePointData) -> float:
        """
        Compute cost for a single point given weighted stress.
        """
        data = point.observed_data
        dtype = point.data_type

        if dtype == DataType.JOINT:
            return cost_single_joint(data["normal"], weighted_stress)

        elif dtype == DataType.STYLOLITE:
            return cost_single_stylolite(data["normal"], weighted_stress)

        elif dtype == DataType.STRESS_DIRECTION:
            return cost_direction(
                data["direction"],
                weighted_stress,
                data["principal_index"]
            )

        elif dtype == DataType.STRESS_RATIO:
            return cost_stress_ratio(data["R"], weighted_stress)

        elif dtype == DataType.STRESS_DIRECTION_AND_RATIO:
            c_dir = cost_direction(
                data["direction"],
                weighted_stress,
                data["principal_index"]
            )
            c_ratio = cost_stress_ratio(data["R"], weighted_stress)
            w_dir = data["weight_direction"]
            w_ratio = data["weight_ratio"]
            total = w_dir + w_ratio
            return (w_dir * c_dir + w_ratio * c_ratio) / total

        else:
            raise ValueError(f"Unknown data type: {dtype}")

    def compute_total_cost(self, tectonic_weights: np.ndarray,
                           density_ratio: float,
                           pressures: np.ndarray) -> float:
        """
        Compute total cost for given parameters.
        """
        total_cost = 0.0
        total_weight = 0.0

        for point in self.points:
            weighted_stress = self.compute_weighted_stress(
                tectonic_weights, density_ratio, pressures, point
            )
            cost = self.compute_point_cost(weighted_stress, point)
            total_cost += point.weight * cost
            total_weight += point.weight

        return total_cost / total_weight if total_weight > 0 else 0.0

    def run(self, n_iterations: int = 10000,
            tectonic_range: tuple = (-1, 1),
            density_range: tuple = (0, 2),
            pressure_range: tuple = (0, 100),
            seed: int = None,
            store_history: bool = False,
            progress_callback: Callable[[int, float], None] = None
            ) -> PressureInversionResult:
        """
        Run Monte Carlo inversion.

        Args:
            n_iterations: Number of random parameter combinations to try.
            tectonic_range: (min, max) range for tectonic weights.
            density_range: (min, max) range for R_v (typically >= 0).
            pressure_range: (min, max) range for pressures (typically >= 0).
            seed: Random seed for reproducibility.
            store_history: If True, store all costs.
            progress_callback: Optional callback(iteration, best_cost).

        Returns:
            PressureInversionResult with optimal parameters and diagnostics.
        """
        if len(self.points) == 0:
            raise ValueError("No observation points added")

        if seed is not None:
            np.random.seed(seed)

        best_tectonic = None
        best_density = 0.0
        best_pressures = np.array([])
        best_cost = float('inf')
        all_costs = [] if store_history else None

        for i in range(n_iterations):
            # Generate random tectonic weights
            tectonic_weights = np.random.uniform(
                tectonic_range[0], tectonic_range[1], self.n_tectonic
            )

            # Generate random density ratio (if gravity included)
            if self.include_gravity:
                density_ratio = np.random.uniform(density_range[0], density_range[1])
            else:
                density_ratio = 0.0

            # Generate random pressures (if pressure sources)
            if self.n_pressure > 0:
                pressures = np.random.uniform(
                    pressure_range[0], pressure_range[1], self.n_pressure
                )
            else:
                pressures = np.array([])

            cost = self.compute_total_cost(tectonic_weights, density_ratio, pressures)

            if store_history:
                all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_tectonic = tectonic_weights.copy()
                best_density = density_ratio
                best_pressures = pressures.copy() if len(pressures) > 0 else np.array([])

            if progress_callback is not None and (i + 1) % 1000 == 0:
                progress_callback(i + 1, best_cost)

        # Compute per-point costs and weighted stresses for best parameters
        n_points = len(self.points)
        costs_per_point = np.zeros(n_points)
        weighted_stresses = np.zeros((n_points, 6))

        for j, point in enumerate(self.points):
            ws = self.compute_weighted_stress(
                best_tectonic, best_density, best_pressures, point
            )
            weighted_stresses[j] = ws
            costs_per_point[j] = self.compute_point_cost(ws, point)

        return PressureInversionResult(
            tectonic_weights=best_tectonic,
            density_ratio=best_density if self.include_gravity else None,
            pressures=best_pressures if self.n_pressure > 0 else None,
            best_cost=best_cost,
            n_iterations=n_iterations,
            costs_per_point=costs_per_point,
            weighted_stresses=weighted_stresses,
            all_costs=np.array(all_costs) if store_history else None
        )

    def __repr__(self) -> str:
        type_counts = {}
        for p in self.points:
            t = p.data_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        return (f"PressureInversionModel(n_tectonic={self.n_tectonic}, "
                f"gravity={self.include_gravity}, n_pressure={self.n_pressure}, "
                f"n_points={len(self.points)}, types={type_counts})")


# =============================================================================
# Direct Stress Inversion Model (using parameterizations)
# =============================================================================

from .stress_tensor import StressParameterization


@dataclass
class DirectInversionResult:
    """Result of direct stress inversion using parameterization."""
    best_params: Dict                  # Optimal parameter values
    best_stress: np.ndarray            # Optimal stress tensor (6 components)
    best_cost: float                   # Minimum cost achieved
    n_iterations: int                  # Number of iterations
    costs_per_point: np.ndarray        # Cost at each point with best stress
    param_names: List[str]             # Parameter names for reference
    all_costs: Optional[np.ndarray] = None  # Full cost history (optional)
    all_params: Optional[List[Dict]] = None  # All sampled parameters (optional)


@dataclass
class DirectPointData:
    """Data for a single observation point (no source stresses needed)."""
    data_type: DataType
    observed_data: Dict
    weight: float = 1.0


class DirectStressInversionModel:
    """
    Model for directly inverting stress tensor using parameterizations.

    Unlike StressInversionModel which combines pre-computed source stresses,
    this model directly searches for the optimal stress tensor in a parameter
    space defined by a StressParameterization (Anderson, Principal, etc.).

    Example:
        from xali_tools.geophysics import (
            DirectStressInversionModel, AndersonParameterization
        )

        model = DirectStressInversionModel()

        # Add joint observations (only need normal, no source stresses)
        model.add_joint(normal=[0, 0, 1])
        model.add_joint(normal=[0.7, 0.7, 0])

        # Add stress direction observation
        model.add_stress_direction_and_ratio(
            direction=[1, 0, 0], R=0.5, principal_index=0
        )

        # Define parameterization (Anderson model)
        param = AndersonParameterization(
            Sh_range=(5, 30),
            SH_range=(10, 50),
            Sv_range=(20, 60),
            theta_range=(0, 180)
        )

        # Run inversion
        result = model.run(param, n_iterations=50000)
        print(f"Best params: {result.best_params}")
        print(f"Best stress: {result.best_stress}")
    """

    def __init__(self):
        """Initialize the direct stress inversion model."""
        self.points: List[DirectPointData] = []

    def add_point(self, data_type: DataType, observed_data: Dict,
                  weight: float = 1.0) -> None:
        """
        Add an observation point.

        Args:
            data_type: Type of observation data.
            observed_data: Dictionary with observation data.
            weight: Weight for this point in total cost.
        """
        self.points.append(DirectPointData(
            data_type=data_type,
            observed_data=observed_data,
            weight=weight
        ))

    def add_joint(self, normal: np.ndarray, weight: float = 1.0) -> None:
        """
        Add a joint/fracture observation.

        Args:
            normal: Joint normal vector [nx, ny, nz].
            weight: Weight for this point.
        """
        self.add_point(
            data_type=DataType.JOINT,
            observed_data={"normal": np.asarray(normal, dtype=np.float64)},
            weight=weight
        )

    def add_stylolite(self, normal: np.ndarray, weight: float = 1.0) -> None:
        """
        Add a stylolite observation.

        Args:
            normal: Stylolite normal vector [nx, ny, nz].
            weight: Weight for this point.
        """
        self.add_point(
            data_type=DataType.STYLOLITE,
            observed_data={"normal": np.asarray(normal, dtype=np.float64)},
            weight=weight
        )

    def add_stress_direction(self, direction: np.ndarray,
                             principal_index: int = 0,
                             weight: float = 1.0) -> None:
        """
        Add a principal stress direction observation.

        Args:
            direction: Observed principal direction [nx, ny, nz].
            principal_index: Which principal stress (0=S1, 1=S2, 2=S3).
            weight: Weight for this point.
        """
        self.add_point(
            data_type=DataType.STRESS_DIRECTION,
            observed_data={
                "direction": np.asarray(direction, dtype=np.float64),
                "principal_index": principal_index
            },
            weight=weight
        )

    def add_stress_ratio(self, R: float, weight: float = 1.0) -> None:
        """
        Add a stress ratio observation.

        Args:
            R: Observed stress ratio (S2-S3)/(S1-S3).
            weight: Weight for this point.
        """
        self.add_point(
            data_type=DataType.STRESS_RATIO,
            observed_data={"R": float(R)},
            weight=weight
        )

    def add_stress_direction_and_ratio(self, direction: np.ndarray, R: float,
                                       principal_index: int = 0,
                                       weight_direction: float = 0.5,
                                       weight_ratio: float = 0.5,
                                       weight: float = 1.0) -> None:
        """
        Add a combined direction + ratio observation.

        Args:
            direction: Observed principal direction [nx, ny, nz].
            R: Observed stress ratio (S2-S3)/(S1-S3).
            principal_index: Which principal stress (0=S1, 1=S2, 2=S3).
            weight_direction: Internal weight for direction vs ratio.
            weight_ratio: Internal weight for ratio vs direction.
            weight: Weight for this point.
        """
        self.add_point(
            data_type=DataType.STRESS_DIRECTION_AND_RATIO,
            observed_data={
                "direction": np.asarray(direction, dtype=np.float64),
                "R": float(R),
                "principal_index": principal_index,
                "weight_direction": weight_direction,
                "weight_ratio": weight_ratio
            },
            weight=weight
        )

    def compute_point_cost(self, stress: np.ndarray,
                           point: DirectPointData) -> float:
        """
        Compute cost for a single point given stress tensor.

        Args:
            stress: Stress tensor (6 components).
            point: Point data with observation.

        Returns:
            Cost value.
        """
        data = point.observed_data
        dtype = point.data_type

        if dtype == DataType.JOINT:
            return cost_single_joint(data["normal"], stress)

        elif dtype == DataType.STYLOLITE:
            return cost_single_stylolite(data["normal"], stress)

        elif dtype == DataType.STRESS_DIRECTION:
            return cost_direction(
                data["direction"],
                stress,
                data["principal_index"]
            )

        elif dtype == DataType.STRESS_RATIO:
            return cost_stress_ratio(data["R"], stress)

        elif dtype == DataType.STRESS_DIRECTION_AND_RATIO:
            c_dir = cost_direction(
                data["direction"],
                stress,
                data["principal_index"]
            )
            c_ratio = cost_stress_ratio(data["R"], stress)
            w_dir = data["weight_direction"]
            w_ratio = data["weight_ratio"]
            total = w_dir + w_ratio
            return (w_dir * c_dir + w_ratio * c_ratio) / total

        else:
            raise ValueError(f"Unknown data type: {dtype}")

    def compute_total_cost(self, stress: np.ndarray) -> float:
        """
        Compute total cost for given stress tensor.

        Args:
            stress: Stress tensor (6 components).

        Returns:
            Weighted sum of costs across all points.
        """
        total_cost = 0.0
        total_weight = 0.0

        for point in self.points:
            cost = self.compute_point_cost(stress, point)
            total_cost += point.weight * cost
            total_weight += point.weight

        return total_cost / total_weight if total_weight > 0 else 0.0

    def run(self, parameterization: StressParameterization,
            n_iterations: int = 10000,
            seed: int = None,
            store_history: bool = False,
            progress_callback: Callable[[int, float], None] = None
            ) -> DirectInversionResult:
        """
        Run Monte Carlo inversion using the specified parameterization.

        Args:
            parameterization: StressParameterization defining the parameter space.
            n_iterations: Number of random parameter combinations to try.
            seed: Random seed for reproducibility.
            store_history: If True, store all costs and parameters.
            progress_callback: Optional callback(iteration, best_cost).

        Returns:
            DirectInversionResult with optimal parameters and diagnostics.
        """
        if len(self.points) == 0:
            raise ValueError("No observation points added")

        rng = np.random.default_rng(seed)

        best_params = None
        best_stress = None
        best_cost = float('inf')
        all_costs = [] if store_history else None
        all_params = [] if store_history else None

        for i in range(n_iterations):
            # Sample parameters from the parameterization
            params = parameterization.sample(rng)
            stress = parameterization.to_stress(params)

            cost = self.compute_total_cost(stress)

            if store_history:
                all_costs.append(cost)
                all_params.append(params.copy())

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
                best_stress = stress.copy()

            if progress_callback is not None and (i + 1) % 1000 == 0:
                progress_callback(i + 1, best_cost)

        # Compute per-point costs for best stress
        n_points = len(self.points)
        costs_per_point = np.zeros(n_points)

        for j, point in enumerate(self.points):
            costs_per_point[j] = self.compute_point_cost(best_stress, point)

        return DirectInversionResult(
            best_params=best_params,
            best_stress=best_stress,
            best_cost=best_cost,
            n_iterations=n_iterations,
            costs_per_point=costs_per_point,
            param_names=parameterization.param_names,
            all_costs=np.array(all_costs) if store_history else None,
            all_params=all_params if store_history else None
        )

    def __repr__(self) -> str:
        type_counts = {}
        for p in self.points:
            t = p.data_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        return f"DirectStressInversionModel(n_points={len(self.points)}, types={type_counts})"
