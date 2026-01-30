"""
Parameter domain specification for slip envelope analysis.

Provides classes for defining parameter ranges and domain specifications
for 2D and 3D parameter sweeps.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple


class ParameterType(Enum):
    """
    Types of parameters that can be varied in domain sweeps.

    Coulomb parameters:
        FRICTION: Friction coefficient (mu)
        COHESION: Cohesion strength (c)
        PORE_PRESSURE: Pore fluid pressure (Pp)

    Stress parameters:
        STRESS_RATIO: Ratio between principal stresses
        SH: Minimum horizontal stress (Anderson)
        SH_MAX: Maximum horizontal stress (Anderson)
        SV: Vertical stress (Anderson)
        THETA: Stress orientation angle

    Environmental:
        DEPTH: Depth for lithostatic stress calculation
        TEMPERATURE: Temperature for thermal stress effects
    """

    # Coulomb parameters
    FRICTION = "friction"
    COHESION = "cohesion"
    PORE_PRESSURE = "pore_pressure"

    # Stress parameters
    STRESS_RATIO = "stress_ratio"
    SH = "Sh"
    SH_MAX = "SH"
    SV = "Sv"
    THETA = "theta"

    # Environmental
    DEPTH = "depth"
    TEMPERATURE = "temperature"


@dataclass
class ParameterRange:
    """
    Specification for a single parameter range.

    Attributes:
        param_type: Type of parameter
        min_val: Minimum value
        max_val: Maximum value
        n_steps: Number of steps (including endpoints)
    """

    param_type: ParameterType
    min_val: float
    max_val: float
    n_steps: int

    def __post_init__(self):
        if self.n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if self.min_val >= self.max_val:
            raise ValueError("min_val must be less than max_val")

    @property
    def values(self) -> np.ndarray:
        """Get array of parameter values."""
        return np.linspace(self.min_val, self.max_val, self.n_steps)

    @property
    def step_size(self) -> float:
        """Get step size between values."""
        return (self.max_val - self.min_val) / (self.n_steps - 1)

    @property
    def name(self) -> str:
        """Get parameter name string."""
        return self.param_type.value


@dataclass
class DomainSpecification:
    """
    Specification for a parameter domain (1D, 2D, or 3D).

    Defines which parameters to vary and their ranges, plus fixed values
    for parameters not being varied.

    Example:
        >>> # 2D domain: friction vs pore pressure
        >>> domain = DomainSpecification(
        ...     parameters=[
        ...         ParameterRange(ParameterType.FRICTION, 0.3, 0.9, 50),
        ...         ParameterRange(ParameterType.PORE_PRESSURE, 0, 20, 50),
        ...     ],
        ...     fixed_values={ParameterType.COHESION: 0.0}
        ... )
        >>> for params in domain.iter_grid():
        ...     print(params)  # {'friction': 0.3, 'pore_pressure': 0.0, 'cohesion': 0.0}
    """

    parameters: List[ParameterRange]
    fixed_values: Dict[ParameterType, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.parameters) == 0:
            raise ValueError("At least one parameter range required")
        if len(self.parameters) > 3:
            raise ValueError("Maximum 3 parameter ranges supported (1D, 2D, or 3D)")

        # Check for duplicate parameter types
        param_types = [p.param_type for p in self.parameters]
        if len(param_types) != len(set(param_types)):
            raise ValueError("Duplicate parameter types in ranges")

        # Check fixed values don't overlap with ranges
        for param_type in self.fixed_values:
            if param_type in param_types:
                raise ValueError(
                    f"Parameter {param_type.value} cannot be both varied and fixed"
                )

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions (1, 2, or 3)."""
        return len(self.parameters)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the output grid."""
        return tuple(p.n_steps for p in self.parameters)

    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        return int(np.prod(self.shape))

    @property
    def axes(self) -> List[np.ndarray]:
        """List of value arrays for each axis."""
        return [p.values for p in self.parameters]

    @property
    def axis_names(self) -> List[str]:
        """List of parameter names for each axis."""
        return [p.name for p in self.parameters]

    def get_meshgrid(self) -> List[np.ndarray]:
        """
        Get meshgrid arrays for all dimensions.

        Returns:
            List of meshgrid arrays, one per dimension
        """
        return list(np.meshgrid(*self.axes, indexing="ij"))

    def iter_grid(self) -> Iterator[Dict[str, float]]:
        """
        Iterate over all grid points, yielding parameter dictionaries.

        Yields:
            Dictionary mapping parameter names to values
        """
        # Get meshgrid for iteration
        grids = self.get_meshgrid()

        # Flatten for iteration
        flat_grids = [g.flatten() for g in grids]

        for i in range(self.n_points):
            params = {}
            # Add varied parameters
            for j, param_range in enumerate(self.parameters):
                params[param_range.name] = flat_grids[j][i]
            # Add fixed parameters
            for param_type, value in self.fixed_values.items():
                params[param_type.value] = value
            yield params

    def iter_grid_with_indices(self) -> Iterator[Tuple[Tuple[int, ...], Dict[str, float]]]:
        """
        Iterate over grid points, yielding indices and parameter dictionaries.

        Yields:
            Tuple of (indices, params_dict)
        """
        for idx in np.ndindex(self.shape):
            params = {}
            # Add varied parameters
            for j, param_range in enumerate(self.parameters):
                params[param_range.name] = param_range.values[idx[j]]
            # Add fixed parameters
            for param_type, value in self.fixed_values.items():
                params[param_type.value] = value
            yield idx, params


# Factory functions for common domain specifications


def friction_pore_pressure_domain(
    friction_range: Tuple[float, float] = (0.3, 0.9),
    friction_steps: int = 50,
    pore_pressure_range: Tuple[float, float] = (0.0, 20.0),
    pore_pressure_steps: int = 50,
    cohesion: float = 0.0,
) -> DomainSpecification:
    """
    Create a 2D domain varying friction and pore pressure.

    Args:
        friction_range: (min, max) friction coefficient
        friction_steps: Number of friction steps
        pore_pressure_range: (min, max) pore pressure
        pore_pressure_steps: Number of pore pressure steps
        cohesion: Fixed cohesion value

    Returns:
        DomainSpecification for friction vs pore pressure
    """
    return DomainSpecification(
        parameters=[
            ParameterRange(
                ParameterType.FRICTION,
                friction_range[0],
                friction_range[1],
                friction_steps,
            ),
            ParameterRange(
                ParameterType.PORE_PRESSURE,
                pore_pressure_range[0],
                pore_pressure_range[1],
                pore_pressure_steps,
            ),
        ],
        fixed_values={ParameterType.COHESION: cohesion},
    )


def friction_cohesion_domain(
    friction_range: Tuple[float, float] = (0.3, 0.9),
    friction_steps: int = 50,
    cohesion_range: Tuple[float, float] = (0.0, 10.0),
    cohesion_steps: int = 50,
    pore_pressure: float = 0.0,
) -> DomainSpecification:
    """
    Create a 2D domain varying friction and cohesion.

    Args:
        friction_range: (min, max) friction coefficient
        friction_steps: Number of friction steps
        cohesion_range: (min, max) cohesion
        cohesion_steps: Number of cohesion steps
        pore_pressure: Fixed pore pressure value

    Returns:
        DomainSpecification for friction vs cohesion
    """
    return DomainSpecification(
        parameters=[
            ParameterRange(
                ParameterType.FRICTION,
                friction_range[0],
                friction_range[1],
                friction_steps,
            ),
            ParameterRange(
                ParameterType.COHESION,
                cohesion_range[0],
                cohesion_range[1],
                cohesion_steps,
            ),
        ],
        fixed_values={ParameterType.PORE_PRESSURE: pore_pressure},
    )


def stress_ratio_friction_domain(
    stress_ratio_range: Tuple[float, float] = (0.5, 2.0),
    stress_ratio_steps: int = 50,
    friction_range: Tuple[float, float] = (0.3, 0.9),
    friction_steps: int = 50,
    cohesion: float = 0.0,
    pore_pressure: float = 0.0,
) -> DomainSpecification:
    """
    Create a 2D domain varying stress ratio and friction.

    Args:
        stress_ratio_range: (min, max) stress ratio
        stress_ratio_steps: Number of stress ratio steps
        friction_range: (min, max) friction coefficient
        friction_steps: Number of friction steps
        cohesion: Fixed cohesion value
        pore_pressure: Fixed pore pressure value

    Returns:
        DomainSpecification for stress ratio vs friction
    """
    return DomainSpecification(
        parameters=[
            ParameterRange(
                ParameterType.STRESS_RATIO,
                stress_ratio_range[0],
                stress_ratio_range[1],
                stress_ratio_steps,
            ),
            ParameterRange(
                ParameterType.FRICTION,
                friction_range[0],
                friction_range[1],
                friction_steps,
            ),
        ],
        fixed_values={
            ParameterType.COHESION: cohesion,
            ParameterType.PORE_PRESSURE: pore_pressure,
        },
    )


def friction_cohesion_pore_pressure_domain(
    friction_range: Tuple[float, float] = (0.3, 0.9),
    friction_steps: int = 20,
    cohesion_range: Tuple[float, float] = (0.0, 10.0),
    cohesion_steps: int = 20,
    pore_pressure_range: Tuple[float, float] = (0.0, 20.0),
    pore_pressure_steps: int = 20,
) -> DomainSpecification:
    """
    Create a 3D domain varying friction, cohesion, and pore pressure.

    Args:
        friction_range: (min, max) friction coefficient
        friction_steps: Number of friction steps
        cohesion_range: (min, max) cohesion
        cohesion_steps: Number of cohesion steps
        pore_pressure_range: (min, max) pore pressure
        pore_pressure_steps: Number of pore pressure steps

    Returns:
        DomainSpecification for 3D friction/cohesion/pore_pressure cube
    """
    return DomainSpecification(
        parameters=[
            ParameterRange(
                ParameterType.FRICTION,
                friction_range[0],
                friction_range[1],
                friction_steps,
            ),
            ParameterRange(
                ParameterType.COHESION,
                cohesion_range[0],
                cohesion_range[1],
                cohesion_steps,
            ),
            ParameterRange(
                ParameterType.PORE_PRESSURE,
                pore_pressure_range[0],
                pore_pressure_range[1],
                pore_pressure_steps,
            ),
        ]
    )
