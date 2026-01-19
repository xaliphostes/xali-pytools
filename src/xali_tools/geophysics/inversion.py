"""
Monte Carlo stress inversion methods.

Generic inversion that works with any cost function.

Supports:
- General 3D stress (6 parameters): σxx, σxy, σxz, σyy, σyz, σzz
- Andersonian stress (4 parameters): σxx, σxy, σyy, σzz (with σxz = σyz = 0)
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass

from .stress_utils import (
    principal_directions,
    generate_random_stress,
    generate_random_andersonian_stress,
    andersonian_4_to_6,
    stress_6_to_andersonian_4
)


@dataclass
class InversionResult:
    """Result of stress inversion."""
    best_stress: np.ndarray          # Best 6-component stress tensor
    best_cost: float                  # Minimum cost achieved
    principal_values: np.ndarray     # [σ1, σ2, σ3]
    principal_directions: np.ndarray  # (3, 3) matrix, rows are directions
    n_iterations: int                 # Number of Monte Carlo iterations
    andersonian: bool                # Whether Andersonian assumption was used
    all_costs: Optional[np.ndarray] = None   # Cost history (if stored)
    all_stresses: Optional[np.ndarray] = None  # Stress history (if stored)

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (4 for Andersonian, 6 for general)."""
        return 4 if self.andersonian else 6

    @property
    def andersonian_params(self) -> Optional[np.ndarray]:
        """Return 4 Andersonian parameters [σxx, σxy, σyy, σzz] if applicable."""
        if self.andersonian:
            return stress_6_to_andersonian_4(self.best_stress)
        return None


def monte_carlo_inversion(
    cost_function: Callable[[np.ndarray], float],
    n_iterations: int = 10000,
    sigma_range: Tuple[float, float] = (-1, 1),
    andersonian: bool = False,
    store_history: bool = False,
    seed: int = None,
    progress_callback: Callable[[int, float], None] = None
) -> InversionResult:
    """
    Find the best stress tensor using Monte Carlo sampling.

    Args:
        cost_function: Function that takes a 6-component stress array and returns a cost.
        n_iterations: Number of random stress tensors to sample.
        sigma_range: (min, max) range for stress component values.
        andersonian: If True, use Andersonian stress (4 params: σxx, σxy, σyy, σzz).
                     If False, use general 3D stress (6 params).
        store_history: If True, store all sampled stresses and costs.
        seed: Random seed for reproducibility.
        progress_callback: Optional callback(iteration, current_best_cost).

    Returns:
        InversionResult with best stress tensor and diagnostics.

    Example (general 3D stress):
        from xali_tools.geophysics.joint import cost_multiple_joints
        from xali_tools.geophysics.inversion import monte_carlo_inversion

        normals = np.array([[0, 0, 1], [0.1, 0, 0.99]])
        cost_fn = lambda stress: cost_multiple_joints(normals, stress)
        result = monte_carlo_inversion(cost_fn, n_iterations=10000)

    Example (Andersonian stress - one principal axis vertical):
        result = monte_carlo_inversion(cost_fn, n_iterations=10000, andersonian=True)
        print(f"Parameters: {result.andersonian_params}")  # [σxx, σxy, σyy, σzz]
    """
    if seed is not None:
        np.random.seed(seed)

    best_stress = None
    best_cost = float('inf')

    all_costs = [] if store_history else None
    all_stresses = [] if store_history else None

    # Select stress generator based on assumption
    stress_generator = (generate_random_andersonian_stress if andersonian
                        else generate_random_stress)

    for i in range(n_iterations):
        # Generate random stress
        stress = stress_generator(sigma_range)

        # Compute cost
        cost = cost_function(stress)

        if store_history:
            all_costs.append(cost)
            all_stresses.append(stress.copy())

        if cost < best_cost:
            best_cost = cost
            best_stress = stress.copy()

        if progress_callback is not None and (i + 1) % 1000 == 0:
            progress_callback(i + 1, best_cost)

    # Compute principal values and directions for best stress
    values, directions = principal_directions(best_stress)

    return InversionResult(
        best_stress=best_stress,
        best_cost=best_cost,
        principal_values=values,
        principal_directions=directions,
        n_iterations=n_iterations,
        andersonian=andersonian,
        all_costs=np.array(all_costs) if store_history else None,
        all_stresses=np.array(all_stresses) if store_history else None
    )


def monte_carlo_inversion_adaptive(
    cost_function: Callable[[np.ndarray], float],
    n_iterations: int = 10000,
    sigma_range: Tuple[float, float] = (-1, 1),
    andersonian: bool = False,
    n_refinements: int = 3,
    refinement_factor: float = 0.5,
    top_fraction: float = 0.1,
    seed: int = None
) -> InversionResult:
    """
    Adaptive Monte Carlo inversion with iterative refinement.

    Starts with broad sampling, then focuses on promising regions.

    Args:
        cost_function: Function that takes a 6-component stress array and returns a cost.
        n_iterations: Number of iterations per refinement stage.
        sigma_range: Initial (min, max) range for stress component values.
        andersonian: If True, use Andersonian stress (4 params: σxx, σxy, σyy, σzz).
                     If False, use general 3D stress (6 params).
        n_refinements: Number of refinement stages.
        refinement_factor: How much to shrink search space each stage.
        top_fraction: Fraction of best samples to use for refinement.
        seed: Random seed for reproducibility.

    Returns:
        InversionResult with best stress tensor.
    """
    if seed is not None:
        np.random.seed(seed)

    best_stress = None
    best_cost = float('inf')
    current_range = sigma_range

    # Select stress generator based on assumption
    stress_generator = (generate_random_andersonian_stress if andersonian
                        else generate_random_stress)

    for stage in range(n_refinements + 1):
        stresses = []
        costs = []

        for _ in range(n_iterations):
            stress = stress_generator(current_range)
            cost = cost_function(stress)

            stresses.append(stress)
            costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_stress = stress.copy()

        if stage < n_refinements:
            # Select top performers for next stage
            costs = np.array(costs)
            stresses = np.array(stresses)
            n_top = max(1, int(len(costs) * top_fraction))
            top_idx = np.argsort(costs)[:n_top]
            top_stresses = stresses[top_idx]

            # Compute new range based on top stresses
            stress_min = top_stresses.min(axis=0)
            stress_max = top_stresses.max(axis=0)
            margin = (stress_max - stress_min) * refinement_factor
            current_range = (
                max(sigma_range[0], (stress_min - margin).min()),
                min(sigma_range[1], (stress_max + margin).max())
            )

    values, directions = principal_directions(best_stress)

    return InversionResult(
        best_stress=best_stress,
        best_cost=best_cost,
        principal_values=values,
        principal_directions=directions,
        n_iterations=n_iterations * (n_refinements + 1),
        andersonian=andersonian,
        all_costs=None,
        all_stresses=None
    )
