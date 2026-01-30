"""
Coulomb failure criterion for fault slip analysis.

The Coulomb criterion determines when a fault surface will slip based on:
- Normal stress (sigma_n)
- Shear stress (tau)
- Friction coefficient (mu)
- Cohesion (c)
- Pore pressure (Pp)

Sign convention (tension positive):
- sigma_n > 0: tension (reduces resistance to slip)
- sigma_n < 0: compression (increases resistance to slip)

Effective stress: sigma'_n = sigma_n - Pp
- When Pp > 0, effective stress becomes more tensile

Coulomb criterion (tension-positive form):
    Slip occurs when: tau > c - mu * sigma'_n

    When sigma'_n < 0 (effective compression): -mu*sigma'_n > 0, adds friction
    When sigma'_n > 0 (effective tension): -mu*sigma'_n < 0, reduces threshold
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoulombParameters:
    """
    Parameters for Coulomb failure criterion.

    Attributes:
        friction: Friction coefficient (mu), typically 0.6 for rock
        cohesion: Cohesion strength (c) in stress units (e.g., MPa)
        pore_pressure: Pore fluid pressure (Pp) in stress units
    """

    friction: float = 0.6
    cohesion: float = 0.0
    pore_pressure: float = 0.0

    def __post_init__(self):
        if self.friction < 0:
            raise ValueError("Friction coefficient must be non-negative")
        if self.cohesion < 0:
            raise ValueError("Cohesion must be non-negative")

    def effective_normal_stress(self, sigma_n: np.ndarray) -> np.ndarray:
        """
        Compute effective normal stress.

        sigma'_n = sigma_n - Pp

        Args:
            sigma_n: Total normal stress (tension positive)

        Returns:
            Effective normal stress
        """
        return sigma_n - self.pore_pressure

    def slip_threshold(self, sigma_n: np.ndarray) -> np.ndarray:
        """
        Compute the shear stress threshold for slip.

        threshold = c - mu * sigma'_n

        Under compression (sigma'_n < 0):
            threshold = c + mu * |sigma'_n| (higher threshold)

        Under tension (sigma'_n > 0):
            threshold = c - mu * sigma'_n (lower threshold, can go negative)

        Args:
            sigma_n: Total normal stress (tension positive)

        Returns:
            Threshold shear stress for slip
        """
        sigma_n_eff = self.effective_normal_stress(sigma_n)
        return self.cohesion - self.friction * sigma_n_eff

    def coulomb_stress(self, sigma_n: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Compute Coulomb stress (excess shear stress above threshold).

        CFS = tau - threshold = tau - (c - mu * sigma'_n)
            = tau - c + mu * sigma'_n

        CFS > 0 indicates slip
        CFS < 0 indicates stable

        Args:
            sigma_n: Normal stress (tension positive)
            tau: Shear stress magnitude

        Returns:
            Coulomb failure stress (positive = slip, negative = stable)
        """
        threshold = self.slip_threshold(sigma_n)
        return tau - threshold


def evaluate_coulomb(
    sigma_n: np.ndarray, tau: np.ndarray, params: CoulombParameters
) -> np.ndarray:
    """
    Evaluate Coulomb criterion for slip.

    Slip occurs when: tau > threshold = c - mu * sigma'_n

    Args:
        sigma_n: Normal stress on triangles, shape (n_triangles,)
        tau: Shear stress on triangles, shape (n_triangles,)
        params: Coulomb parameters

    Returns:
        Boolean array, True where slip occurs (tau > threshold)
    """
    sigma_n = np.asarray(sigma_n)
    tau = np.asarray(tau)

    threshold = params.slip_threshold(sigma_n)
    return tau > threshold


def coulomb_slip_tendency(
    sigma_n: np.ndarray, tau: np.ndarray, params: CoulombParameters
) -> np.ndarray:
    """
    Compute slip tendency: tau / threshold.

    Slip tendency > 1 indicates slip.
    Slip tendency = 1 is at the failure boundary.
    Slip tendency < 1 indicates stable.

    For cases where threshold <= 0 (pure tension failure),
    returns infinity if tau > 0, or 0 if tau = 0.

    Args:
        sigma_n: Normal stress, shape (n_triangles,)
        tau: Shear stress, shape (n_triangles,)
        params: Coulomb parameters

    Returns:
        Slip tendency, shape (n_triangles,)
    """
    sigma_n = np.asarray(sigma_n)
    tau = np.asarray(tau)

    threshold = params.slip_threshold(sigma_n)

    # Handle threshold <= 0 (effective tension exceeds cohesion)
    # In this case, any shear stress causes slip
    with np.errstate(divide="ignore", invalid="ignore"):
        tendency = np.where(
            threshold > 0,
            tau / threshold,
            np.where(tau > 0, np.inf, 0.0),
        )

    return tendency


def dilation_tendency(sigma_n: np.ndarray, params: CoulombParameters) -> np.ndarray:
    """
    Compute dilation tendency (normalized effective normal stress).

    Dilation tendency measures how close a fracture is to opening.
    Higher values indicate more tendency to dilate/open.

    DT = (sigma_n_max - sigma'_n) / (sigma_n_max - sigma_n_min)

    Simplified version: DT = -sigma'_n / sigma_ref (when sigma'_n < 0 = compression)

    For this implementation, we use:
    DT = -sigma'_n (negative effective normal stress indicates dilation)

    Args:
        sigma_n: Normal stress, shape (n_triangles,)
        params: Coulomb parameters

    Returns:
        Dilation tendency (higher = more likely to open)
    """
    sigma_n_eff = params.effective_normal_stress(np.asarray(sigma_n))
    # Positive dilation tendency when in tension
    return -sigma_n_eff
