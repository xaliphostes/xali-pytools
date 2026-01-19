"""
Stress tensor utilities for converting user-friendly stress parameterizations
to the internal 6-component format [Sxx, Sxy, Sxz, Syy, Syz, Szz].

Provides two main parameterizations:
1. Anderson model: Sh, SH, Sv, theta (with optional lithostatic loading)
2. General case: S1, S2, S3 with Euler angles (alpha, beta, gamma)

Coordinate system convention:
- x: East
- y: North
- z: Up (vertical)
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum


class AndersonRegime(Enum):
    """Anderson faulting regimes based on relative stress magnitudes."""
    NORMAL = "normal"           # Sv > SH > Sh (extensional)
    STRIKE_SLIP = "strike_slip" # SH > Sv > Sh
    REVERSE = "reverse"         # SH > Sh > Sv (compressional)


@dataclass
class StressTensor:
    """
    Represents a 3D stress tensor.

    Internal storage uses 6 components: [Sxx, Sxy, Sxz, Syy, Syz, Szz]
    """
    components: np.ndarray  # Shape (6,)

    def __post_init__(self):
        self.components = np.asarray(self.components, dtype=np.float64)
        if self.components.shape != (6,):
            raise ValueError(f"Expected 6 components, got shape {self.components.shape}")

    @property
    def Sxx(self) -> float:
        return self.components[0]

    @property
    def Sxy(self) -> float:
        return self.components[1]

    @property
    def Sxz(self) -> float:
        return self.components[2]

    @property
    def Syy(self) -> float:
        return self.components[3]

    @property
    def Syz(self) -> float:
        return self.components[4]

    @property
    def Szz(self) -> float:
        return self.components[5]

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 symmetric matrix."""
        return np.array([
            [self.Sxx, self.Sxy, self.Sxz],
            [self.Sxy, self.Syy, self.Syz],
            [self.Sxz, self.Syz, self.Szz]
        ])

    def principal_stresses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute principal stresses and directions.

        Returns:
            eigenvalues: Array of [S1, S2, S3] sorted S1 >= S2 >= S3
            eigenvectors: Array of shape (3, 3) where columns are principal directions
        """
        matrix = self.to_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        # Sort by descending eigenvalue (S1 >= S2 >= S3)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    def stress_ratio(self) -> float:
        """Compute stress ratio R = (S2-S3)/(S1-S3)."""
        eigenvalues, _ = self.principal_stresses()
        S1, S2, S3 = eigenvalues
        denom = S1 - S3
        if abs(denom) < 1e-10:
            return 0.5  # Undefined, return neutral value
        return (S2 - S3) / denom

    def __repr__(self) -> str:
        return (f"StressTensor(Sxx={self.Sxx:.3f}, Sxy={self.Sxy:.3f}, "
                f"Sxz={self.Sxz:.3f}, Syy={self.Syy:.3f}, "
                f"Syz={self.Syz:.3f}, Szz={self.Szz:.3f})")


def from_components(Sxx: float, Sxy: float, Sxz: float,
                    Syy: float, Syz: float, Szz: float) -> StressTensor:
    """
    Create stress tensor from individual components.

    Args:
        Sxx, Sxy, Sxz, Syy, Syz, Szz: Stress tensor components

    Returns:
        StressTensor object
    """
    return StressTensor(np.array([Sxx, Sxy, Sxz, Syy, Syz, Szz]))


def from_matrix(matrix: np.ndarray) -> StressTensor:
    """
    Create stress tensor from 3x3 symmetric matrix.

    Args:
        matrix: 3x3 symmetric stress matrix

    Returns:
        StressTensor object
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected (3,3) matrix, got {matrix.shape}")

    return StressTensor(np.array([
        matrix[0, 0],  # Sxx
        matrix[0, 1],  # Sxy
        matrix[0, 2],  # Sxz
        matrix[1, 1],  # Syy
        matrix[1, 2],  # Syz
        matrix[2, 2],  # Szz
    ]))


# =============================================================================
# Anderson Model
# =============================================================================

def from_anderson(Sh: float, SH: float, Sv: float, theta: float,
                  theta_in_degrees: bool = True) -> StressTensor:
    """
    Create stress tensor from Anderson model parameters.

    In Anderson's faulting theory:
    - Sv: vertical stress (often lithostatic = ρgz)
    - SH: maximum horizontal stress
    - Sh: minimum horizontal stress
    - theta: azimuth of SH measured from x-axis (East), counterclockwise positive

    Coordinate system: x=East, y=North, z=Up

    Args:
        Sh: Minimum horizontal stress magnitude
        SH: Maximum horizontal stress magnitude
        Sv: Vertical stress magnitude
        theta: Azimuth of SH direction from x-axis (East)
        theta_in_degrees: If True, theta is in degrees; if False, in radians

    Returns:
        StressTensor object

    Example:
        # SH oriented N30E (30° from North = 60° from East)
        stress = from_anderson(Sh=10, SH=20, Sv=25, theta=60)
    """
    if theta_in_degrees:
        theta = np.radians(theta)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos2_t = cos_t * cos_t
    sin2_t = sin_t * sin_t

    # Transform horizontal stresses to geographic coordinates
    Sxx = SH * cos2_t + Sh * sin2_t
    Syy = SH * sin2_t + Sh * cos2_t
    Sxy = (SH - Sh) * sin_t * cos_t

    # Vertical stress and no vertical shear
    Szz = Sv
    Sxz = 0.0
    Syz = 0.0

    return StressTensor(np.array([Sxx, Sxy, Sxz, Syy, Syz, Szz]))


def from_anderson_with_lithostatic(Sh_ratio: float, SH_ratio: float,
                                   theta: float, depth: float,
                                   density: float = 2500.0,
                                   g: float = 9.81,
                                   theta_in_degrees: bool = True) -> StressTensor:
    """
    Create stress tensor from Anderson model with lithostatic loading.

    The vertical stress is computed as Sv = ρgz (lithostatic).
    Horizontal stresses are specified as ratios to vertical stress:
        Sh = Sh_ratio * Sv
        SH = SH_ratio * Sv

    Args:
        Sh_ratio: Ratio of minimum horizontal stress to vertical (Sh/Sv)
        SH_ratio: Ratio of maximum horizontal stress to vertical (SH/Sv)
        theta: Azimuth of SH direction from x-axis (East)
        depth: Depth below surface in meters (positive downward)
        density: Rock density in kg/m³ (default 2500)
        g: Gravitational acceleration in m/s² (default 9.81)
        theta_in_degrees: If True, theta is in degrees

    Returns:
        StressTensor object

    Example:
        # Normal faulting regime at 1000m depth
        # Sh_ratio=0.6, SH_ratio=0.8 typical for extensional settings
        stress = from_anderson_with_lithostatic(
            Sh_ratio=0.6, SH_ratio=0.8, theta=45, depth=1000
        )
    """
    # Lithostatic vertical stress (positive compression convention)
    Sv = density * g * abs(depth)

    # Horizontal stresses from ratios
    Sh = Sh_ratio * Sv
    SH = SH_ratio * Sv

    return from_anderson(Sh=Sh, SH=SH, Sv=Sv, theta=theta,
                         theta_in_degrees=theta_in_degrees)


def classify_anderson_regime(Sh: float, SH: float, Sv: float) -> AndersonRegime:
    """
    Classify the Anderson faulting regime based on stress magnitudes.

    Args:
        Sh: Minimum horizontal stress
        SH: Maximum horizontal stress
        Sv: Vertical stress

    Returns:
        AndersonRegime enum value
    """
    stresses = sorted([('Sh', Sh), ('SH', SH), ('Sv', Sv)],
                      key=lambda x: x[1], reverse=True)

    if stresses[0][0] == 'Sv':
        return AndersonRegime.NORMAL
    elif stresses[0][0] == 'SH' and stresses[2][0] == 'Sv':
        return AndersonRegime.REVERSE
    else:
        return AndersonRegime.STRIKE_SLIP


# =============================================================================
# General Case with Principal Stresses and Euler Angles
# =============================================================================

def euler_rotation_matrix(alpha: float, beta: float, gamma: float,
                          angles_in_degrees: bool = True,
                          convention: str = "ZXZ") -> np.ndarray:
    """
    Compute rotation matrix from Euler angles.

    Supports different Euler angle conventions.

    Args:
        alpha: First rotation angle
        beta: Second rotation angle
        gamma: Third rotation angle
        angles_in_degrees: If True, angles are in degrees
        convention: Euler angle convention ("ZXZ", "ZYZ", "XYZ", etc.)

    Returns:
        3x3 rotation matrix
    """
    if angles_in_degrees:
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)

    # Rotation matrices around each axis
    Rx = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    if convention == "ZXZ":
        # Classic Euler: rotate around Z, then X', then Z''
        R = Rz(gamma) @ Rx(beta) @ Rz(alpha)
    elif convention == "ZYZ":
        R = Rz(gamma) @ Ry(beta) @ Rz(alpha)
    elif convention == "XYZ":
        # Tait-Bryan angles (roll-pitch-yaw)
        R = Rz(gamma) @ Ry(beta) @ Rx(alpha)
    elif convention == "ZYX":
        R = Rx(gamma) @ Ry(beta) @ Rz(alpha)
    else:
        raise ValueError(f"Unknown Euler convention: {convention}. "
                         f"Supported: ZXZ, ZYZ, XYZ, ZYX")

    return R


def from_principal(S1: float, S2: float, S3: float,
                   alpha: float = 0, beta: float = 0, gamma: float = 0,
                   angles_in_degrees: bool = True,
                   convention: str = "ZXZ") -> StressTensor:
    """
    Create stress tensor from principal stresses and Euler angles.

    The principal stresses define a diagonal tensor which is then rotated
    to the geographic coordinate system using the specified Euler angles.

    Convention: S1 >= S2 >= S3 (most compressive to least compressive)

    Args:
        S1: Maximum principal stress
        S2: Intermediate principal stress
        S3: Minimum principal stress
        alpha: First Euler rotation angle
        beta: Second Euler rotation angle
        gamma: Third Euler rotation angle
        angles_in_degrees: If True, angles are in degrees
        convention: Euler angle convention (default "ZXZ")

    Returns:
        StressTensor object

    Example:
        # Principal stresses with S1 horizontal pointing N45E
        stress = from_principal(S1=30, S2=20, S3=10, alpha=45, beta=90, gamma=0)
    """
    # Principal stress tensor (diagonal)
    S_principal = np.diag([S1, S2, S3])

    # Rotation matrix from Euler angles
    R = euler_rotation_matrix(alpha, beta, gamma, angles_in_degrees, convention)

    # Transform to geographic coordinates: S_geo = R @ S_principal @ R.T
    S_geo = R @ S_principal @ R.T

    return from_matrix(S_geo)


def from_principal_with_directions(S1: float, S2: float, S3: float,
                                   v1: np.ndarray, v2: np.ndarray = None,
                                   v3: np.ndarray = None) -> StressTensor:
    """
    Create stress tensor from principal stresses and directions.

    At least v1 must be provided. If v2 and v3 are not provided, they are
    computed to form an orthonormal basis.

    Args:
        S1: Maximum principal stress magnitude
        S2: Intermediate principal stress magnitude
        S3: Minimum principal stress magnitude
        v1: Direction of S1 (will be normalized)
        v2: Direction of S2 (optional, will be orthogonalized)
        v3: Direction of S3 (optional, computed if not provided)

    Returns:
        StressTensor object
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v1 = v1 / np.linalg.norm(v1)

    if v2 is None:
        # Create orthogonal v2 using Gram-Schmidt
        # Start with a vector not parallel to v1
        if abs(v1[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        v2 = temp - np.dot(temp, v1) * v1
        v2 = v2 / np.linalg.norm(v2)
    else:
        v2 = np.asarray(v2, dtype=np.float64)
        # Orthogonalize v2 against v1
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / np.linalg.norm(v2)

    if v3 is None:
        # v3 is cross product of v1 and v2
        v3 = np.cross(v1, v2)
    else:
        v3 = np.asarray(v3, dtype=np.float64)
        v3 = v3 / np.linalg.norm(v3)

    # Rotation matrix: columns are principal directions
    R = np.column_stack([v1, v2, v3])

    # Principal stress tensor
    S_principal = np.diag([S1, S2, S3])

    # Transform to geographic coordinates
    S_geo = R @ S_principal @ R.T

    return from_matrix(S_geo)


# =============================================================================
# Convenience functions
# =============================================================================

def to_components(stress: StressTensor) -> np.ndarray:
    """Extract the 6-component array from a StressTensor."""
    return stress.components.copy()


def batch_from_anderson(Sh: np.ndarray, SH: np.ndarray, Sv: np.ndarray,
                        theta: np.ndarray, theta_in_degrees: bool = True) -> np.ndarray:
    """
    Create multiple stress tensors from Anderson parameters.

    Args:
        Sh, SH, Sv, theta: Arrays of length N with stress parameters
        theta_in_degrees: If True, theta values are in degrees

    Returns:
        Array of shape (N, 6) with stress components
    """
    Sh = np.atleast_1d(Sh)
    SH = np.atleast_1d(SH)
    Sv = np.atleast_1d(Sv)
    theta = np.atleast_1d(theta)

    n = len(Sh)
    if not all(len(arr) == n for arr in [SH, Sv, theta]):
        raise ValueError("All arrays must have the same length")

    results = np.zeros((n, 6))
    for i in range(n):
        stress = from_anderson(Sh[i], SH[i], Sv[i], theta[i], theta_in_degrees)
        results[i] = stress.components

    return results


def batch_from_principal(S1: np.ndarray, S2: np.ndarray, S3: np.ndarray,
                         alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                         angles_in_degrees: bool = True,
                         convention: str = "ZXZ") -> np.ndarray:
    """
    Create multiple stress tensors from principal stress parameters.

    Args:
        S1, S2, S3: Arrays of length N with principal stresses
        alpha, beta, gamma: Arrays of length N with Euler angles
        angles_in_degrees: If True, angles are in degrees
        convention: Euler angle convention

    Returns:
        Array of shape (N, 6) with stress components
    """
    S1 = np.atleast_1d(S1)
    S2 = np.atleast_1d(S2)
    S3 = np.atleast_1d(S3)
    alpha = np.atleast_1d(alpha)
    beta = np.atleast_1d(beta)
    gamma = np.atleast_1d(gamma)

    n = len(S1)
    if not all(len(arr) == n for arr in [S2, S3, alpha, beta, gamma]):
        raise ValueError("All arrays must have the same length")

    results = np.zeros((n, 6))
    for i in range(n):
        stress = from_principal(S1[i], S2[i], S3[i],
                                alpha[i], beta[i], gamma[i],
                                angles_in_degrees, convention)
        results[i] = stress.components

    return results


# =============================================================================
# Parameterization Classes for Monte Carlo Sampling
# =============================================================================

from abc import ABC, abstractmethod


@dataclass
class ParameterSpec:
    """Specification for a single parameter with sampling range."""
    name: str
    min_val: float
    max_val: float
    description: str = ""

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a random value within the range."""
        return rng.uniform(self.min_val, self.max_val)


class StressParameterization(ABC):
    """
    Abstract base class for stress parameterizations.

    Subclasses define how to:
    1. Specify parameter ranges
    2. Sample random parameter sets
    3. Convert parameters to 6-component stress tensors

    Used by StressInversionModel to sample in physically meaningful parameter spaces.
    """

    @property
    @abstractmethod
    def parameters(self) -> list:
        """Return list of ParameterSpec for this parameterization."""
        pass

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    @property
    def param_names(self) -> list:
        """Parameter names."""
        return [p.name for p in self.parameters]

    def sample(self, rng: np.random.Generator = None) -> dict:
        """
        Sample random parameters within specified ranges.

        Args:
            rng: NumPy random generator. If None, creates a new one.

        Returns:
            Dictionary mapping parameter names to sampled values.
        """
        if rng is None:
            rng = np.random.default_rng()

        return {p.name: p.sample(rng) for p in self.parameters}

    @abstractmethod
    def to_stress(self, params: dict) -> np.ndarray:
        """
        Convert parameters to 6-component stress tensor.

        Args:
            params: Dictionary of parameter values.

        Returns:
            Array of shape (6,) with [Sxx, Sxy, Sxz, Syy, Syz, Szz].
        """
        pass

    def sample_stress(self, rng: np.random.Generator = None) -> Tuple[dict, np.ndarray]:
        """
        Sample parameters and return both params and resulting stress.

        Returns:
            Tuple of (params_dict, stress_array).
        """
        params = self.sample(rng)
        stress = self.to_stress(params)
        return params, stress


@dataclass
class AndersonParameterization(StressParameterization):
    """
    Anderson faulting model parameterization.

    Parameters: Sh, SH, Sv, theta
    - Sh: Minimum horizontal stress
    - SH: Maximum horizontal stress
    - Sv: Vertical stress
    - theta: Azimuth of SH from x-axis (East), in degrees

    Example:
        param = AndersonParameterization(
            Sh_range=(5, 20),
            SH_range=(10, 40),
            Sv_range=(15, 50),
            theta_range=(0, 180)
        )
        stress = param.sample_stress(rng)[1]
    """
    Sh_range: Tuple[float, float] = (0, 50)
    SH_range: Tuple[float, float] = (0, 100)
    Sv_range: Tuple[float, float] = (0, 100)
    theta_range: Tuple[float, float] = (0, 180)

    @property
    def parameters(self) -> list:
        return [
            ParameterSpec("Sh", self.Sh_range[0], self.Sh_range[1],
                          "Minimum horizontal stress"),
            ParameterSpec("SH", self.SH_range[0], self.SH_range[1],
                          "Maximum horizontal stress"),
            ParameterSpec("Sv", self.Sv_range[0], self.Sv_range[1],
                          "Vertical stress"),
            ParameterSpec("theta", self.theta_range[0], self.theta_range[1],
                          "Azimuth of SH from East (degrees)"),
        ]

    def to_stress(self, params: dict) -> np.ndarray:
        stress = from_anderson(
            Sh=params["Sh"],
            SH=params["SH"],
            Sv=params["Sv"],
            theta=params["theta"],
            theta_in_degrees=True
        )
        return stress.components


@dataclass
class AndersonLithostaticParameterization(StressParameterization):
    """
    Anderson model with lithostatic vertical stress.

    Parameters: Sh_ratio, SH_ratio, theta, depth
    - Sh_ratio: Sh/Sv ratio
    - SH_ratio: SH/Sv ratio
    - theta: Azimuth of SH from x-axis (East), in degrees
    - depth: Depth in meters (Sv = ρgz)

    Fixed parameters (set at initialization):
    - density: Rock density in kg/m³
    - g: Gravitational acceleration

    Example:
        # Normal faulting exploration
        param = AndersonLithostaticParameterization(
            Sh_ratio_range=(0.4, 0.8),
            SH_ratio_range=(0.6, 1.0),
            theta_range=(0, 180),
            depth_range=(500, 2000),
            density=2500
        )
    """
    Sh_ratio_range: Tuple[float, float] = (0.3, 1.0)
    SH_ratio_range: Tuple[float, float] = (0.5, 1.5)
    theta_range: Tuple[float, float] = (0, 180)
    depth_range: Tuple[float, float] = (100, 5000)
    density: float = 2500.0
    g: float = 9.81

    @property
    def parameters(self) -> list:
        return [
            ParameterSpec("Sh_ratio", self.Sh_ratio_range[0], self.Sh_ratio_range[1],
                          "Sh/Sv ratio"),
            ParameterSpec("SH_ratio", self.SH_ratio_range[0], self.SH_ratio_range[1],
                          "SH/Sv ratio"),
            ParameterSpec("theta", self.theta_range[0], self.theta_range[1],
                          "Azimuth of SH from East (degrees)"),
            ParameterSpec("depth", self.depth_range[0], self.depth_range[1],
                          "Depth in meters"),
        ]

    def to_stress(self, params: dict) -> np.ndarray:
        stress = from_anderson_with_lithostatic(
            Sh_ratio=params["Sh_ratio"],
            SH_ratio=params["SH_ratio"],
            theta=params["theta"],
            depth=params["depth"],
            density=self.density,
            g=self.g,
            theta_in_degrees=True
        )
        return stress.components


@dataclass
class PrincipalParameterization(StressParameterization):
    """
    Principal stress parameterization with Euler angles.

    Parameters: S1, S2, S3, alpha, beta, gamma
    - S1: Maximum principal stress (S1 >= S2 >= S3)
    - S2: Intermediate principal stress
    - S3: Minimum principal stress
    - alpha, beta, gamma: Euler angles in degrees

    Example:
        param = PrincipalParameterization(
            S1_range=(20, 50),
            S2_range=(10, 30),
            S3_range=(0, 15),
            alpha_range=(0, 360),
            beta_range=(0, 180),
            gamma_range=(0, 360),
            convention="ZXZ"
        )
    """
    S1_range: Tuple[float, float] = (0, 100)
    S2_range: Tuple[float, float] = (0, 100)
    S3_range: Tuple[float, float] = (0, 100)
    alpha_range: Tuple[float, float] = (0, 360)
    beta_range: Tuple[float, float] = (0, 180)
    gamma_range: Tuple[float, float] = (0, 360)
    convention: str = "ZXZ"

    @property
    def parameters(self) -> list:
        return [
            ParameterSpec("S1", self.S1_range[0], self.S1_range[1],
                          "Maximum principal stress"),
            ParameterSpec("S2", self.S2_range[0], self.S2_range[1],
                          "Intermediate principal stress"),
            ParameterSpec("S3", self.S3_range[0], self.S3_range[1],
                          "Minimum principal stress"),
            ParameterSpec("alpha", self.alpha_range[0], self.alpha_range[1],
                          "First Euler angle (degrees)"),
            ParameterSpec("beta", self.beta_range[0], self.beta_range[1],
                          "Second Euler angle (degrees)"),
            ParameterSpec("gamma", self.gamma_range[0], self.gamma_range[1],
                          "Third Euler angle (degrees)"),
        ]

    def to_stress(self, params: dict) -> np.ndarray:
        stress = from_principal(
            S1=params["S1"],
            S2=params["S2"],
            S3=params["S3"],
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            angles_in_degrees=True,
            convention=self.convention
        )
        return stress.components


@dataclass
class PrincipalLithostaticParameterization(StressParameterization):
    """
    Principal stress parameterization with lithostatic constraint.

    One principal stress is constrained to be vertical (Sv = ρgz).
    The other two are specified as ratios to Sv.

    Parameters: S1_ratio, S3_ratio, alpha, depth
    - S1_ratio: S1/Sv ratio (must be >= 1 for Sv to be S2 or S3)
    - S3_ratio: S3/Sv ratio (must be <= 1 for Sv to be S1 or S2)
    - alpha: Azimuth of S1 horizontal projection from East (degrees)
    - depth: Depth in meters

    The regime (which principal stress is vertical) is determined by ratios:
    - Normal faulting: S1=Sv (S1_ratio=1, S3_ratio<1)
    - Strike-slip: S2=Sv (S1_ratio>1, S3_ratio<1)
    - Reverse faulting: S3=Sv (S1_ratio>1, S3_ratio=1)
    """
    S1_ratio_range: Tuple[float, float] = (0.5, 2.0)
    S3_ratio_range: Tuple[float, float] = (0.2, 1.0)
    alpha_range: Tuple[float, float] = (0, 180)
    depth_range: Tuple[float, float] = (100, 5000)
    density: float = 2500.0
    g: float = 9.81

    @property
    def parameters(self) -> list:
        return [
            ParameterSpec("S1_ratio", self.S1_ratio_range[0], self.S1_ratio_range[1],
                          "S1/Sv ratio"),
            ParameterSpec("S3_ratio", self.S3_ratio_range[0], self.S3_ratio_range[1],
                          "S3/Sv ratio"),
            ParameterSpec("alpha", self.alpha_range[0], self.alpha_range[1],
                          "Azimuth of S1 from East (degrees)"),
            ParameterSpec("depth", self.depth_range[0], self.depth_range[1],
                          "Depth in meters"),
        ]

    def to_stress(self, params: dict) -> np.ndarray:
        Sv = self.density * self.g * abs(params["depth"])
        S1_ratio = params["S1_ratio"]
        S3_ratio = params["S3_ratio"]
        alpha = params["alpha"]

        # Compute absolute stresses
        S1_candidate = S1_ratio * Sv
        S3_candidate = S3_ratio * Sv

        # Sort to ensure S1 >= S2 >= S3
        stresses = sorted([S1_candidate, Sv, S3_candidate], reverse=True)
        S1, S2, S3 = stresses

        # Determine orientation: S1 is horizontal at azimuth alpha
        # Use beta=90 to put S1 in horizontal plane
        stress = from_principal(
            S1=S1, S2=S2, S3=S3,
            alpha=alpha, beta=90, gamma=0,
            angles_in_degrees=True,
            convention="ZXZ"
        )
        return stress.components


@dataclass
class DirectComponentParameterization(StressParameterization):
    """
    Direct parameterization of stress tensor components.

    Parameters: Sxx, Sxy, Sxz, Syy, Syz, Szz

    This is useful when you want to sample directly in component space.
    """
    Sxx_range: Tuple[float, float] = (-100, 100)
    Sxy_range: Tuple[float, float] = (-50, 50)
    Sxz_range: Tuple[float, float] = (-50, 50)
    Syy_range: Tuple[float, float] = (-100, 100)
    Syz_range: Tuple[float, float] = (-50, 50)
    Szz_range: Tuple[float, float] = (-100, 100)

    @property
    def parameters(self) -> list:
        return [
            ParameterSpec("Sxx", self.Sxx_range[0], self.Sxx_range[1], "Sxx component"),
            ParameterSpec("Sxy", self.Sxy_range[0], self.Sxy_range[1], "Sxy component"),
            ParameterSpec("Sxz", self.Sxz_range[0], self.Sxz_range[1], "Sxz component"),
            ParameterSpec("Syy", self.Syy_range[0], self.Syy_range[1], "Syy component"),
            ParameterSpec("Syz", self.Syz_range[0], self.Syz_range[1], "Syz component"),
            ParameterSpec("Szz", self.Szz_range[0], self.Szz_range[1], "Szz component"),
        ]

    def to_stress(self, params: dict) -> np.ndarray:
        return np.array([
            params["Sxx"], params["Sxy"], params["Sxz"],
            params["Syy"], params["Syz"], params["Szz"]
        ])
