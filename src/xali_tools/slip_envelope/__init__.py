"""
Slip Envelope Analysis Module.

This module provides tools for analyzing fault slip using the Coulomb failure
criterion and computing slip envelopes through parameter domain sweeps.

Key Features:
- Single slip analysis on triangulated fault surfaces
- 2D and 3D parameter domain sweeps for slip envelope computation
- Multiple output metrics: slip ratio, binary slip, slip count
- Support for pore pressure and effective stress
- Integration with SurfaceData and AttributeManager

Sign Convention (Tension Positive):
- sigma_n > 0: tension (surfaces pulled apart)
- sigma_n < 0: compression (surfaces pushed together)
- Effective stress: sigma'_n = sigma_n - Pp

Coulomb Criterion:
    Slip occurs when: tau > c - mu * sigma'_n
    Where:
        tau = shear stress magnitude
        c = cohesion
        mu = friction coefficient
        sigma'_n = effective normal stress

Example - Single Analysis:
    >>> from xali_tools.io import load_surface
    >>> from xali_tools.slip_envelope import SlipAnalyzer
    >>>
    >>> surface = load_surface("fault.ts")
    >>> analyzer = SlipAnalyzer.from_surface(surface)
    >>> result = analyzer.analyze(
    ...     stress=[10, 0, 0, 20, 0, 30],  # [Sxx, Sxy, Sxz, Syy, Syz, Szz]
    ...     friction=0.6,
    ...     pore_pressure=5.0
    ... )
    >>> print(f"Slip ratio: {result.slip_ratio:.2%}")
    >>> print(f"Slip count: {result.slip_count} / {result.n_triangles}")

Example - 2D Domain Sweep:
    >>> from xali_tools.slip_envelope import (
    ...     DomainSweepAnalyzer, friction_pore_pressure_domain, OutputMetric
    ... )
    >>> import matplotlib.pyplot as plt
    >>>
    >>> analyzer = DomainSweepAnalyzer.from_surface(surface)
    >>> domain = friction_pore_pressure_domain(
    ...     friction_range=(0.3, 0.9),
    ...     pore_pressure_range=(0, 20),
    ...     friction_steps=100,
    ...     pore_pressure_steps=100
    ... )
    >>> envelope = analyzer.sweep(domain, stress, OutputMetric.SLIP_RATIO)
    >>>
    >>> # Plot iso-contours
    >>> X, Y, Z = envelope.get_2d_contour_data()
    >>> plt.contourf(X, Y, Z, levels=20, cmap='RdYlGn_r')
    >>> plt.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=2)
    >>> plt.xlabel('Friction')
    >>> plt.ylabel('Pore Pressure')
    >>> plt.colorbar(label='Slip Ratio')

Example - 3D Domain Sweep with Slicing:
    >>> domain_3d = friction_cohesion_pore_pressure_domain(
    ...     friction_steps=30, cohesion_steps=30, pore_pressure_steps=30
    ... )
    >>> envelope_3d = analyzer.sweep(domain_3d, stress, OutputMetric.SLIP_RATIO)
    >>>
    >>> # Extract 2D slice at pore_pressure = 10
    >>> slice_2d = envelope_3d.get_slice_at_value(axis=2, value=10.0)
    >>> X, Y, Z = slice_2d.get_2d_contour_data()
"""

# Traction computation
from .traction import (
    compute_triangle_normals,
    compute_traction,
    resolve_stress,
    compute_shear_direction,
    stress_6_to_matrix,
)

# Coulomb failure criterion
from .coulomb import (
    CoulombParameters,
    evaluate_coulomb,
    coulomb_slip_tendency,
    dilation_tendency,
)

# Single slip analysis
from .slip_analysis import (
    SlipResult,
    SlipAnalyzer,
)

# Parameter domain specification
from .parameters import (
    ParameterType,
    ParameterRange,
    DomainSpecification,
    friction_pore_pressure_domain,
    friction_cohesion_domain,
    stress_ratio_friction_domain,
    friction_cohesion_pore_pressure_domain,
)

# Domain sweep analysis
from .domain_sweep import (
    OutputMetric,
    DomainResult,
    DomainSweepAnalyzer,
)

__all__ = [
    # Traction
    "compute_triangle_normals",
    "compute_traction",
    "resolve_stress",
    "compute_shear_direction",
    "stress_6_to_matrix",
    # Coulomb
    "CoulombParameters",
    "evaluate_coulomb",
    "coulomb_slip_tendency",
    "dilation_tendency",
    # Slip analysis
    "SlipResult",
    "SlipAnalyzer",
    # Parameters
    "ParameterType",
    "ParameterRange",
    "DomainSpecification",
    "friction_pore_pressure_domain",
    "friction_cohesion_domain",
    "stress_ratio_friction_domain",
    "friction_cohesion_pore_pressure_domain",
    # Domain sweep
    "OutputMetric",
    "DomainResult",
    "DomainSweepAnalyzer",
]
