import numpy as np
import sys
sys.path.insert(0, 'src')

import matplotlib.pyplot as plt

from xali_tools.slip_envelope import (
    ParameterType,
    ParameterRange,
    DomainSpecification,
    OutputMetric,
    DomainSweepAnalyzer,
)
from xali_tools.io.tsurf_filter import load_tsurf


# Parameter display names and units
PARAM_INFO = {
    ParameterType.FRICTION: ("Friction Coefficient (μ)", ""),
    ParameterType.COHESION: ("Cohesion (MPa)", "MPa"),
    ParameterType.PORE_PRESSURE: ("Pore Pressure (MPa)", "MPa"),
    ParameterType.STRESS_RATIO: ("Stress Ratio", ""),
    ParameterType.SH: ("Min Horizontal Stress Sh (MPa)", "MPa"),
    ParameterType.SH_MAX: ("Max Horizontal Stress SH (MPa)", "MPa"),
    ParameterType.SV: ("Vertical Stress Sv (MPa)", "MPa"),
    ParameterType.THETA: ("Stress Orientation θ (°)", "°"),
    ParameterType.DEPTH: ("Depth (m)", "m"),
    ParameterType.TEMPERATURE: ("Temperature (°C)", "°C"),
}


def get_param_label(param_type: ParameterType) -> str:
    """Get display label for a parameter type."""
    return PARAM_INFO.get(param_type, (param_type.value, ""))[0]


def do_slip2D(
    pathfile: str,
    x_axis: ParameterType,
    x_min: float,
    x_max: float,
    y_axis: ParameterType,
    y_min: float,
    y_max: float,
    n_steps: int = 50,
    stress: list = None,
    fixed_values: dict = None,
):
    """
    Perform slip envelope analysis with user-defined axes.

    Args:
        pathfile: Path to the surface file (.ts)
        x_axis: Parameter type for X axis
        x_min, x_max: Bounds for X axis
        y_axis: Parameter type for Y axis
        y_min, y_max: Bounds for Y axis
        n_steps: Number of steps for each axis
        stress: Stress tensor [Sxx, Sxy, Sxz, Syy, Syz, Szz]
        fixed_values: Dict of {ParameterType: value} for fixed parameters
    """
    if stress is None:
        stress = [0, 0, 5, 0, 0, -15]
    if fixed_values is None:
        fixed_values = {}

    # 1. Load mesh
    print(f"Loading surface from {pathfile}...")
    surf = load_tsurf(pathfile, index=0)
    positions = surf.positions
    indices = surf.indices
    print(f"  Loaded {surf.n_vertices} vertices, {surf.n_triangles} triangles")

    # 2. Build domain specification
    domain = DomainSpecification(
        parameters=[
            ParameterRange(x_axis, x_min, x_max, n_steps),
            ParameterRange(y_axis, y_min, y_max, n_steps),
        ],
        fixed_values=fixed_values,
    )

    print(f"\nDomain specification:")
    print(f"  X axis: {x_axis.value} [{x_min}, {x_max}]")
    print(f"  Y axis: {y_axis.value} [{y_min}, {y_max}]")
    print(f"  Steps: {n_steps} x {n_steps} = {domain.n_points} points")
    if fixed_values:
        print(f"  Fixed: {', '.join(f'{k.value}={v}' for k, v in fixed_values.items())}")

    # 3. Domain sweep
    print("\nRunning domain sweep...")
    sweep_analyzer = DomainSweepAnalyzer(positions, indices)

    envelope = sweep_analyzer.sweep(
        domain=domain,
        stress=stress,
        metric=OutputMetric.SLIP_RATIO,
    )

    print(f"  Envelope shape: {envelope.shape}")
    print(f"  Slip ratio range: [{np.min(envelope.values):.2%}, {np.max(envelope.values):.2%}]")

    # 4. Plot iso-contours
    X, Y, Z = envelope.get_2d_contour_data()

    _, ax = plt.subplots(figsize=(10, 8))

    # Filled contours
    levels = np.linspace(0, 1, 21)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlGn_r')

    # Colorbar
    plt.colorbar(contourf, ax=ax, label='Slip Ratio')

    # Iso-contour lines
    contour_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    contour_lines = ax.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=1.5)
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')

    # Highlight 50% boundary
    ax.contour(X, Y, Z, levels=[0.5], colors='white', linewidths=3)

    # Labels
    ax.set_xlabel(get_param_label(x_axis), fontsize=12)
    ax.set_ylabel(get_param_label(y_axis), fontsize=12)
    ax.set_title(f'Slip Envelope: {x_axis.value} vs {y_axis.value}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return envelope


def do_slip3D(
    pathfile: str,
    x_axis: ParameterType,
    x_min: float,
    x_max: float,
    y_axis: ParameterType,
    y_min: float,
    y_max: float,
    z_axis: ParameterType,
    z_min: float,
    z_max: float,
    n_steps: int = 30,
    stress: list = None,
    fixed_values: dict = None,
    n_slices: int = 4,
):
    """
    Perform 3D slip envelope analysis with user-defined axes.

    Args:
        pathfile: Path to the surface file (.ts)
        x_axis: Parameter type for X axis
        x_min, x_max: Bounds for X axis
        y_axis: Parameter type for Y axis
        y_min, y_max: Bounds for Y axis
        z_axis: Parameter type for Z axis (slicing axis)
        z_min, z_max: Bounds for Z axis
        n_steps: Number of steps for each axis
        stress: Stress tensor [Sxx, Sxy, Sxz, Syy, Syz, Szz]
        fixed_values: Dict of {ParameterType: value} for fixed parameters
        n_slices: Number of slices to display along Z axis
    """
    if stress is None:
        stress = [0, 0, 5, 0, 0, -15]
    if fixed_values is None:
        fixed_values = {}

    # 1. Load mesh
    print(f"Loading surface from {pathfile}...")
    surf = load_tsurf(pathfile, index=0)
    positions = surf.positions
    indices = surf.indices
    print(f"  Loaded {surf.n_vertices} vertices, {surf.n_triangles} triangles")

    # 2. Build 3D domain specification
    domain = DomainSpecification(
        parameters=[
            ParameterRange(x_axis, x_min, x_max, n_steps),
            ParameterRange(y_axis, y_min, y_max, n_steps),
            ParameterRange(z_axis, z_min, z_max, n_steps),
        ],
        fixed_values=fixed_values,
    )

    print(f"\n3D Domain specification:")
    print(f"  X axis: {x_axis.value} [{x_min}, {x_max}]")
    print(f"  Y axis: {y_axis.value} [{y_min}, {y_max}]")
    print(f"  Z axis: {z_axis.value} [{z_min}, {z_max}]")
    print(f"  Steps: {n_steps}^3 = {domain.n_points} points")
    if fixed_values:
        print(f"  Fixed: {', '.join(f'{k.value}={v}' for k, v in fixed_values.items())}")

    # 3. Domain sweep
    print("\nRunning 3D domain sweep...")
    sweep_analyzer = DomainSweepAnalyzer(positions, indices)

    envelope = sweep_analyzer.sweep(
        domain=domain,
        stress=stress,
        metric=OutputMetric.SLIP_RATIO,
    )

    print(f"  Envelope shape: {envelope.shape}")
    print(f"  Slip ratio range: [{np.min(envelope.values):.2%}, {np.max(envelope.values):.2%}]")

    # 4. Plot slices along Z axis
    z_values = envelope.axes[2]
    slice_indices = np.linspace(0, len(z_values) - 1, n_slices, dtype=int)

    # Determine grid layout
    n_cols = min(n_slices, 2)
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_slices == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    levels = np.linspace(0, 1, 21)
    contour_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

    for i, slice_idx in enumerate(slice_indices):
        ax = axes[i]
        z_val = z_values[slice_idx]

        # Get 2D slice
        slice_2d = envelope.get_slice(axis=2, index=slice_idx)
        X, Y, Z = slice_2d.get_2d_contour_data()

        # Filled contours
        contourf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlGn_r')

        # Iso-contour lines
        contour_lines = ax.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=1)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

        # Highlight 50% boundary
        ax.contour(X, Y, Z, levels=[0.5], colors='white', linewidths=2)

        # Labels
        ax.set_xlabel(get_param_label(x_axis), fontsize=10)
        ax.set_ylabel(get_param_label(y_axis), fontsize=10)
        ax.set_title(f'{z_axis.value} = {z_val:.2f}', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(len(slice_indices), len(axes)):
        axes[i].set_visible(False)

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(contourf, cax=cbar_ax, label='Slip Ratio')

    fig.suptitle(f'3D Slip Envelope: {x_axis.value} vs {y_axis.value} (sliced by {z_axis.value})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

    return envelope


def interactive_mode(pathfile: str):
    """Interactive mode to let user choose axes and boundaries."""
    print("\n=== Slip Envelope Analysis ===\n")

    # Choose 2D or 3D
    mode = input("Analysis mode (2D/3D, default 2D): ").strip().upper() or "2D"
    is_3d = mode == "3D"

    # Available parameters
    params = [
        ParameterType.FRICTION,
        ParameterType.COHESION,
        ParameterType.PORE_PRESSURE,
        ParameterType.STRESS_RATIO,
        ParameterType.THETA,
        ParameterType.SH,
        ParameterType.SH_MAX,
        ParameterType.SV,
    ]

    print("\nAvailable parameters:")
    for i, p in enumerate(params):
        print(f"  {i + 1}. {p.value}")

    # X axis
    print("\n--- X Axis ---")
    x_idx = int(input(f"Select X axis parameter (1-{len(params)}): ")) - 1
    x_axis = params[x_idx]
    x_min = float(input(f"  {x_axis.value} min: "))
    x_max = float(input(f"  {x_axis.value} max: "))

    # Y axis
    print("\n--- Y Axis ---")
    remaining = [p for p in params if p != x_axis]
    print("Available parameters:")
    for i, p in enumerate(remaining):
        print(f"  {i + 1}. {p.value}")
    y_idx = int(input(f"Select Y axis parameter (1-{len(remaining)}): ")) - 1
    y_axis = remaining[y_idx]
    y_min = float(input(f"  {y_axis.value} min: "))
    y_max = float(input(f"  {y_axis.value} max: "))

    # Z axis (only for 3D)
    z_axis = None
    z_min = z_max = n_slices = None
    if is_3d:
        print("\n--- Z Axis (slicing axis) ---")
        remaining_z = [p for p in remaining if p != y_axis]
        print("Available parameters:")
        for i, p in enumerate(remaining_z):
            print(f"  {i + 1}. {p.value}")
        z_idx = int(input(f"Select Z axis parameter (1-{len(remaining_z)}): ")) - 1
        z_axis = remaining_z[z_idx]
        z_min = float(input(f"  {z_axis.value} min: "))
        z_max = float(input(f"  {z_axis.value} max: "))
        n_slices = int(input("  Number of slices to display (default 4): ") or "4")

    # Steps
    default_steps = "30" if is_3d else "50"
    n_steps = int(input(f"\nNumber of steps per axis (default {default_steps}): ") or default_steps)

    # Default values for each parameter type
    defaults = {
        ParameterType.FRICTION: 0.6,
        ParameterType.COHESION: 0.0,
        ParameterType.PORE_PRESSURE: 0.0,
        ParameterType.STRESS_RATIO: 1.0,
        ParameterType.THETA: 0.0,
        ParameterType.SH: 10.0,
        ParameterType.SH_MAX: 20.0,
        ParameterType.SV: 25.0,
    }

    # Fixed values for unused parameters
    fixed_values = {}
    used_params = [x_axis, y_axis] + ([z_axis] if is_3d else [])
    unused = [p for p in params if p not in used_params]
    if unused:
        print("\n--- Fixed Values ---")
        for p in unused:
            default = defaults.get(p, 0.0)
            val = input(f"  {p.value} (default {default}): ")
            fixed_values[p] = float(val) if val else default

    # Stress tensor
    print("\n--- Stress Tensor ---")
    stress_input = input("Stress [Sxx,Sxy,Sxz,Syy,Syz,Szz] (default [0,0,5,0,0,-15]): ")
    if stress_input:
        stress = [float(x) for x in stress_input.strip('[]').split(',')]
    else:
        stress = [0, 0, 5, 0, 0, -15]

    if is_3d:
        return do_slip3D(
            pathfile,
            x_axis=x_axis, x_min=x_min, x_max=x_max,
            y_axis=y_axis, y_min=y_min, y_max=y_max,
            z_axis=z_axis, z_min=z_min, z_max=z_max,
            n_steps=n_steps,
            stress=stress,
            fixed_values=fixed_values,
            n_slices=n_slices,
        )
    else:
        return do_slip2D(
            pathfile,
            x_axis=x_axis, x_min=x_min, x_max=x_max,
            y_axis=y_axis, y_min=y_min, y_max=y_max,
            n_steps=n_steps,
            stress=stress,
            fixed_values=fixed_values,
        )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python slip_envelope.py <file.ts> [--interactive | --3d]")
        print("\nExamples:")
        print("  python slip_envelope.py fault.ts --interactive")
        print("  python slip_envelope.py fault.ts --3d    # 3D domain sweep")
        print("  python slip_envelope.py fault.ts         # Default 2D friction vs pore_pressure")
        sys.exit(1)

    pathfile = sys.argv[1]

    if "--interactive" in sys.argv or "-i" in sys.argv:
        envelope = interactive_mode(pathfile)
    elif "--3d" in sys.argv:
        # 3D example: friction vs cohesion vs pore_pressure
        envelope = do_slip3D(
            pathfile,
            x_axis=ParameterType.FRICTION,
            x_min=0.0,
            x_max=0.8,
            y_axis=ParameterType.COHESION,
            y_min=0.0,
            y_max=5.0,
            z_axis=ParameterType.PORE_PRESSURE,
            z_min=0.0,
            z_max=15.0,
            n_steps=30,
            n_slices=4,
        )
    else:
        # Default: friction vs pore_pressure
        envelope = do_slip2D(
            pathfile,
            x_axis=ParameterType.FRICTION,
            x_min=0.0,
            x_max=0.8,
            y_axis=ParameterType.PORE_PRESSURE,
            y_min=0.0,
            y_max=15.0,
            n_steps=50,
            fixed_values={ParameterType.COHESION: 0.0},
        )
