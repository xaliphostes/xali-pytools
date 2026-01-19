# Geophysics - Stress Inversion

## Joint/Fracture and Stylolite Stress Inversion

Find the stress tensor that best explains fracture/stylolite orientations:

```python
import numpy as np
from xali_tools.geophysics.joint import cost_multiple_joints
from xali_tools.geophysics.stylolite import cost_multiple_stylolites
from xali_tools.geophysics.inversion import monte_carlo_inversion

# Joint normals (fractures open perpendicular to S3)
joint_normals = np.array([
    [0.1, 0.05, 0.99],
    [-0.05, 0.1, 0.98],
])

# Stylolite normals (pressure solution perpendicular to S1)
stylolite_normals = np.array([
    [0.98, 0.1, 0.05],
    [0.99, -0.05, 0.1],
])

# Cost functions:
# Joint: c = 1 - |dot(n, S3)|  -> aligned with minimum stress
# Stylolite: c = 1 - |dot(n, S1)|  -> aligned with maximum stress

# Invert for joints only (general 3D stress, 6 parameters)
joint_cost = lambda stress: cost_multiple_joints(joint_normals, stress)
result = monte_carlo_inversion(joint_cost, n_iterations=10000)
print(f"S3 direction: {result.principal_directions[2]}")

# Andersonian stress (4 parameters: sxx, sxy, syy, szz)
# One principal stress is constrained to be vertical (sxz = syz = 0)
result = monte_carlo_inversion(joint_cost, n_iterations=10000, andersonian=True)
print(f"S3 direction: {result.principal_directions[2]}")  # Will be exactly vertical
print(f"Parameters [sxx, sxy, syy, szz]: {result.andersonian_params}")

# Combined inversion (joints + stylolites)
def combined_cost(stress):
    return 0.5 * cost_multiple_joints(joint_normals, stress) + \
           0.5 * cost_multiple_stylolites(stylolite_normals, stress)

result = monte_carlo_inversion(combined_cost, n_iterations=20000)
```

## Multi-Point Stress Inversion Model

Invert for stress weights when you have multiple observation points, each with multiple source stresses:

```python
import numpy as np
from xali_tools.geophysics.stress_inversion_model import StressInversionModel

# Create model with 6 stress sources
model = StressInversionModel(n_sources=6)

# Source stresses at a point (shape: n_sources x 6 components)
# These could come from 6 different faults or loading conditions
source_stresses = np.random.randn(6, 6) * 10

# Add observations with different data types
model.add_joint(stresses=source_stresses, normal=[0, 0, 1])
model.add_stylolite(stresses=source_stresses, normal=[1, 0, 0])
model.add_stress_direction_and_ratio(
    stresses=source_stresses,
    direction=[1, 0, 0],  # Observed S1 direction
    R=0.5,                # Observed stress ratio
    principal_index=0     # Compare with S1
)

# Run inversion to find optimal weights
result = model.run(n_iterations=20000, weight_range=(-2, 2))

print(f"Best weights: {result.best_weights}")
print(f"Best cost: {result.best_cost}")
print(f"Per-point costs: {result.costs_per_point}")

# Weighted stress = sum(weight_i * source_stress_i)
```

## Pressure Inversion Model (Dyke Trajectories)

Joint inversion for tectonic stress, gravity (density ratio), and magma pressures:

```python
import numpy as np
from xali_tools.geophysics.stress_inversion_model import PressureInversionModel

# Create model with:
# - 6 tectonic stress sources
# - gravity/lithostatic stress (scaled by R_v density ratio)
# - 2 pressure sources (magma chambers)
model = PressureInversionModel(
    n_tectonic=6,
    include_gravity=True,
    n_pressure=2
)

# Source stresses at observation points
tectonic_stresses = np.random.randn(6, 6) * 10  # (n_tectonic, 6)
# Each pressure source has its associated gravity stress field
gravity_stresses = np.random.randn(2, 6) * 5    # (n_pressure, 6)
pressure_stresses = np.random.randn(2, 6) * 5   # (n_pressure, 6)

# Add dyke observations (dyke normals aligned with S3)
model.add_joint(
    tectonic_stresses=tectonic_stresses,
    gravity_stresses=gravity_stresses,
    pressure_stresses=pressure_stresses,
    normal=[0, 0, 1]
)

# Run inversion
# Combined stress: s = sum(ai * s_tectonic_i) + R_v * sum(s_gravity_j) + sum(Pj * s_pressure_j)
result = model.run(
    n_iterations=50000,
    tectonic_range=(-2, 2),   # ai can be negative
    density_range=(0, 2),      # R_v >= 0
    pressure_range=(0, 100)    # Pj can be negative
)

print(f"Tectonic weights: {result.tectonic_weights}")
print(f"Density ratio R_v: {result.density_ratio}")
print(f"Pressures: {result.pressures}")
print(f"All parameters: {result.all_parameters}")
```

## Inversion Summary

| Model | Parameters | Use case |
|-------|------------|----------|
| `monte_carlo_inversion(..., andersonian=False)` | 6 | General 3D stress |
| `monte_carlo_inversion(..., andersonian=True)` | 4 | Andersonian (vertical principal) |
| `StressInversionModel(n_sources=N)` | N weights | Multi-source weight inversion |
| `PressureInversionModel(n_tectonic, gravity, n_pressure)` | N + 1 + M | Dyke/magma inversion |
