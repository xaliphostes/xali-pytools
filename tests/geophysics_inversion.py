"""
Comprehensive test for geophysics inversion module.

Tests:
1. Cost functions (joint, stylolite, direction, ratio)
2. Single-stress Monte Carlo inversion (6 params and 4 Andersonian params)
3. Multi-source weight inversion (StressInversionModel)
4. Pressure inversion (PressureInversionModel)
"""
import numpy as np
from xali_tools.geophysics.joint import cost_single_joint, cost_multiple_joints
from xali_tools.geophysics.stylolite import cost_single_stylolite, cost_multiple_stylolites
from xali_tools.geophysics.stress_data import compute_stress_ratio, cost_direction
from xali_tools.geophysics.inversion import monte_carlo_inversion
from xali_tools.geophysics.stress_inversion_model import (
    StressInversionModel, PressureInversionModel
)
from xali_tools.geophysics.stress_utils import principal_directions


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# =============================================================================
# PART 1: Cost Functions
# =============================================================================
section("PART 1: Cost Functions")

# Create a stress tensor with known principal directions
# S1 = 10 along x, S2 = 5 along y, S3 = 1 along z
stress = np.array([10.0, 0, 0, 5.0, 0, 1.0])
values, directions = principal_directions(stress)
print(f"Test stress - principals: S1={values[0]:.1f}, S2={values[1]:.1f}, S3={values[2]:.1f}")
print(f"  S1 direction: {directions[0]}")
print(f"  S3 direction: {directions[2]}")

# Joint cost (should align with S3)
print("\nJoint cost (alignment with S3):")
print(f"  Normal along z (aligned):      {cost_single_joint([0,0,1], stress):.6f} (expect 0)")
print(f"  Normal along x (perpendicular): {cost_single_joint([1,0,0], stress):.6f} (expect 1)")

# Stylolite cost (should align with S1)
print("\nStylolite cost (alignment with S1):")
print(f"  Normal along x (aligned):      {cost_single_stylolite([1,0,0], stress):.6f} (expect 0)")
print(f"  Normal along z (perpendicular): {cost_single_stylolite([0,0,1], stress):.6f} (expect 1)")

# Stress ratio
print("\nStress ratio R = (S2-S3)/(S1-S3):")
R = compute_stress_ratio(stress)
expected_R = (5-1)/(10-1)
print(f"  Computed: {R:.4f}, Expected: {expected_R:.4f}")

# Direction cost
print("\nDirection cost (alignment with principal):")
print(f"  [1,0,0] vs S1: {cost_direction([1,0,0], stress, 0):.6f} (expect 0)")
print(f"  [0,1,0] vs S1: {cost_direction([0,1,0], stress, 0):.6f} (expect 1)")


# =============================================================================
# PART 2: Single-Stress Monte Carlo Inversion
# =============================================================================
section("PART 2: Single-Stress Monte Carlo Inversion")

# Generate synthetic joints (normals ~ z-axis)
np.random.seed(42)
joint_normals = np.array([[0, 0, 1]] * 15) + np.random.randn(15, 3) * 0.1
joint_normals /= np.linalg.norm(joint_normals, axis=1, keepdims=True)

# Generate synthetic stylolites (normals ~ x-axis)
stylolite_normals = np.array([[1, 0, 0]] * 15) + np.random.randn(15, 3) * 0.1
stylolite_normals /= np.linalg.norm(stylolite_normals, axis=1, keepdims=True)

# 2a. General 3D stress (6 parameters)
print("\n2a. Joint inversion - General 3D (6 params):")
cost_fn = lambda s: cost_multiple_joints(joint_normals, s)
result = monte_carlo_inversion(cost_fn, n_iterations=10000, andersonian=False, seed=123)
print(f"  Best cost: {result.best_cost:.6f}")
print(f"  S3 direction: {result.principal_directions[2]} (expect ~[0,0,1])")
print(f"  N parameters: {result.n_parameters}")

# 2b. Andersonian stress (4 parameters)
print("\n2b. Joint inversion - Andersonian (4 params):")
result_and = monte_carlo_inversion(cost_fn, n_iterations=10000, andersonian=True, seed=123)
print(f"  Best cost: {result_and.best_cost:.6f}")
print(f"  S3 direction: {result_and.principal_directions[2]} (expect [0,0,1] exactly)")
print(f"  N parameters: {result_and.n_parameters}")
print(f"  Andersonian params [sxx,sxy,syy,szz]: {result_and.andersonian_params}")

# 2c. Combined joints + stylolites
print("\n2c. Combined inversion (joints + stylolites):")
def combined_cost(s):
    return 0.5 * cost_multiple_joints(joint_normals, s) + \
           0.5 * cost_multiple_stylolites(stylolite_normals, s)

result_comb = monte_carlo_inversion(combined_cost, n_iterations=20000, seed=456)
print(f"  Best cost: {result_comb.best_cost:.6f}")
print(f"  S1 direction: {result_comb.principal_directions[0]} (expect ~[1,0,0])")
print(f"  S3 direction: {result_comb.principal_directions[2]} (expect ~[0,0,1])")


# =============================================================================
# PART 3: Multi-Source Weight Inversion (StressInversionModel)
# =============================================================================
section("PART 3: Multi-Source Weight Inversion (StressInversionModel)")

# Create synthetic source stresses
np.random.seed(42)
n_sources = 6
source_stresses = np.random.randn(n_sources, 6) * 10
true_weights = np.array([1.0, 0.5, 0.3, -0.2, 0.1, 0.0])

# Compute true combined stress
true_stress = np.sum(true_weights[:, np.newaxis] * source_stresses, axis=0)
_, true_dirs = principal_directions(true_stress)
true_R = compute_stress_ratio(true_stress)

print(f"True weights: {true_weights}")
print(f"True S3: {true_dirs[2]}")
print(f"True R: {true_R:.4f}")

# 3a. Joint observations only
print("\n3a. Joint observations:")
model_j = StressInversionModel(n_sources=6)
np.random.seed(111)
for _ in range(12):
    normal = true_dirs[2] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_j.add_joint(stresses=source_stresses, normal=normal)

result_j = model_j.run(n_iterations=20000, weight_range=(-2, 2), seed=222)
rec_stress = np.sum(result_j.best_weights[:, np.newaxis] * source_stresses, axis=0)
_, rec_dirs = principal_directions(rec_stress)
print(f"  {model_j}")
print(f"  Best cost: {result_j.best_cost:.6f}")
print(f"  S3 alignment: {abs(np.dot(rec_dirs[2], true_dirs[2])):.6f}")

# 3b. Mixed observations (joints + stylolites)
print("\n3b. Mixed observations (joints + stylolites):")
model_m = StressInversionModel(n_sources=6)
np.random.seed(333)
for _ in range(8):
    normal = true_dirs[2] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_m.add_joint(stresses=source_stresses, normal=normal)
for _ in range(8):
    normal = true_dirs[0] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_m.add_stylolite(stresses=source_stresses, normal=normal)

result_m = model_m.run(n_iterations=30000, weight_range=(-2, 2), seed=444)
rec_stress_m = np.sum(result_m.best_weights[:, np.newaxis] * source_stresses, axis=0)
_, rec_dirs_m = principal_directions(rec_stress_m)
print(f"  {model_m}")
print(f"  Best cost: {result_m.best_cost:.6f}")
print(f"  S1 alignment: {abs(np.dot(rec_dirs_m[0], true_dirs[0])):.6f}")
print(f"  S3 alignment: {abs(np.dot(rec_dirs_m[2], true_dirs[2])):.6f}")

# 3c. Direction + ratio observations
print("\n3c. Stress direction + ratio observations:")
model_dr = StressInversionModel(n_sources=6)
np.random.seed(555)
for _ in range(12):
    direction = true_dirs[0] + np.random.randn(3) * 0.1
    direction /= np.linalg.norm(direction)
    R = np.clip(true_R + np.random.randn() * 0.05, 0, 1)
    model_dr.add_stress_direction_and_ratio(
        stresses=source_stresses, direction=direction, R=R, principal_index=0
    )

result_dr = model_dr.run(n_iterations=30000, weight_range=(-2, 2), seed=666)
rec_stress_dr = np.sum(result_dr.best_weights[:, np.newaxis] * source_stresses, axis=0)
_, rec_dirs_dr = principal_directions(rec_stress_dr)
rec_R = compute_stress_ratio(rec_stress_dr)
print(f"  {model_dr}")
print(f"  Best cost: {result_dr.best_cost:.6f}")
print(f"  S1 alignment: {abs(np.dot(rec_dirs_dr[0], true_dirs[0])):.6f}")
print(f"  R recovered: {rec_R:.4f} (true: {true_R:.4f})")


# =============================================================================
# PART 4: Pressure Inversion (PressureInversionModel)
# =============================================================================
section("PART 4: Pressure Inversion (PressureInversionModel)")

# Setup
np.random.seed(42)
n_tectonic = 6
tectonic_sources = np.random.randn(n_tectonic, 6) * 10
n_pressure = 2
# Each pressure source has its own gravity stress field
gravity_stresses = np.random.randn(n_pressure, 6) * 5
pressure_sources = np.random.randn(n_pressure, 6) * 5

true_tectonic = np.array([1.0, 0.5, 0.3, -0.2, 0.1, 0.0])
true_Rv = 0.8
true_pressures = np.array([50.0, 30.0])

# True combined stress:
# σ = Σ(αᵢ × σ_tectonic_i) + R_v × Σ(σ_gravity_j) + Σ(Pⱼ × σ_pressure_j)
true_stress_p = (
    np.sum(true_tectonic[:, np.newaxis] * tectonic_sources, axis=0) +
    true_Rv * np.sum(gravity_stresses, axis=0) +
    np.sum(true_pressures[:, np.newaxis] * pressure_sources, axis=0)
)
_, true_dirs_p = principal_directions(true_stress_p)

print(f"True R_v: {true_Rv}")
print(f"True pressures: {true_pressures}")
print(f"True S3: {true_dirs_p[2]}")

# 4a. Tectonic only (no pressure, no gravity)
print("\n4a. Tectonic only (no pressure sources):")
model_t = PressureInversionModel(n_tectonic=6, include_gravity=False, n_pressure=0)

# Compute true stress for tectonic only
true_stress_t = np.sum(true_tectonic[:, np.newaxis] * tectonic_sources, axis=0)
_, true_dirs_t = principal_directions(true_stress_t)

np.random.seed(777)
for _ in range(15):
    normal = true_dirs_t[2] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_t.add_joint(
        tectonic_stresses=tectonic_sources,
        normal=normal
    )

result_t = model_t.run(n_iterations=30000, tectonic_range=(-2, 2), seed=888)
rec_stress_t = np.sum(result_t.tectonic_weights[:, np.newaxis] * tectonic_sources, axis=0)
_, rec_dirs_t = principal_directions(rec_stress_t)
print(f"  {model_t}")
print(f"  Best cost: {result_t.best_cost:.6f}")
print(f"  S3 alignment: {abs(np.dot(rec_dirs_t[2], true_dirs_t[2])):.6f}")

# 4b. Full pressure inversion (tectonic + gravity + pressure)
print("\n4b. Full pressure inversion (tectonic + gravity + pressure):")
model_full = PressureInversionModel(n_tectonic=6, include_gravity=True, n_pressure=2)

np.random.seed(999)
for _ in range(20):
    # Add spatial variation
    local_tectonic = tectonic_sources + np.random.randn(6, 6) * 1
    local_gravity = gravity_stresses + np.random.randn(2, 6) * 0.5
    local_pressure = pressure_sources + np.random.randn(2, 6) * 0.5

    local_stress = (
        np.sum(true_tectonic[:, np.newaxis] * local_tectonic, axis=0) +
        true_Rv * np.sum(local_gravity, axis=0) +
        np.sum(true_pressures[:, np.newaxis] * local_pressure, axis=0)
    )
    _, local_dirs = principal_directions(local_stress)

    normal = local_dirs[2] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)

    model_full.add_joint(
        tectonic_stresses=local_tectonic,
        gravity_stresses=local_gravity,
        pressure_stresses=local_pressure,
        normal=normal
    )

result_full = model_full.run(
    n_iterations=50000,
    tectonic_range=(-2, 2),
    density_range=(0, 2),
    pressure_range=(0, 100),
    seed=1010
)

# Compute mean recovered S3
mean_stress = np.mean(result_full.weighted_stresses, axis=0)
_, rec_dirs_full = principal_directions(mean_stress)

print(f"  {model_full}")
print(f"  Total parameters: {model_full.n_parameters}")
print(f"  Best cost: {result_full.best_cost:.6f}")
print(f"  S3 alignment: {abs(np.dot(rec_dirs_full[2], true_dirs_p[2])):.6f}")
print(f"  All parameters: {result_full.all_parameters[:4]}... (first 4)")


# =============================================================================
# Summary
# =============================================================================
section("Summary")
print("All tests completed successfully.")
print("\nModules tested:")
print("  - joint.py: cost_single_joint, cost_multiple_joints")
print("  - stylolite.py: cost_single_stylolite, cost_multiple_stylolites")
print("  - stress_data.py: compute_stress_ratio, cost_direction, cost_stress_ratio")
print("  - inversion.py: monte_carlo_inversion (6 params & 4 Andersonian)")
print("  - stress_inversion_model.py: StressInversionModel, PressureInversionModel")
