"""
Comprehensive test for geophysics inversion module.

Tests:
1. Cost functions (joint, stylolite, direction, ratio)
2. Single-stress Monte Carlo inversion (6 params and 4 Andersonian params)
3. Multi-source weight inversion (StressInversionModel)
4. Pressure inversion (PressureInversionModel)
5. Stress tensor utilities (from_anderson, from_principal, etc.)
6. Parameterization classes (AndersonParameterization, etc.)
7. Direct stress inversion (DirectStressInversionModel)
"""
import numpy as np
from xali_tools.geophysics.joint import cost_single_joint, cost_multiple_joints
from xali_tools.geophysics.stylolite import cost_single_stylolite, cost_multiple_stylolites
from xali_tools.geophysics.stress_data import compute_stress_ratio, cost_direction
from xali_tools.geophysics.inversion import monte_carlo_inversion
from xali_tools.geophysics.stress_inversion_model import (
    StressInversionModel, PressureInversionModel, DirectStressInversionModel
)
from xali_tools.geophysics.stress_utils import principal_directions
from xali_tools.geophysics.stress_tensor import (
    StressTensor,
    from_anderson,
    from_anderson_with_lithostatic,
    from_principal,
    from_principal_with_directions,
    from_components,
    classify_anderson_regime,
    AndersonRegime,
    AndersonParameterization,
    AndersonLithostaticParameterization,
    PrincipalParameterization,
    DirectComponentParameterization,
)


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
# PART 5: Stress Tensor Utilities
# =============================================================================
section("PART 5: Stress Tensor Utilities")

# 5a. from_anderson - verify stress tensor construction
print("\n5a. Anderson model stress tensor:")
# SH along x-axis (theta=0), normal faulting regime (Sv > SH > Sh)
stress_and = from_anderson(Sh=10, SH=20, Sv=30, theta=0)
print(f"  Input: Sh=10, SH=20, Sv=30, theta=0°")
print(f"  Components: {stress_and.components}")
print(f"  Expected: Sxx=20, Syy=10, Szz=30, Sxy=Sxz=Syz=0")

# Verify principal stresses
principals, dirs = stress_and.principal_stresses()
print(f"  Principal stresses: S1={principals[0]:.1f}, S2={principals[1]:.1f}, S3={principals[2]:.1f}")
print(f"  S1 direction (should be ~z): {dirs[:, 0]}")

# Test rotated SH (45 degrees from x)
stress_rot = from_anderson(Sh=10, SH=30, Sv=20, theta=45)
print(f"\n  Rotated (theta=45°): Sxx={stress_rot.Sxx:.2f}, Syy={stress_rot.Syy:.2f}, Sxy={stress_rot.Sxy:.2f}")
print(f"  Expected: Sxx=Syy=20, Sxy=10")

# 5b. from_anderson_with_lithostatic
print("\n5b. Anderson with lithostatic loading:")
stress_litho = from_anderson_with_lithostatic(
    Sh_ratio=0.6, SH_ratio=0.8, theta=30, depth=1000, density=2500
)
expected_Sv = 2500 * 9.81 * 1000
print(f"  Input: Sh_ratio=0.6, SH_ratio=0.8, depth=1000m, density=2500")
print(f"  Expected Sv = ρgz = {expected_Sv/1e6:.2f} MPa")
print(f"  Szz component: {stress_litho.Szz/1e6:.2f} MPa")

# 5c. classify_anderson_regime
print("\n5c. Anderson regime classification:")
print(f"  Sv=30, SH=20, Sh=10 → {classify_anderson_regime(10, 20, 30)} (expect NORMAL)")
print(f"  Sv=20, SH=30, Sh=10 → {classify_anderson_regime(10, 30, 20)} (expect STRIKE_SLIP)")
print(f"  Sv=10, SH=30, Sh=20 → {classify_anderson_regime(20, 30, 10)} (expect REVERSE)")

# 5d. from_principal with Euler angles
print("\n5d. Principal stress with Euler angles:")
# No rotation - should give diagonal tensor
stress_p0 = from_principal(S1=30, S2=20, S3=10, alpha=0, beta=0, gamma=0)
print(f"  No rotation (alpha=beta=gamma=0):")
print(f"  Components: {stress_p0.components}")
principals_p0, _ = stress_p0.principal_stresses()
print(f"  Recovered principals: {principals_p0}")

# 5e. from_principal_with_directions
print("\n5e. Principal stress with direction vectors:")
# S1 along x, S2 along y, S3 along z
stress_dir = from_principal_with_directions(
    S1=50, S2=30, S3=10,
    v1=[1, 0, 0]  # S1 along x
)
print(f"  S1=50 along x, S2=30, S3=10")
print(f"  Sxx={stress_dir.Sxx:.1f} (expect 50)")
print(f"  Stress ratio R: {stress_dir.stress_ratio():.4f} (expect 0.5)")

# 5f. StressTensor properties
print("\n5f. StressTensor methods:")
tensor = from_components(Sxx=20, Sxy=5, Sxz=0, Syy=15, Syz=0, Szz=10)
print(f"  Components: {tensor.components}")
print(f"  As matrix:\n{tensor.to_matrix()}")
principals, _ = tensor.principal_stresses()
print(f"  Principals: {principals}")
print(f"  Stress ratio: {tensor.stress_ratio():.4f}")


# =============================================================================
# PART 6: Parameterization Classes
# =============================================================================
section("PART 6: Parameterization Classes")

# 6a. AndersonParameterization
print("\n6a. AndersonParameterization:")
param_and = AndersonParameterization(
    Sh_range=(5, 20),
    SH_range=(15, 40),
    Sv_range=(25, 50),
    theta_range=(0, 180)
)
print(f"  Parameters: {param_and.param_names}")
print(f"  N params: {param_and.n_params}")

rng = np.random.default_rng(42)
sample = param_and.sample(rng)
print(f"  Sample: {sample}")
stress_sample = param_and.to_stress(sample)
print(f"  Stress: {stress_sample}")

# 6b. AndersonLithostaticParameterization
print("\n6b. AndersonLithostaticParameterization:")
param_litho = AndersonLithostaticParameterization(
    Sh_ratio_range=(0.4, 0.9),
    SH_ratio_range=(0.6, 1.2),
    theta_range=(0, 180),
    depth_range=(500, 2000),
    density=2700
)
print(f"  Parameters: {param_litho.param_names}")
sample_litho = param_litho.sample(rng)
print(f"  Sample: {sample_litho}")
stress_litho = param_litho.to_stress(sample_litho)
print(f"  Stress Szz (vertical): {stress_litho[5]/1e6:.2f} MPa")

# 6c. PrincipalParameterization
print("\n6c. PrincipalParameterization:")
param_princ = PrincipalParameterization(
    S1_range=(20, 50),
    S2_range=(10, 30),
    S3_range=(0, 15),
    alpha_range=(0, 360),
    beta_range=(0, 180),
    gamma_range=(0, 360)
)
print(f"  Parameters: {param_princ.param_names}")
sample_princ = param_princ.sample(rng)
print(f"  Sample S1={sample_princ['S1']:.1f}, S2={sample_princ['S2']:.1f}, S3={sample_princ['S3']:.1f}")

# 6d. DirectComponentParameterization
print("\n6d. DirectComponentParameterization:")
param_direct = DirectComponentParameterization(
    Sxx_range=(0, 50),
    Syy_range=(0, 50),
    Szz_range=(0, 50),
    Sxy_range=(-10, 10),
    Sxz_range=(-10, 10),
    Syz_range=(-10, 10)
)
print(f"  Parameters: {param_direct.param_names}")
sample_direct, stress_direct = param_direct.sample_stress(rng)
print(f"  Sample stress: {stress_direct}")


# =============================================================================
# PART 7: Direct Stress Inversion (DirectStressInversionModel)
# =============================================================================
section("PART 7: Direct Stress Inversion (DirectStressInversionModel)")

# 7a. Invert for Anderson stress from joint observations
print("\n7a. Invert Anderson stress from joints:")

# True stress: normal faulting, SH oriented N45E
true_Sh, true_SH, true_Sv, true_theta = 15, 25, 40, 45
true_stress_d = from_anderson(true_Sh, true_SH, true_Sv, true_theta)
_, true_dirs_d = true_stress_d.principal_stresses()
true_S3_dir = true_dirs_d[:, 2]  # S3 direction

print(f"  True params: Sh={true_Sh}, SH={true_SH}, Sv={true_Sv}, theta={true_theta}°")
print(f"  True S3 direction: {true_S3_dir}")

# Generate synthetic joint observations (normals ~ S3)
np.random.seed(42)
model_d = DirectStressInversionModel()
for _ in range(20):
    normal = true_S3_dir + np.random.randn(3) * 0.15
    normal /= np.linalg.norm(normal)
    model_d.add_joint(normal=normal)

print(f"  {model_d}")

# Invert with Anderson parameterization
param_inv = AndersonParameterization(
    Sh_range=(5, 30),
    SH_range=(10, 50),
    Sv_range=(20, 60),
    theta_range=(0, 180)
)

result_d = model_d.run(param_inv, n_iterations=30000, seed=123)
print(f"  Best cost: {result_d.best_cost:.6f}")
print(f"  Recovered params:")
for name in result_d.param_names:
    print(f"    {name}: {result_d.best_params[name]:.2f}")

# Compare S3 directions
rec_tensor = StressTensor(result_d.best_stress)
_, rec_dirs_d = rec_tensor.principal_stresses()
rec_S3 = rec_dirs_d[:, 2]
alignment = abs(np.dot(rec_S3, true_S3_dir))
print(f"  S3 alignment: {alignment:.4f}")

# 7b. Invert with mixed observations (joints + stylolites + direction/ratio)
print("\n7b. Mixed observations with Anderson inversion:")

# True stress
true_stress_m = from_anderson(Sh=10, SH=30, Sv=20, theta=60)
vals_m, dirs_m = true_stress_m.principal_stresses()
true_R_m = true_stress_m.stress_ratio()

print(f"  True: Sh=10, SH=30, Sv=20, theta=60°")
print(f"  True R: {true_R_m:.4f}")

model_m2 = DirectStressInversionModel()
np.random.seed(456)

# Add joints (S3 aligned)
for _ in range(10):
    normal = dirs_m[:, 2] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_m2.add_joint(normal=normal)

# Add stylolites (S1 aligned)
for _ in range(10):
    normal = dirs_m[:, 0] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_m2.add_stylolite(normal=normal)

# Add direction + ratio
for _ in range(5):
    direction = dirs_m[:, 0] + np.random.randn(3) * 0.1
    direction /= np.linalg.norm(direction)
    R = np.clip(true_R_m + np.random.randn() * 0.05, 0, 1)
    model_m2.add_stress_direction_and_ratio(direction=direction, R=R, principal_index=0)

print(f"  {model_m2}")

param_m2 = AndersonParameterization(
    Sh_range=(0, 40),
    SH_range=(10, 50),
    Sv_range=(5, 40),
    theta_range=(0, 180)
)

result_m2 = model_m2.run(param_m2, n_iterations=50000, seed=789)
print(f"  Best cost: {result_m2.best_cost:.6f}")
print(f"  Recovered Sh: {result_m2.best_params['Sh']:.1f} (true: 10)")
print(f"  Recovered SH: {result_m2.best_params['SH']:.1f} (true: 30)")
print(f"  Recovered Sv: {result_m2.best_params['Sv']:.1f} (true: 20)")
print(f"  Recovered theta: {result_m2.best_params['theta']:.1f}° (true: 60°)")

rec_tensor_m2 = StressTensor(result_m2.best_stress)
rec_R_m2 = rec_tensor_m2.stress_ratio()
print(f"  Recovered R: {rec_R_m2:.4f} (true: {true_R_m:.4f})")

# 7c. Invert with Principal parameterization
print("\n7c. Invert with PrincipalParameterization:")

# True stress with specific orientation
true_S1, true_S2, true_S3 = 50, 30, 10
true_stress_p = from_principal(true_S1, true_S2, true_S3, alpha=30, beta=45, gamma=0)
_, dirs_p = true_stress_p.principal_stresses()

model_p = DirectStressInversionModel()
np.random.seed(111)

# Add joints
for _ in range(15):
    normal = dirs_p[:, 2] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_p.add_joint(normal=normal)

# Add stylolites
for _ in range(15):
    normal = dirs_p[:, 0] + np.random.randn(3) * 0.1
    normal /= np.linalg.norm(normal)
    model_p.add_stylolite(normal=normal)

param_p = PrincipalParameterization(
    S1_range=(30, 70),
    S2_range=(15, 45),
    S3_range=(0, 25),
    alpha_range=(0, 360),
    beta_range=(0, 180),
    gamma_range=(0, 360)
)

result_p = model_p.run(param_p, n_iterations=50000, seed=222)
print(f"  True: S1={true_S1}, S2={true_S2}, S3={true_S3}")
print(f"  Best cost: {result_p.best_cost:.6f}")
print(f"  Recovered S1: {result_p.best_params['S1']:.1f}")
print(f"  Recovered S2: {result_p.best_params['S2']:.1f}")
print(f"  Recovered S3: {result_p.best_params['S3']:.1f}")

rec_tensor_p = StressTensor(result_p.best_stress)
_, rec_dirs_p = rec_tensor_p.principal_stresses()
print(f"  S1 alignment: {abs(np.dot(rec_dirs_p[:, 0], dirs_p[:, 0])):.4f}")
print(f"  S3 alignment: {abs(np.dot(rec_dirs_p[:, 2], dirs_p[:, 2])):.4f}")


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
print("  - stress_tensor.py: StressTensor, from_anderson, from_principal, etc.")
print("  - stress_tensor.py: Parameterization classes (Anderson, Principal, etc.)")
print("  - stress_inversion_model.py: DirectStressInversionModel")
