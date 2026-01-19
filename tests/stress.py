import numpy as np
from xali_tools.math.stress import StressProperties

# Example: 100 vertices with 3 stress properties
n_vertices = 100

# Create stress storage
sp = StressProperties(n_vertices=n_vertices)

# Create sample stress data (6 components per vertex)
# Stress order: [σxx, σxy, σxz, σyy, σyz, σzz]

# Remote stress: uniform tension in x direction
remote = np.zeros(6 * n_vertices)
remote[0::6] = 10.0  # σxx = 10 for all vertices

# Fault 1 stress: shear stress
fault1 = np.zeros(6 * n_vertices)
fault1[1::6] = 5.0   # σxy = 5 for all vertices

# Fault 2 stress: compression in y direction
fault2 = np.zeros(6 * n_vertices)
fault2[3::6] = -3.0  # σyy = -3 for all vertices

# Add stresses with names
sp.add(remote, name="remote")
sp.add(fault1, name="fault_1")
sp.add(fault2, name="fault_2")

print(sp)
print()

# Weighted sum: 1.0*remote + 0.5*fault_1 + 0.3*fault_2
weights = [1.0, 0.5, 0.3]
combined = sp.weighted_sum(weights)


print("Weighted sum with weights:", weights)
combined_matrix = sp.to_matrix(stress_array=combined)
print(f"Combined stress at vertex 0: {combined_matrix[0]}")
# Expected: [10, 2.5, 0, -0.9, 0, 0]
# Order: [σxx, σxy, σxz, σyy, σyz, σzz]
#   σxx = 1.0*10 + 0.5*0 + 0.3*0 = 10
#   σxy = 1.0*0 + 0.5*5 + 0.3*0 = 2.5
#   σyy = 1.0*0 + 0.5*0 + 0.3*(-3) = -0.9
print()

# Alternative: weighted sum by name
combined2 = sp.weighted_sum_by_name({
    "remote": 1.0,
    "fault_1": 0.5,
    "fault_2": 0.3
})
print("Same result using weighted_sum_by_name:")
print(f"Combined stress at vertex 0: {sp.to_matrix(stress_array=combined2)[0]}")
print()

# Extract a single component
sigma_xx = sp.component("remote", "xx")
print(f"σxx from remote stress (first 5 vertices): {sigma_xx[:5]}")
print()

# Get a specific stress by name
fault1_stress = sp.get("fault_1")
print(f"fault_1 stress shape: {fault1_stress.shape}")
print()

# =============================================
# Principal values and directions
# =============================================
print("=" * 50)
print("Principal stress analysis")
print("=" * 50)

# Create a simple stress state for testing
# Pure tension in x: σxx = 10, all others = 0
# Expected principal values: σ1=10, σ2=0, σ3=0
# Expected σ1 direction: [1, 0, 0]
simple_stress = np.zeros(6 * n_vertices)
simple_stress[0::6] = 10.0  # σxx = 10

sp2 = StressProperties(n_vertices=n_vertices)
sp2.add(simple_stress, name="tension_x")

# Get principal values (flat array)
pv = sp2.principal_values("tension_x")
print(f"Principal values (flat) shape: {pv.shape}")
print(f"Principal values at vertex 0: {pv[:3]}")  # [10, 0, 0]

# Get principal values as matrix
pv_matrix = sp2.principal_values_matrix("tension_x")
print(f"Principal values matrix shape: {pv_matrix.shape}")
print(f"σ1 for first 5 vertices: {pv_matrix[:5, 0]}")
print()

# Get principal directions (flat array)
pd = sp2.principal_directions("tension_x")
print(f"Principal directions (flat) shape: {pd.shape}")
print(f"Principal directions at vertex 0: {pd[:9]}")

# Get principal directions as matrix
pd_matrix = sp2.principal_directions_matrix("tension_x")
print(f"Principal directions matrix shape: {pd_matrix.shape}")
print(f"σ1 direction at vertex 0: {pd_matrix[0, 0, :]}")  # Should be [1, 0, 0] or [-1, 0, 0]
print()

# Test with a more complex stress state
# Biaxial: σxx = 10, σyy = 5
biaxial = np.zeros(6 * n_vertices)
biaxial[0::6] = 10.0  # σxx
biaxial[3::6] = 5.0   # σyy

sp2.add(biaxial, name="biaxial")
pv_biax = sp2.principal_values_matrix("biaxial")
print(f"Biaxial stress principal values at vertex 0: {pv_biax[0]}")  # [10, 5, 0]
