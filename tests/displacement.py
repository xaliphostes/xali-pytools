import numpy as np
from xali_tools.math.displacement import DisplacementProperties

# Example: 100 vertices with 3 displacement properties
n_vertices = 100

# Create displacement storage
dp = DisplacementProperties(n_vertices=n_vertices)

# Create sample displacement data (3 components per vertex)
# Displacement order: [ux, uy, uz]

# Remote displacement: uniform in x direction
remote = np.zeros(3 * n_vertices)
remote[0::3] = 1.0  # ux = 1 for all vertices

# Fault 1 displacement: vertical
fault1 = np.zeros(3 * n_vertices)
fault1[2::3] = 0.5  # uz = 0.5 for all vertices

# Fault 2 displacement: in y direction
fault2 = np.zeros(3 * n_vertices)
fault2[1::3] = -0.3  # uy = -0.3 for all vertices

# Add displacements with names
dp.add(remote, name="remote")
dp.add(fault1, name="fault_1")
dp.add(fault2, name="fault_2")

print(dp)
print()

# Weighted sum: 1.0*remote + 0.5*fault_1 + 0.3*fault_2
weights = [1.0, 0.5, 0.3]
combined = dp.weighted_sum(weights)

print("Weighted sum with weights:", weights)
combined_matrix = dp.to_matrix(displacement_array=combined)
print(f"Combined displacement at vertex 0: {combined_matrix[0]}")
# Expected: [1.0, -0.09, 0.25]
# Order: [ux, uy, uz]
#   ux = 1.0*1.0 + 0.5*0 + 0.3*0 = 1.0
#   uy = 1.0*0 + 0.5*0 + 0.3*(-0.3) = -0.09
#   uz = 1.0*0 + 0.5*0.5 + 0.3*0 = 0.25
print()

# Alternative: weighted sum by name
combined2 = dp.weighted_sum_by_name({
    "remote": 1.0,
    "fault_1": 0.5,
    "fault_2": 0.3
})
print("Same result using weighted_sum_by_name:")
print(f"Combined displacement at vertex 0: {dp.to_matrix(displacement_array=combined2)[0]}")
print()

# Extract a single component
ux = dp.component("remote", "x")
print(f"ux from remote displacement (first 5 vertices): {ux[:5]}")
print()

# Compute magnitude
mag = dp.magnitude("remote")
print(f"Magnitude of remote displacement (first 5 vertices): {mag[:5]}")
print()

# Get a specific displacement by name
fault1_disp = dp.get("fault_1")
print(f"fault_1 displacement shape: {fault1_disp.shape}")
