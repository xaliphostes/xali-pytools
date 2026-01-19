# Math Operations

## Displacement Field Operations

Store and combine multiple displacement properties at mesh vertices:

```python
import numpy as np
from xali_tools.math.displacement import DisplacementProperties

# Create storage for 100 vertices
dp = DisplacementProperties(n_vertices=100)

# Add displacement properties (each is a flat array of 3*n_vertices)
# Displacement components: [ux, uy, uz]
remote = np.zeros(3 * 100)
remote[0::3] = 1.0  # ux = 1 for all vertices

fault1 = np.zeros(3 * 100)
fault1[2::3] = 0.5  # uz = 0.5 for all vertices

dp.add(remote, name="remote")
dp.add(fault1, name="fault_1")

# Weighted sum: 1.0*remote + 0.5*fault_1
combined = dp.weighted_sum([1.0, 0.5])

# Or by name
combined = dp.weighted_sum_by_name({
    "remote": 1.0,
    "fault_1": 0.5
})

# Extract a single component (e.g., ux)
ux = dp.component("remote", "x")

# Compute magnitude at each vertex
mag = dp.magnitude(displacement_array=combined)

# Reshape to (m, 3) matrix
matrix = dp.to_matrix(displacement_array=combined)
print(f"Displacement at vertex 0: {matrix[0]}")
```

## Stress Tensor Operations

Store and combine multiple stress properties at mesh vertices:

```python
import numpy as np
from xali_tools.math.stress import StressProperties

# Create storage for 100 vertices
sp = StressProperties(n_vertices=100)

# Add stress properties (each is a flat array of 6*n_vertices)
# Stress components: [sxx, sxy, sxz, syy, syz, szz]
remote = np.zeros(6 * 100)
remote[0::6] = 10.0  # sxx = 10 for all vertices

fault1 = np.zeros(6 * 100)
fault1[3::6] = 5.0   # sxy = 5 for all vertices

sp.add(remote, name="remote")
sp.add(fault1, name="fault_1")

# Weighted sum: 1.0*remote + 0.5*fault_1
combined = sp.weighted_sum([1.0, 0.5])

# Or by name
combined = sp.weighted_sum_by_name({
    "remote": 1.0,
    "fault_1": 0.5
})

# Extract a single component (e.g., sxx)
sigma_xx = sp.component("remote", "xx")

# Reshape to (m, 6) matrix
matrix = sp.to_matrix(stress_array=combined)
print(f"Stress at vertex 0: {matrix[0]}")
```
