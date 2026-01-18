# Xali Python Tools

## Installation

```sh
pip install xali-tools
```

Or for development:
```sh
pip install -e .
```

## Examples

### Plotting Streamlines

```python
import numpy as np
from xali_tools.plots.streamlines import plotStreamlinesFromFlatArray

# Create a sample vector field on a 20x20 grid
n, m = 20, 20
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, m)
xx, yy = np.meshgrid(x, y)

# Simple rotational field: vx = -y, vy = x, vz = 0
vx = -yy.flatten()
vy = xx.flatten()
vz = np.zeros(n * m)

# Flatten to [vx0, vy0, vz0, vx1, vy1, vz1, ...]
vector_field = np.column_stack([vx, vy, vz]).flatten()

# Plot streamlines
plotStreamlinesFromFlatArray(
    vector_field,
    n=n, m=m,
    x_range=(-2, 2),
    y_range=(-2, 2),
    density=1.5,
    title="Rotational Flow"
)
```

### Plotting Vector Field (Quiver)

```python
import numpy as np
from xali_tools.plots.vectorfield import plotVectorFieldFromFlatArray

# Create a sample vector field on a 20x20 grid
n, m = 20, 20
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, m)
xx, yy = np.meshgrid(x, y)

# Simple rotational field: vx = -y, vy = x, vz = 0
vx = -yy.flatten()
vy = xx.flatten()
vz = np.zeros(n * m)

# Flatten to [vx0, vy0, vz0, vx1, vy1, vz1, ...]
vector_field = np.column_stack([vx, vy, vz]).flatten()

# Plot vector field with arrows
plotVectorFieldFromFlatArray(
    vector_field,
    n=n, m=m,
    x_range=(-2, 2),
    y_range=(-2, 2),
    skip=1,                    # Plot every arrow (use 2+ for sparser grids)
    normalize_arrows=False,    # True to show direction only
    title="Rotational Vector Field"
)
```

### Plotting Iso-contours

```python
import numpy as np
from xali_tools.plots.isocontours import plotIsoContoursFromFlatArray

# Create a sample scalar field on a 50x50 grid
n, m = 50, 50
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, m)
xx, yy = np.meshgrid(x, y)

# Gaussian-like scalar field
scalar_field = np.exp(-(xx**2 + yy**2)).flatten()

# Plot iso-contours
plotIsoContoursFromFlatArray(
    scalar_field,
    n=n, m=m,
    x_range=(-2, 2),
    y_range=(-2, 2),
    num_levels=15,
    cmap="viridis",
    title="Gaussian Field"
)
```

### Loading and Saving TSurf Files

```python
from xali_tools.io.tsurf_filter import load_tsurf, save_tsurf, TSurfData
import numpy as np

# Load a Gocad TSurf file (first surface)
data = load_tsurf("surface.ts")

print(f"Name: {data.name}")
print(f"Vertices: {len(data.positions) // 3}")
print(f"Triangles: {len(data.indices) // 3}")
print(f"Properties: {list(data.properties.keys())}")

# Access the data
positions = data.positions  # [x0, y0, z0, x1, y1, z1, ...]
indices = data.indices      # [i0, j0, k0, i1, j1, k1, ...]

# Save to a new file
save_tsurf(data, "output.ts")

# Create a new surface from scratch
new_surface = TSurfData(
    positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64),
    indices=np.array([0, 1, 2], dtype=np.uint32),
    properties={"elevation": np.array([0.0, 0.5, 1.0])},
    name="triangle"
)
save_tsurf(new_surface, "triangle.ts")
```

### Working with Multiple Surfaces

A single TSurf file can contain multiple surfaces. Use these functions to handle them:

```python
from xali_tools.io.tsurf_filter import (
    load_all_tsurf, load_tsurf, save_all_tsurf
)

# Load all surfaces from a file
surfaces = load_all_tsurf("model.ts")
print(f"Found {len(surfaces)} surfaces")

for surf in surfaces:
    n_verts = len(surf.positions) // 3
    n_tris = len(surf.indices) // 3
    print(f"  {surf.name}: {n_verts} vertices, {n_tris} triangles")

# Load a specific surface by index (0-based)
second_surface = load_tsurf("model.ts", index=1)

# Load a specific surface by name
fault = load_tsurf("model.ts", name="fault_1")

# Save multiple surfaces to a single file
save_all_tsurf(surfaces, "combined.ts")
```

### Displacement Field Operations

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

### Stress Tensor Operations

Store and combine multiple stress properties at mesh vertices:

```python
import numpy as np
from xali_tools.math.stress import StressProperties

# Create storage for 100 vertices
sp = StressProperties(n_vertices=100)

# Add stress properties (each is a flat array of 6*n_vertices)
# Stress components: [σxx, σxy, σxz, σyy, σyz, σzz]
remote = np.zeros(6 * 100)
remote[0::6] = 10.0  # σxx = 10 for all vertices

fault1 = np.zeros(6 * 100)
fault1[3::6] = 5.0   # σxy = 5 for all vertices

sp.add(remote, name="remote")
sp.add(fault1, name="fault_1")

# Weighted sum: 1.0*remote + 0.5*fault_1
combined = sp.weighted_sum([1.0, 0.5])

# Or by name
combined = sp.weighted_sum_by_name({
    "remote": 1.0,
    "fault_1": 0.5
})

# Extract a single component (e.g., σxx)
sigma_xx = sp.component("remote", "xx")

# Reshape to (m, 6) matrix
matrix = sp.to_matrix(stress_array=combined)
print(f"Stress at vertex 0: {matrix[0]}")
```

## Generating the Wheel

Install the necessary packages:
```sh
pip install build
```

Create the wheel:
```sh
python -m build
```

This creates `dist/xali_tools-0.1.0-py3-none-any.whl`.
