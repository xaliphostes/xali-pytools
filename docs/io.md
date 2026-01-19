# File I/O

## Unified Surface Loader

Load and save 3D mesh files using a single unified API with the `SurfaceData` structure:

```python
from xali_tools.io import load_surface, load_surfaces, save_surface, SurfaceData

# Load from any supported format (format detected by extension)
surface = load_surface("model.stl")
surface = load_surface("fault.ts")   # Gocad TSurf
surface = load_surface("mesh.obj")
surface = load_surface("cloud.ply")

print(f"Name: {surface.name}")
print(f"Vertices: {surface.n_vertices}")
print(f"Triangles: {surface.n_triangles}")

# Access data as flat arrays
positions = surface.positions  # [x0, y0, z0, x1, y1, z1, ...]
indices = surface.indices      # [i0, j0, k0, i1, j1, k1, ...]

# Or as (n, 3) matrices
vertices = surface.get_positions_matrix()  # Shape (n_vertices, 3)
faces = surface.get_indices_matrix()       # Shape (n_triangles, 3)

# Properties (normals, colors, TSurf attributes, etc.)
print(f"Properties: {list(surface.properties.keys())}")

# Load all surfaces from a file (multi-object OBJ, multi-surface TSurf, etc.)
surfaces = load_surfaces("model.obj")
surfaces = load_surfaces("faults.ts")
for surf in surfaces:
    print(f"{surf.name}: {surf.n_vertices} vertices, {surf.n_triangles} triangles")

# Load by name or index
surface = load_surface("model.obj", name="object_name")
surface = load_surface("faults.ts", index=1)

# Save to file (format inferred from extension)
save_surface(surface, "output.ply")
save_surface(surface, "output.ts")  # Gocad TSurf

# Save multiple surfaces
save_surfaces([surf1, surf2], "combined.ts")
save_surfaces([surf1, surf2], "combined.obj")
```

**Supported formats:**
- **Gocad TSurf** (`.ts`, `.tsurf`) - no extra dependencies
- **STL, OBJ, PLY, OFF, GLB, GLTF** and more - requires `trimesh`

**Installation:**
```sh
pip install xali-tools         # TSurf only
pip install xali-tools[mesh]   # All formats (includes trimesh)
```

---

## TSurf-Specific Functions

For direct access to TSurf-specific load/save functions:

```python
from xali_tools.io import SurfaceData, load_tsurf, save_tsurf, load_all_tsurf
import numpy as np

# Load using TSurf-specific functions
data = load_tsurf("surface.ts")

# Create a surface from scratch
new_surface = SurfaceData(
    positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64),
    indices=np.array([0, 1, 2], dtype=np.uint32),
    properties={"elevation": np.array([0.0, 0.5, 1.0])},
    name="triangle"
)
save_tsurf(new_surface, "triangle.ts")
```

> **Note:** `TSurfData` is an alias for `SurfaceData` for backward compatibility.
