# File I/O

## SurfaceData

The `SurfaceData` dataclass is the unified structure for 3D mesh data across all file formats.

### Structure

```python
from xali_tools.io import SurfaceData
import numpy as np

# Create a surface from scratch
surface = SurfaceData(
    positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64),  # Flat [x0,y0,z0,...]
    indices=np.array([0, 1, 2], dtype=np.uint32),  # Flat [i0,j0,k0,...]
    properties={"elevation": np.array([0.0, 0.5, 1.0])},
    property_sizes={"elevation": 1},  # 1=scalar, 3=vector, 6=tensor
    name="triangle"
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `positions` | `np.ndarray` | Flat vertex positions `[x0, y0, z0, x1, y1, z1, ...]` |
| `indices` | `np.ndarray` or `None` | Flat triangle indices `[i0, j0, k0, ...]` |
| `properties` | `Dict[str, np.ndarray]` | Named properties (scalars, vectors, tensors) |
| `property_sizes` | `Dict[str, int]` | Item size for each property (1=scalar, 3=vector, 6=tensor) |
| `name` | `str` | Surface name |

### Properties and Methods

```python
# Basic info
print(surface.n_vertices)    # Number of vertices
print(surface.n_triangles)   # Number of triangles
print(surface.property_names())  # List of property names

# Get data as matrices
vertices = surface.get_positions_matrix()  # Shape (n_vertices, 3)
faces = surface.get_indices_matrix()       # Shape (n_triangles, 3)

# Get property as Attribute (view, no copy)
from xali_tools.core import Attribute
elevation = surface.get_property("elevation")  # Returns Attribute
print(elevation.item_size)  # 1

# Set property
surface.set_property("normals", normal_data, item_size=3)
```

### Integration with AttributeManager

Use `AttributeManager` to access derived properties (components, principal values, etc.):

```python
from xali_tools.io import load_surface
from xali_tools.core import AttributeManager

# Load surface with stress tensor property
surface = load_surface("fault.ts")

# Create manager from surface properties
manager = AttributeManager()
for name in surface.property_names():
    manager.add(name, surface.get_property(name))

# Query available scalars for visualization
scalars = manager.get_scalar_names()
# ['stress:xx', 'stress:von_mises', 'stress:S1', ...]

# Get derived property
von_mises = manager.get("stress:von_mises")
```

---

## Unified Surface Loader

Load and save 3D mesh files using a single unified API:

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

# Load all surfaces from a TSurf file
surfaces = load_all_tsurf("faults.ts")

# Create a surface from scratch
new_surface = SurfaceData(
    positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64),
    indices=np.array([0, 1, 2], dtype=np.uint32),
    properties={"elevation": np.array([0.0, 0.5, 1.0])},
    property_sizes={"elevation": 1},
    name="triangle"
)
save_tsurf(new_surface, "triangle.ts")
```
