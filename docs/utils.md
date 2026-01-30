# Utilities

## Surface Info

Display comprehensive information about loaded 3D surfaces, including geometry statistics, bounding box, triangle/edge metrics, memory usage, and property analysis.

### Basic Usage

```python
from xali_tools.io import load_surfaces
from xali_tools.utils import print_surface_info, surface_info

# Load and display surface information
surfaces = load_surfaces("model.ts")
print_surface_info(surfaces)

# Brief output (less detail)
print_surface_info(surfaces, verbose=False)

# Get information as dictionary for programmatic use
info = surface_info(surfaces[0])
print(info["n_vertices"])
print(info["bounding_box"]["extent"])
```

### Command Line Usage

```bash
# Full verbose output
python -m xali_tools.utils.surface_info model.ts

# Brief output
python -m xali_tools.utils.surface_info model.ts --brief
```

### Example Output

```
============================================================
Surface: fault_1
============================================================

Geometry:
  Vertices:  12,450
  Triangles: 24,562

Bounding Box:
  Min:    [125000.000000, 4500000.000000, -3500.000000]
  Max:    [135000.000000, 4520000.000000, -500.000000]
  Center: [130000.000000, 4510000.000000, -2000.000000]
  Extent: [10000.000000, 20000.000000, 3000.000000]

Triangle Statistics:
  Total area: 125000000.000000
  Area range: [150.234000, 2500.450000]
  Area mean:  5089.123000 (std: 1234.567000)

Edge Length Statistics:
  Range: [25.123000, 150.456000]
  Mean:  75.234000 (std: 25.678000)

Memory Usage:
  Total: 2.45 MB (2,568,192 bytes)
    Positions:  298,800 bytes
    Indices:    294,744 bytes
    Properties: 1,974,648 bytes

Properties (3):

  'elevation' (scalar, 12450 items):
    Range: [-3500.000000, -500.000000]
    Mean:  -2000.123000 (std: 850.456000)

  'slip' (vector3, 12450 items):
    Range: [-2.500000, 3.200000]
    Mean:  0.125000 (std: 1.234000)
    Magnitude range: [0.001234, 3.456789]
    Magnitude mean:  1.234567

  'stress' (sym_tensor3, 12450 items):
    Range: [-50.000000, 120.000000]
    Mean:  25.123000 (std: 35.456000)
    Magnitude range: [10.123000, 145.678000]
    Magnitude mean:  65.432100
```

### Information Returned

The `surface_info()` function returns a dictionary with the following structure:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Surface name |
| `n_vertices` | `int` | Number of vertices |
| `n_triangles` | `int` | Number of triangles |
| `has_indices` | `bool` | Whether surface has triangle indices |
| `bounding_box` | `dict` | Min, max, center, extent arrays |
| `memory` | `dict` | Memory usage breakdown in bytes |
| `triangles` | `dict` | Triangle area statistics (if indexed) |
| `edges` | `dict` | Edge length statistics (if indexed) |
| `properties` | `dict` | Per-property statistics |

### Property Statistics

For each property, the following statistics are computed:

| Field | Description |
|-------|-------------|
| `item_size` | Components per item (1=scalar, 3=vector, 6=tensor) |
| `type` | Human-readable type name |
| `n_items` | Number of items |
| `dtype` | NumPy data type |
| `shape` | Array shape |
| `min`, `max` | Value range |
| `mean`, `std` | Statistical moments |
| `has_nan` | True if contains NaN values |
| `has_inf` | True if contains infinite values |
| `magnitude_*` | Magnitude statistics (for vectors/tensors) |

### Programmatic Access

```python
from xali_tools.utils import surface_info

info = surface_info(surface)

# Access bounding box
bbox = info["bounding_box"]
print(f"Surface spans {bbox['extent']} units")
print(f"Center at {bbox['center']}")

# Check total surface area
if "triangles" in info:
    print(f"Total area: {info['triangles']['total_area']:.2f}")

# Analyze properties
for name, prop in info["properties"].items():
    print(f"{name}: {prop['type']}, range [{prop['min']:.3f}, {prop['max']:.3f}]")
    if prop["has_nan"]:
        print(f"  WARNING: {name} contains NaN values!")

# Memory usage
mem_mb = info["memory"]["total_bytes"] / (1024 * 1024)
print(f"Memory usage: {mem_mb:.2f} MB")
```

### Working with Multiple Surfaces

```python
from xali_tools.io import load_surfaces
from xali_tools.utils import print_surface_info, surface_info

# Load all surfaces
surfaces = load_surfaces("faults.ts")

# Print info for all surfaces
print_surface_info(surfaces)

# Aggregate statistics
total_vertices = sum(surface_info(s)["n_vertices"] for s in surfaces)
total_area = sum(surface_info(s)["triangles"]["total_area"] for s in surfaces)
print(f"Total: {len(surfaces)} surfaces, {total_vertices} vertices, area={total_area:.2f}")
```
