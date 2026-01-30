# Core Data Structures

## Attribute

An `Attribute` encapsulates a numpy array with metadata about its semantic type (scalar, vector, tensor). It provides convenient iteration and access methods for scientific data.

### Basic Usage

```python
from xali_tools.core import Attribute, AttributeType
import numpy as np

# Scalar property (item_size=1)
pressure = Attribute([1.0, 2.0, 3.0], item_size=1, name="pressure")
print(pressure.n_items)      # 3
print(pressure.item_size)    # 1
print(pressure.attribute_type)   # "scalar"

# Vector property (item_size=3)
velocity = Attribute([1, 2, 3, 4, 5, 6], item_size=3, name="velocity")
print(velocity.n_items)      # 2
print(velocity.item_size)    # 3
print(velocity.attribute_type)   # "vector3"

# Symmetric 3x3 tensor (item_size=6: xx, xy, xz, yy, yz, zz)
stress = Attribute(np.zeros((10, 6)), item_size=6, name="stress")
print(stress.attribute_type)     # "sym_tensor3"

# Iteration
for v in velocity:
    print(v)  # [1, 2, 3], [4, 5, 6]

# Array access
print(velocity[0])           # [1. 2. 3.]
print(velocity.as_array())   # Full numpy array
```

### AttributeType Constants

For ambiguous `item_size` values (e.g., 3 could be a vector or a 2D symmetric tensor), you can explicitly specify the type:

| AttributeType | item_size | Description |
|-----------|-----------|-------------|
| `SCALAR` | 1 | Scalar value |
| `VECTOR2` | 2 | 2D vector (x, y) |
| `VECTOR3` | 3 | 3D vector (x, y, z) |
| `SYM_TENSOR2` | 3 | 2D symmetric tensor (xx, xy, yy) |
| `SYM_TENSOR3` | 6 | 3D symmetric tensor (xx, xy, xz, yy, yz, zz) |
| `TENSOR3` | 9 | Full 3x3 tensor |

```python
# Explicit type for ambiguous item_size=3
strain_2d = Attribute(data, item_size=3, attribute_type=AttributeType.SYM_TENSOR2)
```

### Creating Attributes

```python
# From list or array (makes a copy)
s = Attribute([1, 2, 3], item_size=1)
s = Attribute(np.array([[1,2,3], [4,5,6]]), item_size=3)

# From flat array
s = Attribute.from_flat([1, 2, 3, 4, 5, 6], item_size=3)  # 2 vectors

# As view (no copy, shares memory)
data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
s = Attribute.as_view(data, item_size=3)
s[0] = [10, 20, 30]  # Modifies original data!
```

### Properties

| Property | Description |
|----------|-------------|
| `name` | Property name |
| `item_size` | Number of components per item |
| `attribute_type` | Semantic type string |
| `n_items` | Number of items |
| `shape` | Shape of underlying array |
| `dtype` | Data type |
| `is_scalar` | True if item_size=1 |
| `is_vector` | True if item_size=3 |
| `is_tensor` | True if item_size=6 |

---

## AttributeManager

A container that holds multiple Attributes and provides automatic decomposition into derived properties. This enables querying for all available scalars, vectors, or tensors for visualization.

### Basic Usage

```python
from xali_tools.core import Attribute, AttributeManager
import numpy as np

# Create manager and add attributes
manager = AttributeManager()
manager.add("temperature", Attribute(np.random.rand(100), item_size=1))
manager.add("velocity", Attribute(np.random.rand(100, 3), item_size=3))
manager.add("stress", Attribute(np.random.rand(100, 6), item_size=6))

# Query available properties
print(manager.get_scalar_names())
# ['temperature', 'velocity:norm', 'velocity:x', 'velocity:y', 'velocity:z',
#  'stress:S1', 'stress:S2', 'stress:S3', 'stress:trace', 'stress:von_mises',
#  'stress:xx', 'stress:xy', 'stress:xz', 'stress:yy', 'stress:yz', 'stress:zz']

print(manager.get_vector3_names())
# ['velocity', 'stress:S1_vec', 'stress:S2_vec', 'stress:S3_vec', 'stress:principal_values']

# Get a derived property (lazy computed, cached)
von_mises = manager.get("stress:von_mises")
print(von_mises.n_items)  # 100

# Access original attributes
vel = manager.get("velocity")
```

### Query Methods

| Method | Returns |
|--------|---------|
| `get_scalar_names()` | All itemSize=1 property names |
| `get_vector2_names()` | All itemSize=2 property names |
| `get_vector3_names()` | All itemSize=3 property names |
| `get_sym_tensor3_names()` | All itemSize=6 property names |
| `get_tensor3_names()` | All itemSize=9 property names |
| `get_available(item_size=None)` | Dict of {name: item_size} |

### Derived Properties

Derived properties use the naming convention `base_name:suffix`:

**From Vector3 (item_size=3):**
- `velocity:x`, `velocity:y`, `velocity:z` - components
- `velocity:norm` - magnitude

**From SymTensor3 (item_size=6):**
- `stress:xx`, `stress:xy`, `stress:xz`, `stress:yy`, `stress:yz`, `stress:zz` - components
- `stress:trace` - tensor trace
- `stress:von_mises` - von Mises equivalent stress
- `stress:S1`, `stress:S2`, `stress:S3` - principal values (sorted by magnitude)
- `stress:S1_vec`, `stress:S2_vec`, `stress:S3_vec` - principal vectors
- `stress:principal_values` - all 3 principal values as vector3

### Use Case: Visualization

```python
# For iso-contouring (need scalars)
scalar_names = manager.get_scalar_names()
# User picks "stress:von_mises"
data = manager.get("stress:von_mises")

# For streamlines (need vectors)
vector_names = manager.get_vector3_names()
# User picks "stress:S1_vec" (max principal direction)
directions = manager.get("stress:S1_vec")

# Manager summary
print(manager.summary())
```

---

## Decomposer Architecture

The decomposition system is extensible. You can register custom decomposers.

### Built-in Decomposers

| Decomposer | Input Type | Produces |
|------------|------------|----------|
| `Vector2Decomposer` | vector2 | x, y, norm |
| `Vector3Decomposer` | vector3 | x, y, z, norm |
| `SymTensor2Decomposer` | sym_tensor2 | xx, xy, yy, trace |
| `SymTensor3Decomposer` | sym_tensor3 | xx...zz, trace, von_mises |
| `PrincipalDecomposer` | sym_tensor3 | S1, S2, S3, S1_vec, S2_vec, S3_vec |
| `Tensor3Decomposer` | tensor3 | all 9 components, trace, symmetric, antisymmetric |

### Custom Decomposer

```python
from xali_tools.core import Decomposer, DecomposerRegistry, Attribute, AttributeType
import numpy as np

class MagnitudeDecomposer(Decomposer):
    """Custom decomposer that computes magnitude for any vector."""

    @property
    def input_attribute_type(self) -> str:
        return AttributeType.VECTOR3

    @property
    def input_item_size(self) -> int:
        return 3

    def get_available(self, base_name: str):
        return {f"{base_name}:magnitude": 1}

    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        data = attribute.as_array()
        mag = np.linalg.norm(data, axis=1)
        return Attribute(mag, item_size=1, name=derived_name)

# Register the custom decomposer
registry = DecomposerRegistry.get_instance()
registry.register(MagnitudeDecomposer())
```

### Tensor Storage Conventions

**Symmetric 3x3 tensor (item_size=6):**
```
[xx, xy, xz, yy, yz, zz]

Matrix form:
| xx  xy  xz |
| xy  yy  yz |
| xz  yz  zz |
```

**Full 3x3 tensor (item_size=9, row-major):**
```
[xx, xy, xz, yx, yy, yz, zx, zy, zz]

Matrix form:
| xx  xy  xz |
| yx  yy  yz |
| zx  zy  zz |
```
