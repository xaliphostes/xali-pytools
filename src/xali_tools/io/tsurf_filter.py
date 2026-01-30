"""
Gocad TSurf file loader.

Loads triangulated surfaces from Gocad TSurf format and returns
flattened arrays for positions, indices, and vertex properties.
Supports files with multiple surfaces.

Examples:
```py
from xali_tools.io.tsurf_filter import load_tsurf, load_all_tsurf

# Load a single surface (first one in file)
data = load_tsurf("surface.ts")

# Load all surfaces from a file
surfaces = load_all_tsurf("multi_surface.ts")
for surf in surfaces:
    print(surf.name, len(surf.positions) // 3, "vertices")

# Load a specific surface by index or name
data = load_tsurf("multi_surface.ts", index=2)
data = load_tsurf("multi_surface.ts", name="fault_1")
```
"""

import numpy as np
from typing import List, Optional

from xali_tools.geom import SurfaceData


def _parse_single_surface(lines: List[str], start_idx: int) -> tuple:
    """
    Parse a single surface starting at start_idx.

    Returns:
        Tuple of (SurfaceData, end_index) where end_index is the line after END.
    """
    vertices = []
    triangles = []
    properties = {}
    property_names = []  # Raw property names from PROPERTIES line
    property_sizes = []  # Sizes from ESIZES line (1 for scalar, >1 for vector)
    vertex_id_map = {}
    name = ""

    i = start_idx
    in_header = False

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue

        # Skip the GOCAD TSurf header line (case-insensitive)
        if line.upper().startswith('GOCAD TSURF'):
            i += 1
            continue

        # Parse header block (handles "HEADER {" and "HEADER: {")
        if line.startswith('HEADER'):
            in_header = True
            i += 1
            continue

        if in_header:
            if '}' in line:
                in_header = False
            else:
                # Handle both "name: value" and "name= value" formats
                stripped = line.strip()
                if stripped.startswith('name:'):
                    name = stripped.split(':', 1)[1].strip()
                elif stripped.startswith('name='):
                    name = stripped.split('=', 1)[1].strip()
            i += 1
            continue

        # Parse property definitions
        if line.startswith('PROPERTIES'):
            parts = line.split()[1:]
            property_names = parts
            for prop_name in property_names:
                properties[prop_name] = []
            i += 1
            continue

        # Parse ESIZES (dimensions of each property)
        # Example: PROPERTIES U a b  with  ESIZES 3 1 1
        # means: U is a 3D vector (3 values), a and b are scalars (1 value each)
        # Sum of ESIZES = total number of values after vertex coordinates
        if line.startswith('ESIZES'):
            parts = line.split()[1:]
            property_sizes = [int(s) for s in parts]
            i += 1
            continue

        # Parse vertices with properties (PVRTX id x y z prop1 prop2 ...)
        # Note: Must check PVRTX before VRTX since PVRTX starts with VRTX
        if line.startswith('PVRTX'):
            parts = line.split()
            vid = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            vertex_id_map[vid] = len(vertices)
            vertices.append((x, y, z))

            # Parse property values using ESIZES
            prop_values = parts[5:]
            value_idx = 0

            for j, prop_name in enumerate(property_names):
                # Get size for this property (default to 1 if no ESIZES)
                size = property_sizes[j] if j < len(property_sizes) else 1

                if size == 1:
                    # Scalar property
                    if value_idx < len(prop_values):
                        try:
                            properties[prop_name].append(float(prop_values[value_idx]))
                        except ValueError:
                            properties[prop_name].append(0.0)
                    value_idx += 1
                else:
                    # Vector property - collect 'size' values
                    vec_values = []
                    for _ in range(size):
                        if value_idx < len(prop_values):
                            try:
                                vec_values.append(float(prop_values[value_idx]))
                            except ValueError:
                                vec_values.append(0.0)
                        else:
                            vec_values.append(0.0)
                        value_idx += 1
                    properties[prop_name].append(vec_values)
            i += 1
            continue

        # Parse vertices without properties (VRTX id x y z)
        if line.startswith('VRTX'):
            parts = line.split()
            vid = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            vertex_id_map[vid] = len(vertices)
            vertices.append((x, y, z))
            i += 1
            continue

        # Parse triangles (TRGL id1 id2 id3)
        if line.startswith('TRGL'):
            parts = line.split()
            v1, v2, v3 = int(parts[1]), int(parts[2]), int(parts[3])
            triangles.append((
                vertex_id_map[v1],
                vertex_id_map[v2],
                vertex_id_map[v3]
            ))
            i += 1
            continue

        # Handle TFACE marker
        if line.startswith('TFACE'):
            i += 1
            continue

        # End of this surface
        if line.startswith('END'):
            i += 1
            break

        i += 1

    if not vertices:
        return None, i

    if not triangles:
        return None, i

    # Convert to flattened numpy arrays
    positions = np.array(vertices, dtype=np.float64).flatten()
    indices = np.array(triangles, dtype=np.uint32).flatten()

    # Process properties with ESIZES grouping
    prop_arrays, prop_sizes_dict = _group_properties_by_esizes(
        properties, property_names, property_sizes
    )

    surface = SurfaceData(
        positions=positions,
        indices=indices,
        properties=prop_arrays,
        property_sizes=prop_sizes_dict,
        name=name
    )

    return surface, i


def _group_properties_by_esizes(
    properties: dict,
    property_names: List[str],
    property_sizes: List[int]
) -> tuple:
    """
    Convert properties to numpy arrays using ESIZES information.

    With ESIZES, each property name corresponds to one size value:
        PROPERTIES U a b
        ESIZES 3 1 1

    Results in:
        - "U": array of shape (n_vertices, 3) - vector property
        - "a": array of shape (n_vertices,) - scalar property
        - "b": array of shape (n_vertices,) - scalar property

    Returns:
        Tuple of (property_arrays dict, property_sizes dict).
    """
    prop_arrays = {}
    prop_sizes_dict = {}

    if not property_names:
        return prop_arrays, prop_sizes_dict

    for i, prop_name in enumerate(property_names):
        if prop_name not in properties or not properties[prop_name]:
            continue

        values = properties[prop_name]
        size = property_sizes[i] if i < len(property_sizes) else 1

        if size == 1:
            # Scalar property - values is a list of floats
            prop_arrays[prop_name] = np.array(values, dtype=np.float64)
        else:
            # Vector property - values is a list of lists
            prop_arrays[prop_name] = np.array(values, dtype=np.float64)

        prop_sizes_dict[prop_name] = size

    return prop_arrays, prop_sizes_dict


def load_all_tsurf(filepath: str) -> List[SurfaceData]:
    """
    Load all surfaces from a Gocad TSurf file.

    Args:
        filepath: Path to the TSurf file.

    Returns:
        List of SurfaceData objects, one per surface in the file.

    Raises:
        ValueError: If no valid surfaces are found.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    surfaces = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for start of a new surface (case-insensitive)
        if line.upper().startswith('GOCAD TSURF'):
            surface, i = _parse_single_surface(lines, i)
            if surface is not None:
                surfaces.append(surface)
        else:
            i += 1

    if not surfaces:
        raise ValueError(f"No valid surfaces found in {filepath}")

    return surfaces


def load_tsurf(
    filepath: str,
    index: int = 0,
    name: Optional[str] = None
) -> SurfaceData:
    """
    Load a Gocad TSurf file.

    Args:
        filepath: Path to the TSurf file.
        index: Index of the surface to load (0-based). Default is 0 (first surface).
        name: Name of the surface to load. If provided, index is ignored.

    Returns:
        SurfaceData containing flattened positions, indices, and properties.

    Raises:
        ValueError: If the file format is invalid or surface not found.
    """
    surfaces = load_all_tsurf(filepath)

    if name is not None:
        for surface in surfaces:
            if surface.name == name:
                return surface
        available = [s.name for s in surfaces]
        raise ValueError(f"Surface '{name}' not found. Available: {available}")

    if index < 0 or index >= len(surfaces):
        raise ValueError(f"Index {index} out of range. File contains {len(surfaces)} surface(s).")

    return surfaces[index]


def load_tsurf_as_dict(
    filepath: str,
    index: int = 0,
    name: Optional[str] = None
) -> dict:
    """
    Load a Gocad TSurf file and return as a dictionary.

    Args:
        filepath: Path to the TSurf file.
        index: Index of the surface to load (0-based).
        name: Name of the surface to load. If provided, index is ignored.

    Returns:
        Dictionary with keys: 'positions', 'indices', 'properties', 'name'
    """
    data = load_tsurf(filepath, index=index, name=name)
    return {
        'positions': data.positions,
        'indices': data.indices,
        'properties': data.properties,
        'name': data.name
    }


def load_all_tsurf_as_dict(filepath: str) -> List[dict]:
    """
    Load all surfaces from a Gocad TSurf file as dictionaries.

    Args:
        filepath: Path to the TSurf file.

    Returns:
        List of dictionaries, each with keys: 'positions', 'indices', 'properties', 'name'
    """
    surfaces = load_all_tsurf(filepath)
    return [
        {
            'positions': s.positions,
            'indices': s.indices,
            'properties': s.properties,
            'name': s.name
        }
        for s in surfaces
    ]

def get_tsurf(data: SurfaceData) -> str:
    """Write a single surface to an open file handle."""
    n_vertices = len(data.positions) // 3
    n_triangles = len(data.indices) // 3

    positions = data.positions.reshape(-1, 3)
    indices = data.indices.reshape(-1, 3)

    has_properties = bool(data.properties)

    # Header
    s = "GOCAD TSurf 1\n"
    s += "HEADER {\n"
    s += f"name: {data.name or 'surface'}\n"
    s += "}\n"

    # Prepare property information for ESIZES support
    # Each entry: (name, esize, values)
    property_info = []

    if has_properties:
        for name, values in data.properties.items():
            values = np.asarray(values)
            # Use property_sizes if available, otherwise infer from shape
            esize = data.property_sizes.get(name, None)
            if esize is None:
                # Infer from array shape
                esize = values.shape[1] if values.ndim == 2 and values.shape[1] > 1 else 1

            if esize > 1:
                # Vector/tensor property - reshape to (n_vertices, esize)
                values = values.reshape(-1, esize)
                property_info.append((name, esize, values))
            else:
                # Scalar property
                flat_values = values.flatten()
                property_info.append((name, 1, flat_values))

        # Build PROPERTIES and ESIZES lines
        prop_names = [name for name, _, _ in property_info]
        esizes = [str(esize) for _, esize, _ in property_info]

        s += f"PROPERTIES {' '.join(prop_names)}\n"
        s += f"ESIZES {' '.join(esizes)}\n"

    s += "TFACE\n"

    # Vertices
    for i in range(n_vertices):
        x, y, z = positions[i]
        if has_properties:
            # Collect all property values for this vertex
            prop_values_list = []
            for name, esize, values in property_info:
                if esize > 1:
                    # Vector property - write all components
                    for j in range(esize):
                        prop_values_list.append(f"{values[i, j]:.6g}")
                else:
                    # Scalar property
                    prop_values_list.append(f"{values[i]:.6g}")
            prop_values = ' '.join(prop_values_list)
            s += f"PVRTX {i + 1} {x:.6g} {y:.6g} {z:.6g} {prop_values}\n"
        else:
            s += f"VRTX {i + 1} {x:.6g} {y:.6g} {z:.6g}\n"

    # Triangles (convert 0-based to 1-based)
    for i in range(n_triangles):
        v1, v2, v3 = indices[i]
        s += f"TRGL {int(v1) + 1} {int(v2) + 1} {int(v3) + 1}\n"

    s += "END\n"
    return s

def _write_single_surface(f, data: SurfaceData) -> None:
    f.write(get_tsurf(data))


def save_tsurf(data: SurfaceData, filepath: str) -> None:
    """
    Save SurfaceData to a Gocad TSurf file.

    Args:
        data: SurfaceData object containing positions, indices, and properties.
        filepath: Output file path.
    """
    with open(filepath, 'w') as f:
        _write_single_surface(f, data)


def save_all_tsurf(surfaces: List[SurfaceData], filepath: str) -> None:
    """
    Save multiple surfaces to a single Gocad TSurf file.

    Args:
        surfaces: List of SurfaceData objects.
        filepath: Output file path.
    """
    with open(filepath, 'w') as f:
        for surface in surfaces:
            _write_single_surface(f, surface)


def save_tsurf_from_dict(data: dict, filepath: str) -> None:
    """
    Save a dictionary to a Gocad TSurf file.

    Args:
        data: Dictionary with keys 'positions', 'indices', and optionally
              'properties' and 'name'.
        filepath: Output file path.
    """
    surface_data = SurfaceData(
        positions=np.asarray(data['positions'], dtype=np.float64),
        indices=np.asarray(data['indices'], dtype=np.uint32),
        properties=data.get('properties', {}),
        name=data.get('name', '')
    )
    save_tsurf(surface_data, filepath)


def save_all_tsurf_from_dict(surfaces: List[dict], filepath: str) -> None:
    """
    Save multiple surfaces (as dictionaries) to a single Gocad TSurf file.

    Args:
        surfaces: List of dictionaries, each with keys 'positions', 'indices',
                  and optionally 'properties' and 'name'.
        filepath: Output file path.
    """
    surface_list = [
        SurfaceData(
            positions=np.asarray(s['positions'], dtype=np.float64),
            indices=np.asarray(s['indices'], dtype=np.uint32),
            properties=s.get('properties', {}),
            name=s.get('name', '')
        )
        for s in surfaces
    ]
    save_all_tsurf(surface_list, filepath)
