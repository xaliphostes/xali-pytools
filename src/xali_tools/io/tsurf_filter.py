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
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TSurfData:
    """Container for TSurf data."""
    positions: np.ndarray  # Flattened [x0, y0, z0, x1, y1, z1, ...]
    indices: np.ndarray    # Flattened [i0, j0, k0, i1, j1, k1, ...]
    properties: Dict[str, np.ndarray]  # Property name -> flattened values
    name: str = ""


def _parse_single_surface(lines: List[str], start_idx: int) -> tuple:
    """
    Parse a single surface starting at start_idx.

    Returns:
        Tuple of (TSurfData, end_index) where end_index is the line after END.
    """
    vertices = []
    triangles = []
    properties = {}
    property_names = []
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

        # Parse vertices (VRTX id x y z) - no properties
        if line.startswith('VRTX'):
            parts = line.split()
            vid = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            vertex_id_map[vid] = len(vertices)
            vertices.append((x, y, z))
            i += 1
            continue

        # Parse vertices with properties (PVRTX id x y z prop1 prop2 ...)
        if line.startswith('PVRTX'):
            parts = line.split()
            vid = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            vertex_id_map[vid] = len(vertices)
            vertices.append((x, y, z))

            # Parse property values
            prop_values = parts[5:]
            for j, prop_name in enumerate(property_names):
                if j < len(prop_values):
                    properties[prop_name].append(float(prop_values[j]))
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

    # Convert properties to flattened arrays
    prop_arrays = {}
    for prop_name, values in properties.items():
        if values:
            prop_arrays[prop_name] = np.array(values, dtype=np.float64)

    surface = TSurfData(
        positions=positions,
        indices=indices,
        properties=prop_arrays,
        name=name
    )

    return surface, i


def load_all_tsurf(filepath: str) -> List[TSurfData]:
    """
    Load all surfaces from a Gocad TSurf file.

    Args:
        filepath: Path to the TSurf file.

    Returns:
        List of TSurfData objects, one per surface in the file.

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
) -> TSurfData:
    """
    Load a Gocad TSurf file.

    Args:
        filepath: Path to the TSurf file.
        index: Index of the surface to load (0-based). Default is 0 (first surface).
        name: Name of the surface to load. If provided, index is ignored.

    Returns:
        TSurfData containing flattened positions, indices, and properties.

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


def _write_single_surface(f, data: TSurfData) -> None:
    """Write a single surface to an open file handle."""
    n_vertices = len(data.positions) // 3
    n_triangles = len(data.indices) // 3

    positions = data.positions.reshape(-1, 3)
    indices = data.indices.reshape(-1, 3)

    has_properties = bool(data.properties)
    prop_names = list(data.properties.keys()) if has_properties else []

    # Header
    f.write("GOCAD TSurf 1\n")
    f.write("HEADER {\n")
    f.write(f"name: {data.name or 'surface'}\n")
    f.write("}\n")

    # Property definitions
    if has_properties:
        f.write(f"PROPERTIES {' '.join(prop_names)}\n")

    f.write("TFACE\n")

    # Vertices
    for i in range(n_vertices):
        x, y, z = positions[i]
        if has_properties:
            prop_values = ' '.join(
                f"{data.properties[name][i]:.6g}" for name in prop_names
            )
            f.write(f"PVRTX {i + 1} {x:.6g} {y:.6g} {z:.6g} {prop_values}\n")
        else:
            f.write(f"VRTX {i + 1} {x:.6g} {y:.6g} {z:.6g}\n")

    # Triangles (convert 0-based to 1-based)
    for i in range(n_triangles):
        v1, v2, v3 = indices[i]
        f.write(f"TRGL {int(v1) + 1} {int(v2) + 1} {int(v3) + 1}\n")

    f.write("END\n")


def save_tsurf(data: TSurfData, filepath: str) -> None:
    """
    Save TSurfData to a Gocad TSurf file.

    Args:
        data: TSurfData object containing positions, indices, and properties.
        filepath: Output file path.
    """
    with open(filepath, 'w') as f:
        _write_single_surface(f, data)


def save_all_tsurf(surfaces: List[TSurfData], filepath: str) -> None:
    """
    Save multiple surfaces to a single Gocad TSurf file.

    Args:
        surfaces: List of TSurfData objects.
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
    tsurf_data = TSurfData(
        positions=np.asarray(data['positions'], dtype=np.float64),
        indices=np.asarray(data['indices'], dtype=np.uint32),
        properties=data.get('properties', {}),
        name=data.get('name', '')
    )
    save_tsurf(tsurf_data, filepath)


def save_all_tsurf_from_dict(surfaces: List[dict], filepath: str) -> None:
    """
    Save multiple surfaces (as dictionaries) to a single Gocad TSurf file.

    Args:
        surfaces: List of dictionaries, each with keys 'positions', 'indices',
                  and optionally 'properties' and 'name'.
        filepath: Output file path.
    """
    tsurf_list = [
        TSurfData(
            positions=np.asarray(s['positions'], dtype=np.float64),
            indices=np.asarray(s['indices'], dtype=np.uint32),
            properties=s.get('properties', {}),
            name=s.get('name', '')
        )
        for s in surfaces
    ]
    save_all_tsurf(tsurf_list, filepath)
