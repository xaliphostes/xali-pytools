"""
VTP (VTK PolyData) file loader and converter.

Loads triangulated surfaces from VTK PolyData format (.vtp) and converts
to SurfaceData or directly to Gocad TSurf format.

Requires: pyvista (pip install pyvista)

Examples:
```py
from xali_tools.io.vtp_filter import load_vtp, vtp_to_tsurf

# Load a VTP file as SurfaceData
surface = load_vtp("mesh.vtp")

# Convert VTP directly to Gocad TSurf
vtp_to_tsurf("input.vtp", "output.ts")
```
"""

import numpy as np
from typing import List, Optional, Dict

from xali_tools.geom import SurfaceData

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def _check_pyvista():
    """Check if pyvista is available."""
    if not HAS_PYVISTA:
        raise ImportError(
            "pyvista is required for VTP file support. "
            "Install it with: pip install pyvista"
        )


def _extract_triangles(mesh: "pv.PolyData") -> np.ndarray:
    """
    Extract triangle indices from a PyVista PolyData mesh.

    Handles meshes with mixed cell types by triangulating if needed.

    Returns:
        Flat array of triangle indices [i0, j0, k0, i1, j1, k1, ...]
    """
    # Triangulate to ensure all faces are triangles
    triangulated = mesh.triangulate()

    # Get faces array - format is [n, i0, i1, i2, n, i0, i1, i2, ...]
    faces = triangulated.faces

    if len(faces) == 0:
        return np.array([], dtype=np.uint32)

    # Parse the faces array
    triangles = []
    i = 0
    while i < len(faces):
        n_verts = faces[i]
        if n_verts == 3:
            triangles.append(faces[i+1:i+4])
        i += n_verts + 1

    if not triangles:
        return np.array([], dtype=np.uint32)

    return np.array(triangles, dtype=np.uint32).flatten()


def _extract_properties(mesh: "pv.PolyData") -> tuple:
    """
    Extract point data (vertex properties) from a PyVista mesh.

    Returns:
        Tuple of (properties dict, property_sizes dict)
    """
    properties = {}
    property_sizes = {}

    for name in mesh.point_data.keys():
        data = mesh.point_data[name]
        data = np.asarray(data, dtype=np.float64)

        # Determine item size
        if data.ndim == 1:
            item_size = 1
        else:
            item_size = data.shape[1] if data.shape[1] > 1 else 1

        properties[name] = data
        property_sizes[name] = item_size

    return properties, property_sizes


def load_vtp(filepath: str, name: Optional[str] = None) -> SurfaceData:
    """
    Load a VTP (VTK PolyData) file as SurfaceData.

    Args:
        filepath: Path to the VTP file.
        name: Optional surface name. If not provided, uses filename stem.

    Returns:
        SurfaceData containing positions, indices, and properties.

    Raises:
        ImportError: If pyvista is not installed.
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file contains no valid mesh data.
    """
    _check_pyvista()

    from pathlib import Path
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load with pyvista
    mesh = pv.read(filepath)

    if not isinstance(mesh, pv.PolyData):
        raise ValueError(f"Expected PolyData, got {type(mesh)}")

    if mesh.n_points == 0:
        raise ValueError(f"No vertices found in {filepath}")

    # Extract positions (flat array)
    positions = mesh.points.flatten().astype(np.float64)

    # Extract triangles
    indices = _extract_triangles(mesh)

    if len(indices) == 0:
        raise ValueError(f"No triangles found in {filepath}")

    # Extract properties
    properties, property_sizes = _extract_properties(mesh)

    # Surface name
    surface_name = name if name else path.stem

    return SurfaceData(
        positions=positions,
        indices=indices,
        properties=properties,
        property_sizes=property_sizes,
        name=surface_name
    )


def load_all_vtp(filepath: str) -> List[SurfaceData]:
    """
    Load a VTP file. Returns a list for API consistency.

    VTP files typically contain a single surface, but this function
    returns a list for consistency with other loaders.

    Args:
        filepath: Path to the VTP file.

    Returns:
        List containing one SurfaceData object.
    """
    return [load_vtp(filepath)]


def vtp_to_tsurf(
    input_path: str,
    output_path: str,
    name: Optional[str] = None
) -> None:
    """
    Convert a VTP file to Gocad TSurf format.

    Args:
        input_path: Path to input VTP file.
        output_path: Path to output TSurf file (.ts).
        name: Optional surface name for the TSurf file.

    Example:
        vtp_to_tsurf("mesh.vtp", "mesh.ts")
    """
    from .tsurf_filter import save_tsurf

    surface = load_vtp(input_path, name=name)
    save_tsurf(surface, output_path)


def get_vtp_info(filepath: str) -> Dict:
    """
    Get information about a VTP file without fully loading it.

    Args:
        filepath: Path to the VTP file.

    Returns:
        Dictionary with file information:
        - n_points: Number of vertices
        - n_cells: Number of cells (faces)
        - bounds: Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
        - point_data: List of point data array names
        - cell_data: List of cell data array names
    """
    _check_pyvista()

    mesh = pv.read(filepath)

    return {
        'n_points': mesh.n_points,
        'n_cells': mesh.n_cells,
        'bounds': list(mesh.bounds),
        'point_data': list(mesh.point_data.keys()),
        'cell_data': list(mesh.cell_data.keys()),
    }
