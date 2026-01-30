"""
Utility to display comprehensive information about a loaded 3D surface.
"""

import numpy as np
from typing import Dict, Any, List, Union
from xali_tools.io import SurfaceData


def _compute_bounding_box(positions: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute bounding box from flat positions array."""
    coords = positions.reshape(-1, 3)
    return {
        "min": coords.min(axis=0),
        "max": coords.max(axis=0),
        "center": coords.mean(axis=0),
        "extent": coords.max(axis=0) - coords.min(axis=0),
    }


def _compute_triangle_areas(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area of each triangle."""
    coords = positions.reshape(-1, 3)
    tris = indices.reshape(-1, 3)

    v0 = coords[tris[:, 0]]
    v1 = coords[tris[:, 1]]
    v2 = coords[tris[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas


def _compute_edge_lengths(positions: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    """Compute edge length statistics."""
    coords = positions.reshape(-1, 3)
    tris = indices.reshape(-1, 3)

    v0 = coords[tris[:, 0]]
    v1 = coords[tris[:, 1]]
    v2 = coords[tris[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    all_edges = np.concatenate([e0, e1, e2])
    return {
        "min": float(all_edges.min()),
        "max": float(all_edges.max()),
        "mean": float(all_edges.mean()),
        "std": float(all_edges.std()),
    }


def _property_type_name(item_size: int) -> str:
    """Get human-readable property type name."""
    type_names = {
        1: "scalar",
        2: "vector2",
        3: "vector3",
        6: "sym_tensor3",
        9: "tensor3",
    }
    return type_names.get(item_size, f"array({item_size})")


def _format_number(value: float) -> str:
    """Format number for display."""
    if value == 0:
        return "0.000000"
    if abs(value) < 1e-6 or abs(value) > 1e6:
        return f"{value:.6e}"
    return f"{value:.6f}"


def surface_info(surface: SurfaceData) -> Dict[str, Any]:
    """
    Gather comprehensive information about a 3D surface.

    Args:
        surface: The SurfaceData object to analyze.

    Returns:
        Dictionary containing all surface information.
    """
    info = {
        "name": surface.name,
        "n_vertices": surface.n_vertices,
        "n_triangles": surface.n_triangles,
        "has_indices": surface.indices is not None,
    }

    # Bounding box
    info["bounding_box"] = _compute_bounding_box(surface.positions)

    # Memory usage
    mem_positions = surface.positions.nbytes
    mem_indices = surface.indices.nbytes if surface.indices is not None else 0
    mem_properties = sum(p.nbytes for p in surface.properties.values())
    info["memory"] = {
        "positions_bytes": mem_positions,
        "indices_bytes": mem_indices,
        "properties_bytes": mem_properties,
        "total_bytes": mem_positions + mem_indices + mem_properties,
    }

    # Triangle statistics (if we have indices)
    if surface.indices is not None and len(surface.indices) > 0:
        areas = _compute_triangle_areas(surface.positions, surface.indices)
        info["triangles"] = {
            "total_area": float(areas.sum()),
            "area_min": float(areas.min()),
            "area_max": float(areas.max()),
            "area_mean": float(areas.mean()),
            "area_std": float(areas.std()),
        }
        info["edges"] = _compute_edge_lengths(surface.positions, surface.indices)

    # Property information
    info["properties"] = {}
    for name in surface.properties:
        data = surface.properties[name]
        item_size = surface.property_sizes.get(name, 1)
        flat_data = data.flatten()

        prop_info = {
            "item_size": item_size,
            "type": _property_type_name(item_size),
            "n_items": len(flat_data) // item_size,
            "dtype": str(data.dtype),
            "shape": data.shape,
            "min": float(flat_data.min()),
            "max": float(flat_data.max()),
            "mean": float(flat_data.mean()),
            "std": float(flat_data.std()),
            "has_nan": bool(np.isnan(flat_data).any()),
            "has_inf": bool(np.isinf(flat_data).any()),
            "memory_bytes": data.nbytes,
        }

        # For vectors, compute magnitude statistics
        if item_size > 1:
            reshaped = flat_data.reshape(-1, item_size)
            magnitudes = np.linalg.norm(reshaped, axis=1)
            prop_info["magnitude_min"] = float(magnitudes.min())
            prop_info["magnitude_max"] = float(magnitudes.max())
            prop_info["magnitude_mean"] = float(magnitudes.mean())

        info["properties"][name] = prop_info

    return info


def print_surface_info(
    surface: Union[SurfaceData, List[SurfaceData]],
    verbose: bool = True
) -> None:
    """
    Print comprehensive information about a 3D surface.

    Args:
        surface: The SurfaceData object(s) to analyze.
        verbose: If True, print detailed statistics. If False, print summary only.
    """
    surfaces = [surface] if isinstance(surface, SurfaceData) else surface

    for surf in surfaces:
        info = surface_info(surf)

        print("=" * 60)
        print(f"Surface: {info['name']}")
        print("=" * 60)

        # Basic geometry
        print(f"\nGeometry:")
        print(f"  Vertices:  {info['n_vertices']:,}")
        print(f"  Triangles: {info['n_triangles']:,}")

        # Bounding box
        bbox = info["bounding_box"]
        print(f"\nBounding Box:")
        print(f"  Min:    [{_format_number(bbox['min'][0])}, "
              f"{_format_number(bbox['min'][1])}, {_format_number(bbox['min'][2])}]")
        print(f"  Max:    [{_format_number(bbox['max'][0])}, "
              f"{_format_number(bbox['max'][1])}, {_format_number(bbox['max'][2])}]")
        print(f"  Center: [{_format_number(bbox['center'][0])}, "
              f"{_format_number(bbox['center'][1])}, {_format_number(bbox['center'][2])}]")
        print(f"  Extent: [{_format_number(bbox['extent'][0])}, "
              f"{_format_number(bbox['extent'][1])}, {_format_number(bbox['extent'][2])}]")

        # Triangle statistics
        if "triangles" in info:
            tri = info["triangles"]
            print(f"\nTriangle Statistics:")
            print(f"  Total area: {_format_number(tri['total_area'])}")
            if verbose:
                print(f"  Area range: [{_format_number(tri['area_min'])}, "
                      f"{_format_number(tri['area_max'])}]")
                print(f"  Area mean:  {_format_number(tri['area_mean'])} "
                      f"(std: {_format_number(tri['area_std'])})")

            edges = info["edges"]
            print(f"\nEdge Length Statistics:")
            print(f"  Range: [{_format_number(edges['min'])}, {_format_number(edges['max'])}]")
            if verbose:
                print(f"  Mean:  {_format_number(edges['mean'])} (std: {_format_number(edges['std'])})")

        # Memory usage
        mem = info["memory"]
        total_mb = mem["total_bytes"] / (1024 * 1024)
        print(f"\nMemory Usage:")
        print(f"  Total: {total_mb:.2f} MB ({mem['total_bytes']:,} bytes)")
        if verbose:
            print(f"    Positions:  {mem['positions_bytes']:,} bytes")
            print(f"    Indices:    {mem['indices_bytes']:,} bytes")
            print(f"    Properties: {mem['properties_bytes']:,} bytes")

        # Properties
        if info["properties"]:
            print(f"\nProperties ({len(info['properties'])}):")
            for name, prop in info["properties"].items():
                print(f"\n  '{name}' ({prop['type']}, {prop['n_items']} items):")
                print(f"    Range: [{_format_number(prop['min'])}, {_format_number(prop['max'])}]")
                if verbose:
                    print(f"    Mean:  {_format_number(prop['mean'])} (std: {_format_number(prop['std'])})")
                    if "magnitude_mean" in prop:
                        print(f"    Magnitude range: [{_format_number(prop['magnitude_min'])}, "
                              f"{_format_number(prop['magnitude_max'])}]")
                        print(f"    Magnitude mean:  {_format_number(prop['magnitude_mean'])}")
                    if prop["has_nan"]:
                        print(f"    WARNING: Contains NaN values!")
                    if prop["has_inf"]:
                        print(f"    WARNING: Contains infinite values!")
        else:
            print(f"\nProperties: None")

        print()


if __name__ == "__main__":
    import sys
    from xali_tools.io import load_surfaces

    if len(sys.argv) < 2:
        print("Usage: python -m xali_tools.utils.surface_info <surface_file>")
        print("       python -m xali_tools.utils.surface_info <surface_file> --brief")
        sys.exit(1)

    filepath = sys.argv[1]
    verbose = "--brief" not in sys.argv

    try:
        surfaces = load_surfaces(filepath)
        print_surface_info(surfaces, verbose=verbose)
    except Exception as e:
        print(f"Error loading surface: {e}")
        sys.exit(1)
