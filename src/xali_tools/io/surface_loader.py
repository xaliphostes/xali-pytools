"""
Unified surface loader for common 3D mesh formats.

Supports: STL, OBJ, PLY, OFF, Gocad TSurf (.ts), and many more via trimesh.

Usage:
    from xali_tools.io.surface_loader import load_surfaces, SurfaceData

    # Load all surfaces from a file (any supported format)
    surfaces = load_surfaces("model.obj")
    surfaces = load_surfaces("fault.ts")  # Gocad TSurf

    for surf in surfaces:
        print(f"{surf.name}: {surf.n_vertices} vertices, {surf.n_triangles} triangles")
        print(f"Properties: {list(surf.properties.keys())}")
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from xali_tools.geom import SurfaceData

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# TSurf extensions (case-insensitive)
TSURF_EXTENSIONS = {'.ts', '.tsurf'}

# VTP extensions (case-insensitive)
VTP_EXTENSIONS = {'.vtp'}


def _is_tsurf_file(filepath: str) -> bool:
    """Check if file is a Gocad TSurf file by extension."""
    return Path(filepath).suffix.lower() in TSURF_EXTENSIONS


def _is_vtp_file(filepath: str) -> bool:
    """Check if file is a VTP file by extension."""
    return Path(filepath).suffix.lower() in VTP_EXTENSIONS


def _trimesh_to_surface_data(mesh: "trimesh.Trimesh", name: str = "surface") -> SurfaceData:
    """Convert a trimesh object to SurfaceData."""
    # Positions as flat array
    positions = mesh.vertices.flatten().astype(np.float64)

    # Indices as flat array (if faces exist)
    if mesh.faces is not None and len(mesh.faces) > 0:
        indices = mesh.faces.flatten().astype(np.uint32)
    else:
        indices = None

    # Extract properties
    properties = {}

    # Vertex colors
    if mesh.visual is not None:
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors
            if len(colors) == mesh.vertices.shape[0]:
                properties['vertex_colors'] = colors.astype(np.float64)

    # Vertex normals
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        try:
            normals = mesh.vertex_normals
            if len(normals) == mesh.vertices.shape[0]:
                properties['vertex_normals'] = normals.flatten().astype(np.float64)
        except Exception:
            pass  # Some meshes don't have valid normals

    # Face normals
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        try:
            face_normals = mesh.face_normals
            if mesh.faces is not None and len(face_normals) == len(mesh.faces):
                properties['face_normals'] = face_normals.flatten().astype(np.float64)
        except Exception:
            pass

    # Vertex metadata (if available)
    if hasattr(mesh, 'metadata') and mesh.metadata:
        for key, value in mesh.metadata.items():
            if isinstance(value, np.ndarray):
                properties[f'metadata_{key}'] = value

    return SurfaceData(
        positions=positions,
        indices=indices,
        properties=properties,
        name=name
    )


def load_surfaces(
    filepath: str,
    force_mesh: bool = True
) -> List[SurfaceData]:
    """
    Load surfaces from a file.

    Supports: STL, OBJ, PLY, OFF, GLB, GLTF, Gocad TSurf (.ts), and many more.

    Args:
        filepath: Path to the mesh file.
        force_mesh: If True, convert point clouds to meshes if possible.
                   (Only applies to trimesh formats, not TSurf.)

    Returns:
        List of SurfaceData objects.

    Raises:
        ImportError: If trimesh is not installed (for non-TSurf formats).
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.

    Example:
        surfaces = load_surfaces("model.obj")
        surfaces = load_surfaces("fault.ts")  # Gocad TSurf
        for surf in surfaces:
            print(f"{surf.name}: {surf.n_vertices} vertices")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Handle TSurf files natively (no trimesh required)
    if _is_tsurf_file(filepath):
        from .tsurf_filter import load_all_tsurf
        return load_all_tsurf(filepath)

    # Handle VTP files with pyvista
    if _is_vtp_file(filepath):
        from .vtp_filter import load_all_vtp
        return load_all_vtp(filepath)

    # For other formats, use trimesh
    if not HAS_TRIMESH:
        raise ImportError(
            "trimesh is required for loading mesh files. "
            "Install it with: pip install trimesh"
        )

    # Load with trimesh
    loaded = trimesh.load(filepath, force='scene' if not force_mesh else None)

    surfaces = []

    # Handle different return types from trimesh
    if isinstance(loaded, trimesh.Scene):
        # Multiple meshes in scene
        for name, geometry in loaded.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                surfaces.append(_trimesh_to_surface_data(geometry, name=name))
            elif isinstance(geometry, trimesh.PointCloud):
                # Point cloud (no faces)
                surf = SurfaceData(
                    positions=geometry.vertices.flatten().astype(np.float64),
                    indices=None,
                    properties={},
                    name=name
                )
                if geometry.colors is not None:
                    surf.properties['vertex_colors'] = geometry.colors.astype(np.float64)
                surfaces.append(surf)
    elif isinstance(loaded, trimesh.Trimesh):
        # Single mesh
        name = path.stem
        surfaces.append(_trimesh_to_surface_data(loaded, name=name))
    elif isinstance(loaded, trimesh.PointCloud):
        # Point cloud
        name = path.stem
        surf = SurfaceData(
            positions=loaded.vertices.flatten().astype(np.float64),
            indices=None,
            properties={},
            name=name
        )
        if loaded.colors is not None:
            surf.properties['vertex_colors'] = loaded.colors.astype(np.float64)
        surfaces.append(surf)
    else:
        raise ValueError(f"Unsupported geometry type: {type(loaded)}")

    if not surfaces:
        raise ValueError(f"No surfaces found in file: {filepath}")

    return surfaces


def load_surface(
    filepath: str,
    index: int = 0,
    name: Optional[str] = None
) -> SurfaceData:
    """
    Load a single surface from a file.

    Args:
        filepath: Path to the mesh file.
        index: Index of surface to load (0-based).
        name: If provided, load surface with this name.

    Returns:
        SurfaceData object.

    Example:
        surface = load_surface("model.obj")
        print(f"Loaded {surface.n_vertices} vertices")
    """
    surfaces = load_surfaces(filepath)

    if name is not None:
        for surf in surfaces:
            if surf.name == name:
                return surf
        raise ValueError(f"No surface named '{name}' found in file")

    if index >= len(surfaces):
        raise IndexError(f"Surface index {index} out of range (file has {len(surfaces)} surfaces)")

    return surfaces[index]


def save_surface(
    surface: SurfaceData,
    filepath: str,
    file_type: Optional[str] = None
) -> None:
    """
    Save a surface to a file.

    Args:
        surface: SurfaceData to save.
        filepath: Output file path.
        file_type: File format (inferred from extension if not provided).
                   Supported: 'stl', 'obj', 'ply', 'off', 'glb', 'ts', etc.

    Example:
        save_surface(surface, "output.stl")
        save_surface(surface, "output.ts")  # Gocad TSurf
    """
    if surface.indices is None:
        raise ValueError("Cannot save point cloud as mesh (no indices)")

    # Handle TSurf files natively
    if _is_tsurf_file(filepath) or file_type in ('ts', 'tsurf'):
        from .tsurf_filter import save_tsurf
        save_tsurf(surface, filepath)
        return

    # For other formats, use trimesh
    if not HAS_TRIMESH:
        raise ImportError(
            "trimesh is required for saving mesh files. "
            "Install it with: pip install trimesh"
        )

    # Convert to trimesh
    vertices = surface.get_positions_matrix()
    faces = surface.get_indices_matrix()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Export
    mesh.export(filepath, file_type=file_type)


def save_surfaces(
    surfaces: List[SurfaceData],
    filepath: str,
    file_type: Optional[str] = None
) -> None:
    """
    Save multiple surfaces to a file.

    Note: Not all formats support multiple meshes. OBJ, GLB/GLTF, and TSurf do.

    Args:
        surfaces: List of SurfaceData to save.
        filepath: Output file path.
        file_type: File format (inferred from extension if not provided).

    Example:
        save_surfaces([surf1, surf2], "output.obj")
        save_surfaces([surf1, surf2], "output.ts")  # Gocad TSurf
    """
    # Handle TSurf files natively
    if _is_tsurf_file(filepath) or file_type in ('ts', 'tsurf'):
        from .tsurf_filter import save_all_tsurf
        # Filter out point clouds (surfaces without indices)
        mesh_surfaces = [surf for surf in surfaces if surf.indices is not None]
        save_all_tsurf(mesh_surfaces, filepath)
        return

    # For other formats, use trimesh
    if not HAS_TRIMESH:
        raise ImportError(
            "trimesh is required for saving mesh files. "
            "Install it with: pip install trimesh"
        )

    # Create scene with multiple meshes
    scene = trimesh.Scene()

    for surf in surfaces:
        if surf.indices is None:
            continue  # Skip point clouds

        vertices = surf.get_positions_matrix()
        faces = surf.get_indices_matrix()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        scene.add_geometry(mesh, node_name=surf.name)

    scene.export(filepath, file_type=file_type)
