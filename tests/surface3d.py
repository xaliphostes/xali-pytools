"""
Test 3D triangulated surface plotting with iso-contours using PyVista.
"""
import numpy as np
from xali_tools.plots import (
    create_pyvista_mesh,
    plot_triangulated_surface,
    plot_surface_with_isocontours,
    plot_surface_with_colored_contours,
    plot_multiple_surfaces,
    plot_surface_vectors,
)


def create_test_mesh(n=20):
    """
    Create a test triangulated surface (a wavy surface).

    Returns vertices, triangles, and a scalar field.
    """
    # Create a grid of points
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)

    # Wavy surface
    Z = np.sin(X * 2) * np.cos(Y * 2) + 0.5 * np.sin(X + Y)

    # Flatten to vertex array
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Create triangles from grid
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            # Two triangles per grid cell
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles)

    # Scalar field: distance from center
    scalar_field = np.sqrt(X.ravel()**2 + Y.ravel()**2)

    return vertices, triangles, scalar_field


def create_hemisphere_mesh(n=30):
    """
    Create a hemisphere mesh for testing.
    """
    # Spherical coordinates
    phi = np.linspace(0, np.pi/2, n)  # 0 to 90 degrees (hemisphere)
    theta = np.linspace(0, 2*np.pi, 2*n)

    PHI, THETA = np.meshgrid(phi, theta)

    # Spherical to Cartesian
    R = 1.0
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Create triangles
    triangles = []
    n_theta = 2 * n
    n_phi = n

    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            v0 = i * n_phi + j
            v1 = v0 + 1
            v2 = v0 + n_phi
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles)

    # Scalar field: latitude (z-coordinate)
    scalar_field = vertices[:, 2]

    return vertices, triangles, scalar_field


if __name__ == "__main__":
    print("=" * 60)
    print("Test 3D Surface Plotting with PyVista")
    print("=" * 60)

    # Create test mesh
    print("\nCreating test mesh (wavy surface)...")
    vertices, triangles, scalar_field = create_test_mesh(n=30)
    print(f"  Vertices: {vertices.shape}")
    print(f"  Triangles: {triangles.shape}")
    print(f"  Scalar field: min={scalar_field.min():.2f}, max={scalar_field.max():.2f}")

    # Test 1: Basic triangulated surface
    print("\n1. Basic triangulated surface (colored by scalar field)...")
    plot_triangulated_surface(
        vertices, triangles, scalar_field,
        cmap="plasma",
        show_edges=True,
        edge_color="gray",
        opacity=0.9,
        title="Wavy Surface - Distance from Center",
        screenshot="test_pyvista_basic.png",
        show=False
    )
    print("  Saved: test_pyvista_basic.png")

    # Test 2: Surface with black iso-contours
    print("\n2. Surface with iso-contour lines...")
    plot_surface_with_isocontours(
        vertices, triangles, scalar_field,
        num_levels=12,
        surface_cmap="coolwarm",
        contour_color="black",
        contour_width=3.0,
        surface_opacity=0.7,
        title="Wavy Surface with Iso-contours",
        screenshot="test_pyvista_isocontours.png",
        show=False
    )
    print("  Saved: test_pyvista_isocontours.png")

    # Test 3: Surface with colored iso-contours
    print("\n3. Surface with colored iso-contours...")
    plot_surface_with_colored_contours(
        vertices, triangles, scalar_field,
        num_levels=15,
        cmap="viridis",
        contour_width=4.0,
        surface_opacity=0.3,
        title="Wavy Surface with Colored Iso-contours",
        screenshot="test_pyvista_colored_contours.png",
        show=False
    )
    print("  Saved: test_pyvista_colored_contours.png")

    # Test 4: Contours only (no surface)
    print("\n4. Iso-contours only (no surface)...")
    plot_surface_with_isocontours(
        vertices, triangles, scalar_field,
        num_levels=20,
        contour_color="darkblue",
        contour_width=2.0,
        show_surface=False,
        title="Iso-contours Only",
        screenshot="test_pyvista_contours_only.png",
        show=False
    )
    print("  Saved: test_pyvista_contours_only.png")

    # Test 5: Hemisphere with latitude contours
    print("\n5. Hemisphere with latitude contours...")
    verts_h, tris_h, scalar_h = create_hemisphere_mesh(n=25)
    print(f"  Hemisphere: {verts_h.shape[0]} vertices, {tris_h.shape[0]} triangles")

    plot_surface_with_isocontours(
        verts_h, tris_h, scalar_h,
        num_levels=10,
        surface_cmap="terrain",
        contour_color="black",
        contour_width=2.0,
        surface_opacity=0.85,
        title="Hemisphere with Latitude Contours",
        screenshot="test_pyvista_hemisphere.png",
        show=False
    )
    print("  Saved: test_pyvista_hemisphere.png")

    # Test 6: Using z-coordinate as scalar (default)
    print("\n6. Surface colored by z-coordinate (default)...")
    plot_triangulated_surface(
        vertices, triangles,  # No scalar field provided
        cmap="terrain",
        title="Surface Colored by Elevation (Z)",
        screenshot="test_pyvista_elevation.png",
        show=False
    )
    print("  Saved: test_pyvista_elevation.png")

    # Test 7: Surface with vectors
    print("\n7. Surface with vector field...")
    # Create normal vectors pointing outward
    normals = np.zeros_like(vertices)
    normals[:, 2] = 1.0  # Simple upward normals
    # Add some variation
    normals[:, 0] = 0.2 * np.sin(vertices[:, 0])
    normals[:, 1] = 0.2 * np.cos(vertices[:, 1])
    # Normalize
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    plot_surface_vectors(
        vertices, triangles, normals * 0.1,
        scalar_field=scalar_field,
        vector_scale=1.0,
        vector_color="red",
        surface_cmap="viridis",
        surface_opacity=0.6,
        title="Surface with Vector Field",
        screenshot="test_pyvista_vectors.png",
        show=False
    )
    print("  Saved: test_pyvista_vectors.png")

    # Test 8: Multiple surfaces
    print("\n8. Multiple surfaces in one view...")
    # Create a second surface (shifted)
    vertices2 = vertices.copy()
    vertices2[:, 2] += 2  # Shift up

    plot_multiple_surfaces(
        meshes=[
            (vertices, triangles, scalar_field),
            (vertices2, triangles, scalar_field),
        ],
        cmaps=["viridis", "plasma"],
        opacities=[0.8, 0.8],
        show_edges=False,
        screenshot="test_pyvista_multiple.png",
        show=False
    )
    print("  Saved: test_pyvista_multiple.png")

    # Test 9: Create PyVista mesh directly
    print("\n9. Direct PyVista mesh creation...")
    mesh = create_pyvista_mesh(vertices, triangles)
    mesh["distance"] = scalar_field
    print(f"  Mesh: {mesh}")
    print(f"  N points: {mesh.n_points}")
    print(f"  N cells: {mesh.n_cells}")
    print(f"  Arrays: {list(mesh.array_names)}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

    # Clean up test files
    import os
    for f in [
        "test_pyvista_basic.png",
        "test_pyvista_isocontours.png",
        "test_pyvista_colored_contours.png",
        "test_pyvista_contours_only.png",
        "test_pyvista_hemisphere.png",
        "test_pyvista_elevation.png",
        "test_pyvista_vectors.png",
        "test_pyvista_multiple.png",
    ]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Cleaned up: {f}")
