"""
3D triangulated surface plotting with iso-contours using PyVista.
"""
import numpy as np
from typing import Tuple, List, Optional

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def _check_pyvista():
    if not PYVISTA_AVAILABLE:
        raise ImportError(
            "pyvista is required for 3D surface plotting. "
            "Install it with: pip install pyvista"
        )


def create_pyvista_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray
) -> "pv.PolyData":
    """
    Create a PyVista PolyData mesh from vertices and triangles.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : np.ndarray
        Array of shape (n_triangles, 3) containing vertex indices.

    Returns
    -------
    pv.PolyData
        PyVista mesh object.
    """
    _check_pyvista()

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    # PyVista expects faces as [n_points, p1, p2, p3, n_points, p1, ...]
    n_triangles = len(triangles)
    faces = np.column_stack([
        np.full(n_triangles, 3),
        triangles
    ]).ravel()

    mesh = pv.PolyData(vertices, faces)
    return mesh


def plot_triangulated_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    scalar_field: np.ndarray = None,
    scalar_name: str = "scalar",
    cmap: str = "viridis",
    show_edges: bool = False,
    edge_color: str = "black",
    opacity: float = 1.0,
    clim: Tuple[float, float] = None,
    show_scalar_bar: bool = True,
    title: str = None,
    window_size: Tuple[int, int] = (1024, 768),
    background: str = "white",
    plotter: "pv.Plotter" = None,
    show: bool = True,
    screenshot: str = None
) -> "pv.Plotter":
    """
    Plot a 3D triangulated surface with optional scalar field coloring.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : np.ndarray
        Array of shape (n_triangles, 3) containing vertex indices.
    scalar_field : np.ndarray, optional
        Array of size n_vertices containing scalar values.
        If None, surface is colored by z-coordinate.
    scalar_name : str
        Name for the scalar field (shown in scalar bar).
    cmap : str
        Colormap name.
    show_edges : bool
        Whether to show mesh edges.
    edge_color : str
        Color of mesh edges.
    opacity : float
        Surface opacity (0-1).
    clim : tuple, optional
        (min, max) for color scaling.
    show_scalar_bar : bool
        Whether to show the scalar bar.
    title : str, optional
        Plot title.
    window_size : tuple
        Window size (width, height).
    background : str
        Background color.
    plotter : pv.Plotter, optional
        Existing plotter to add mesh to.
    show : bool
        Whether to display the plot.
    screenshot : str, optional
        Path to save screenshot.

    Returns
    -------
    pv.Plotter
        The plotter object.
    """
    _check_pyvista()

    mesh = create_pyvista_mesh(vertices, triangles)

    if scalar_field is None:
        scalar_field = vertices[:, 2]  # Color by z-coordinate
        scalar_name = "elevation"

    mesh[scalar_name] = np.asarray(scalar_field)

    if plotter is None:
        off_screen = screenshot is not None and not show
        plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
        plotter.set_background(background)

    plotter.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap=cmap,
        show_edges=show_edges,
        edge_color=edge_color,
        opacity=opacity,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={"title": scalar_name}
    )

    if title:
        plotter.add_title(title)

    if screenshot and not show:
        plotter.show(screenshot=screenshot)
    elif screenshot and show:
        plotter.show(screenshot=screenshot)
    elif show:
        plotter.show()

    return plotter


def plot_surface_with_isocontours(
    vertices: np.ndarray,
    triangles: np.ndarray,
    scalar_field: np.ndarray,
    num_levels: int = 10,
    levels: np.ndarray = None,
    surface_cmap: str = "viridis",
    surface_opacity: float = 0.8,
    contour_color: str = "black",
    contour_width: float = 3.0,
    show_surface: bool = True,
    clim: Tuple[float, float] = None,
    show_scalar_bar: bool = True,
    title: str = None,
    window_size: Tuple[int, int] = (1024, 768),
    background: str = "white",
    plotter: "pv.Plotter" = None,
    show: bool = True,
    screenshot: str = None
) -> "pv.Plotter":
    """
    Plot a 3D triangulated surface with iso-contour lines.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : np.ndarray
        Array of shape (n_triangles, 3) containing vertex indices.
    scalar_field : np.ndarray
        Array of size n_vertices containing scalar values.
    num_levels : int
        Number of contour levels (ignored if levels is provided).
    levels : np.ndarray, optional
        Explicit contour level values.
    surface_cmap : str
        Colormap for surface coloring.
    surface_opacity : float
        Surface opacity (0-1).
    contour_color : str
        Color of contour lines (single color) or "scalar" to color by value.
    contour_width : float
        Width of contour lines.
    show_surface : bool
        Whether to show the surface.
    clim : tuple, optional
        (min, max) for color scaling.
    show_scalar_bar : bool
        Whether to show the scalar bar.
    title : str, optional
        Plot title.
    window_size : tuple
        Window size (width, height).
    background : str
        Background color.
    plotter : pv.Plotter, optional
        Existing plotter to add to.
    show : bool
        Whether to display the plot.
    screenshot : str, optional
        Path to save screenshot.

    Returns
    -------
    pv.Plotter
        The plotter object.
    """
    _check_pyvista()

    mesh = create_pyvista_mesh(vertices, triangles)
    scalar_field = np.asarray(scalar_field)
    mesh["scalar"] = scalar_field

    vmin = scalar_field.min() if clim is None else clim[0]
    vmax = scalar_field.max() if clim is None else clim[1]

    if levels is None:
        levels = np.linspace(vmin, vmax, num_levels)

    if plotter is None:
        off_screen = screenshot is not None and not show
        plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
        plotter.set_background(background)

    # Add surface
    if show_surface:
        plotter.add_mesh(
            mesh,
            scalars="scalar",
            cmap=surface_cmap,
            opacity=surface_opacity,
            clim=(vmin, vmax),
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": "scalar"}
        )

    # Add iso-contours
    contours = mesh.contour(isosurfaces=levels, scalars="scalar")

    if contour_color == "scalar":
        plotter.add_mesh(
            contours,
            scalars="scalar",
            cmap=surface_cmap,
            line_width=contour_width,
            clim=(vmin, vmax),
            show_scalar_bar=False
        )
    else:
        plotter.add_mesh(
            contours,
            color=contour_color,
            line_width=contour_width
        )

    if title:
        plotter.add_title(title)

    if screenshot and not show:
        plotter.show(screenshot=screenshot)
    elif screenshot and show:
        plotter.show(screenshot=screenshot)
    elif show:
        plotter.show()

    return plotter


def plot_surface_with_colored_contours(
    vertices: np.ndarray,
    triangles: np.ndarray,
    scalar_field: np.ndarray,
    num_levels: int = 10,
    levels: np.ndarray = None,
    cmap: str = "viridis",
    contour_width: float = 4.0,
    surface_opacity: float = 0.3,
    show_surface: bool = True,
    clim: Tuple[float, float] = None,
    show_scalar_bar: bool = True,
    title: str = None,
    window_size: Tuple[int, int] = (1024, 768),
    background: str = "white",
    plotter: "pv.Plotter" = None,
    show: bool = True,
    screenshot: str = None
) -> "pv.Plotter":
    """
    Plot a 3D surface with iso-contours colored by their scalar value.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : np.ndarray
        Array of shape (n_triangles, 3) containing vertex indices.
    scalar_field : np.ndarray
        Array of size n_vertices containing scalar values.
    num_levels : int
        Number of contour levels.
    levels : np.ndarray, optional
        Explicit contour level values.
    cmap : str
        Colormap for both surface and contours.
    contour_width : float
        Width of contour lines.
    surface_opacity : float
        Surface opacity.
    show_surface : bool
        Whether to show the underlying surface.
    clim : tuple, optional
        (min, max) for color scaling.
    show_scalar_bar : bool
        Whether to show the scalar bar.
    title : str, optional
        Plot title.
    window_size : tuple
        Window size.
    background : str
        Background color.
    plotter : pv.Plotter, optional
        Existing plotter.
    show : bool
        Whether to display.
    screenshot : str, optional
        Path to save screenshot.

    Returns
    -------
    pv.Plotter
        The plotter object.
    """
    return plot_surface_with_isocontours(
        vertices=vertices,
        triangles=triangles,
        scalar_field=scalar_field,
        num_levels=num_levels,
        levels=levels,
        surface_cmap=cmap,
        surface_opacity=surface_opacity,
        contour_color="scalar",  # Color contours by scalar value
        contour_width=contour_width,
        show_surface=show_surface,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        title=title,
        window_size=window_size,
        background=background,
        plotter=plotter,
        show=show,
        screenshot=screenshot
    )


def plot_multiple_surfaces(
    meshes: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    cmaps: List[str] = None,
    opacities: List[float] = None,
    show_edges: bool = False,
    window_size: Tuple[int, int] = (1024, 768),
    background: str = "white",
    show: bool = True,
    screenshot: str = None
) -> "pv.Plotter":
    """
    Plot multiple surfaces in a single view or side-by-side.

    Parameters
    ----------
    meshes : list of tuples
        Each tuple is (vertices, triangles, scalar_field).
        scalar_field can be None.
    cmaps : list of str, optional
        Colormaps for each surface.
    opacities : list of float, optional
        Opacities for each surface.
    show_edges : bool
        Whether to show mesh edges.
    window_size : tuple
        Window size.
    background : str
        Background color.
    show : bool
        Whether to display.
    screenshot : str, optional
        Path to save screenshot.

    Returns
    -------
    pv.Plotter
        The plotter object.
    """
    _check_pyvista()

    n_meshes = len(meshes)

    if cmaps is None:
        cmaps = ["viridis"] * n_meshes
    if opacities is None:
        opacities = [1.0] * n_meshes

    # Single view with all meshes
    off_screen = screenshot is not None and not show
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
    plotter.set_background(background)

    for i, (verts, tris, scalars) in enumerate(meshes):
        mesh = create_pyvista_mesh(verts, tris)

        if scalars is not None:
            mesh[f"scalar_{i}"] = np.asarray(scalars)
            plotter.add_mesh(
                mesh,
                scalars=f"scalar_{i}",
                cmap=cmaps[i],
                opacity=opacities[i],
                show_edges=show_edges,
                show_scalar_bar=True
            )
        else:
            plotter.add_mesh(
                mesh,
                opacity=opacities[i],
                show_edges=show_edges
            )

    if screenshot and not show:
        plotter.show(screenshot=screenshot)
    elif screenshot and show:
        plotter.show(screenshot=screenshot)
    elif show:
        plotter.show()

    return plotter


def plot_surface_vectors(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vectors: np.ndarray,
    scalar_field: np.ndarray = None,
    vector_scale: float = 1.0,
    vector_color: str = "red",
    surface_cmap: str = "viridis",
    surface_opacity: float = 0.7,
    show_surface: bool = True,
    title: str = None,
    window_size: Tuple[int, int] = (1024, 768),
    background: str = "white",
    plotter: "pv.Plotter" = None,
    show: bool = True,
    screenshot: str = None
) -> "pv.Plotter":
    """
    Plot a surface with vector arrows at each vertex.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n_vertices, 3).
    triangles : np.ndarray
        Array of shape (n_triangles, 3).
    vectors : np.ndarray
        Array of shape (n_vertices, 3) containing vector components.
    scalar_field : np.ndarray, optional
        Scalar field for surface coloring.
    vector_scale : float
        Scaling factor for vector arrows.
    vector_color : str
        Color for vector arrows.
    surface_cmap : str
        Colormap for surface.
    surface_opacity : float
        Surface opacity.
    show_surface : bool
        Whether to show the surface.
    title : str, optional
        Plot title.
    window_size : tuple
        Window size.
    background : str
        Background color.
    plotter : pv.Plotter, optional
        Existing plotter.
    show : bool
        Whether to display.
    screenshot : str, optional
        Path to save screenshot.

    Returns
    -------
    pv.Plotter
        The plotter object.
    """
    _check_pyvista()

    mesh = create_pyvista_mesh(vertices, triangles)
    vectors = np.asarray(vectors)
    mesh["vectors"] = vectors

    if scalar_field is not None:
        mesh["scalar"] = np.asarray(scalar_field)

    if plotter is None:
        off_screen = screenshot is not None and not show
        plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
        plotter.set_background(background)

    # Add surface
    if show_surface:
        if scalar_field is not None:
            plotter.add_mesh(
                mesh,
                scalars="scalar",
                cmap=surface_cmap,
                opacity=surface_opacity,
                show_scalar_bar=True
            )
        else:
            plotter.add_mesh(
                mesh,
                opacity=surface_opacity
            )

    # Add vectors as arrows
    arrows = mesh.glyph(
        orient="vectors",
        scale="vectors",
        factor=vector_scale
    )
    plotter.add_mesh(arrows, color=vector_color)

    if title:
        plotter.add_title(title)

    if screenshot and not show:
        plotter.show(screenshot=screenshot)
    elif screenshot and show:
        plotter.show(screenshot=screenshot)
    elif show:
        plotter.show()

    return plotter
