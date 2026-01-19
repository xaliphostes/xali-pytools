# xali_tools.plots subpackage

from .isocontours import plotIsoContoursFromFlatArray
from .streamlines import plotStreamlinesFromFlatArray
from .vectorfield import plotVectorFieldFromFlatArray
from .surface3d import (
    create_pyvista_mesh,
    plot_triangulated_surface,
    plot_surface_with_isocontours,
    plot_surface_with_colored_contours,
    plot_multiple_surfaces,
    plot_surface_vectors,
)

__all__ = [
    "plotIsoContoursFromFlatArray",
    "plotStreamlinesFromFlatArray",
    "plotVectorFieldFromFlatArray",
    # PyVista-based 3D surface plotting
    "create_pyvista_mesh",
    "plot_triangulated_surface",
    "plot_surface_with_isocontours",
    "plot_surface_with_colored_contours",
    "plot_multiple_surfaces",
    "plot_surface_vectors",
]
