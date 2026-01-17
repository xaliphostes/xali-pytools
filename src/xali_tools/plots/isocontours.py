import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 2D plot iso-contours of scalars on a regular grid of size n x m
# from a flattened scalar field array of size n * m
# ---------------------------------------------------------------
def plotIsoContoursFromFlatArray(
    scalarField: np.ndarray,
    n: int,
    m: int,
    x_range: tuple = (0, 1),
    y_range: tuple = (0, 1),
    num_levels: int = 20,
    show_contour_lines: bool = True,
    cmap: str = "jet",
    title: str = "Iso-contours",
    figsize: tuple = (8, 6)
):
    """
    Plot iso-contours of a scalar field on a regular grid.

    Parameters:
    -----------
    scalarField : np.ndarray
        Flattened array of size n*m containing scalar values.
    n : int
        Number of grid points in x direction (columns).
    m : int
        Number of grid points in y direction (rows).
    x_range : tuple
        (xmin, xmax) for the grid.
    y_range : tuple
        (ymin, ymax) for the grid.
    num_levels : int
        Number of contour levels.
    show_contour_lines : bool
        If True, overlay contour lines on the filled contours.
    cmap : str
        Colormap name.
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height).
    """
    # Reshape the flattened array to grid (m rows, n columns)
    scalars = np.array(scalarField).reshape(m, n)

    vmin = np.min(scalars)
    vmax = np.max(scalars)
    print(f'min, max = {vmin}, {vmax}')

    if vmin == vmax:
        print("Warning: constant field, nothing to plot")
        return None, None

    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], m)
    X, Y = np.meshgrid(x, y)

    # Create contour levels
    levels = np.linspace(vmin, vmax, num_levels)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Filled contours
    CS = ax.contourf(X, Y, scalars, levels=levels, cmap=cmap)

    # Overlay contour lines
    if show_contour_lines:
        CS2 = ax.contour(CS, levels=levels[::3], colors='black', linewidths=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    # Colorbar
    cbar = fig.colorbar(CS)
    if show_contour_lines:
        cbar.add_lines(CS2)

    plt.tight_layout()
    plt.show()

    return fig, ax

