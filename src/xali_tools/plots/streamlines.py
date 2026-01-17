import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 2D plot streamlines on a regular grid of size n x m
# from a flattened vector field array of size 3 * n * m
# ---------------------------------------------------
def plotStreamlinesFromFlatArray(
    vectorField: np.ndarray,
    n: int,
    m: int,
    x_range: tuple = (0, 1),
    y_range: tuple = (0, 1),
    density: float = 1.5,
    color_by_magnitude: bool = True,
    title: str = "Streamlines",
    figsize: tuple = (8, 6)
):
    """
    Plot streamlines of a vector field on a regular grid.

    Parameters:
    -----------
    vectorField : np.ndarray
        Flattened array of size 3*n*m containing (vx, vy, vz) components.
        The ordering is assumed to be [vx0, vy0, vz0, vx1, vy1, vz1, ...].
    n : int
        Number of grid points in x direction (columns).
    m : int
        Number of grid points in y direction (rows).
    x_range : tuple
        (xmin, xmax) for the grid.
    y_range : tuple
        (ymin, ymax) for the grid.
    density : float
        Density of streamlines (higher = more lines).
    color_by_magnitude : bool
        If True, color streamlines by vector magnitude.
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height).
    """
    # Reshape the flattened array to (n*m, 3)
    vectors = np.array(vectorField).reshape(-1, 3)

    # Extract u (x-component) and v (y-component), ignoring z
    u_flat = vectors[:, 0]
    v_flat = vectors[:, 1]

    # Reshape to grid (m rows, n columns)
    u = u_flat.reshape(m, n)
    v = v_flat.reshape(m, n)

    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], m)
    X, Y = np.meshgrid(x, y)

    # Compute magnitude for coloring
    speed = np.sqrt(u**2 + v**2)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    if color_by_magnitude:
        strm = ax.streamplot(X, Y, u, v, density=density, color=speed, cmap='viridis')
        cbar = fig.colorbar(strm.lines)
        cbar.set_label('Magnitude')
    else:
        ax.streamplot(X, Y, u, v, density=density)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return fig, ax




# min = -2
# max = 2
# n = 51
# xx, yy = np.meshgrid(np.linspace(min, max, n), np.linspace(min, max, n))
# coords = np.array((xx.flatten(), yy.flatten(), np.linspace(1, 1, n*n).flatten())).T.flatten()
# print( solution.stress(coords) )

# plotIso(postprocess)
# plotStream(postprocess)
