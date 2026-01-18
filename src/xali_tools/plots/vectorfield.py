import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 2D plot vector field (quiver) on a regular grid of size n x m
# from a flattened vector field array of size 3 * n * m
# ---------------------------------------------------
def plotVectorFieldFromFlatArray(
    vectorField: np.ndarray,
    n: int,
    m: int,
    x_range: tuple = (0, 1),
    y_range: tuple = (0, 1),
    skip: int = 1,
    scale: float = None,
    color_by_magnitude: bool = True,
    normalize_arrows: bool = False,
    title: str = "Vector Field",
    figsize: tuple = (8, 6)
):
    """
    Plot a vector field (quiver plot) on a regular grid.

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
    skip : int
        Plot every nth arrow (reduces clutter for dense grids).
    scale : float
        Arrow scale factor. None for auto-scaling.
    color_by_magnitude : bool
        If True, color arrows by vector magnitude.
    normalize_arrows : bool
        If True, normalize arrows to unit length (shows direction only).
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

    # Apply skip for subsampling
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]

    # Compute magnitude for coloring
    speed = np.sqrt(u_sub**2 + v_sub**2)

    # Normalize arrows if requested
    if normalize_arrows:
        magnitude = np.sqrt(u_sub**2 + v_sub**2)
        magnitude[magnitude == 0] = 1  # Avoid division by zero
        u_sub = u_sub / magnitude
        v_sub = v_sub / magnitude

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    if color_by_magnitude:
        quiv = ax.quiver(X_sub, Y_sub, u_sub, v_sub, speed,
                         cmap='viridis', scale=scale)
        cbar = fig.colorbar(quiv)
        cbar.set_label('Magnitude')
    else:
        ax.quiver(X_sub, Y_sub, u_sub, v_sub, scale=scale)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return fig, ax
