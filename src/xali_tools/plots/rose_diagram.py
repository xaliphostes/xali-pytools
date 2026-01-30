import numpy as np
import matplotlib.pyplot as plt


def plotRoseDiagram(
    azimuths: np.ndarray,
    weights: np.ndarray = None,
    num_bins: int = 36,
    bidirectional: bool = True,
    opening_angle: float = 1.0,
    cmap: str = "viridis",
    color: str = None,
    edgecolor: str = "black",
    title: str = "Rose Diagram",
    figsize: tuple = (8, 8),
    show_statistics: bool = True,
):
    """
    Plot a rose diagram (circular histogram) showing orientation distribution.

    Rose diagrams are commonly used in geology to display the orientation
    (direction) of geological features such as fractures, joints, faults,
    or paleocurrent directions.

    Parameters:
    -----------
    azimuths : np.ndarray
        Array of azimuth angles in degrees (0-360, measured clockwise from North).
    weights : np.ndarray, optional
        Weights for each azimuth (e.g., fracture length). If None, all weights = 1.
    num_bins : int
        Number of bins around the circle. Default is 36 (10-degree bins).
    bidirectional : bool
        If True, treat data as bidirectional (e.g., fractures have no "head").
        This folds 180-360 degrees onto 0-180 and mirrors the result.
    opening_angle : float
        Fraction of bin width for bar width (0-1). 1.0 = bars touch, 0.5 = half-width.
    cmap : str
        Colormap name for coloring bars by count/weight. Ignored if color is set.
    color : str, optional
        Single color for all bars (e.g., 'steelblue'). Overrides cmap.
    edgecolor : str
        Edge color of the bars.
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height).
    show_statistics : bool
        If True, display mean direction and resultant length.

    Returns:
    --------
    tuple
        (fig, ax) - Matplotlib figure and polar axes for further customization.

    Examples:
    ---------
    >>> import numpy as np
    >>> from xali_tools.plots import plotRoseDiagram
    >>> # Simulated fracture orientations (mostly NE-SW trending)
    >>> azimuths = np.random.vonmises(np.radians(45), 2, 200)
    >>> azimuths = np.degrees(azimuths) % 360
    >>> plotRoseDiagram(azimuths, bidirectional=True, title="Fracture Orientations")
    """
    azimuths = np.asarray(azimuths).flatten()

    if weights is not None:
        weights = np.asarray(weights).flatten()
        if len(weights) != len(azimuths):
            raise ValueError("weights must have the same length as azimuths")
    else:
        weights = np.ones_like(azimuths)

    # Normalize azimuths to 0-360
    azimuths = azimuths % 360

    if bidirectional:
        # Fold data: convert 180-360 to 0-180
        azimuths = np.where(azimuths >= 180, azimuths - 180, azimuths)
        bin_range = (0, 180)
        effective_bins = num_bins // 2
    else:
        bin_range = (0, 360)
        effective_bins = num_bins

    # Compute histogram
    bin_edges = np.linspace(bin_range[0], bin_range[1], effective_bins + 1)
    counts, _ = np.histogram(azimuths, bins=bin_edges, weights=weights)

    # Compute bin centers and width
    bin_width_deg = (bin_range[1] - bin_range[0]) / effective_bins
    bin_centers_deg = bin_edges[:-1] + bin_width_deg / 2

    # Convert to radians (matplotlib polar uses radians, 0 = East, counter-clockwise)
    # We want 0 = North, clockwise, so we transform
    bin_centers_rad = np.radians(90 - bin_centers_deg)
    bar_width_rad = np.radians(bin_width_deg) * opening_angle

    # For bidirectional data, mirror to the opposite side
    if bidirectional:
        bin_centers_rad = np.concatenate([bin_centers_rad, bin_centers_rad + np.pi])
        counts = np.concatenate([counts, counts])
        bar_width_rad_array = np.full(len(counts), bar_width_rad)
    else:
        bar_width_rad_array = np.full(len(counts), bar_width_rad)

    # Create polar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    # Set North at top, clockwise direction
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Determine colors
    if color is not None:
        bar_colors = color
    else:
        norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
        cmap_obj = plt.cm.get_cmap(cmap)
        bar_colors = cmap_obj(norm(counts))

    # Plot bars
    bars = ax.bar(
        bin_centers_rad,
        counts,
        width=bar_width_rad_array,
        bottom=0,
        color=bar_colors,
        edgecolor=edgecolor,
        linewidth=0.5,
        alpha=0.8,
    )

    # Add colorbar if using colormap
    if color is None and counts.max() > counts.min():
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("Count" if weights is None else "Weighted Count")

    # Compute and display statistics
    if show_statistics:
        # Use original azimuths length for weights (before mirroring for display)
        original_len = len(azimuths)
        stats = compute_circular_statistics(
            azimuths, weights[:original_len], bidirectional
        )
        stats_text = f"n = {stats['n']}\nMean: {stats['mean_direction']:.1f}°\nR = {stats['resultant_length']:.2f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_title(title, pad=20)

    plt.tight_layout()
    plt.show()

    return fig, ax


def plotRoseDiagramFromVectors(
    vectors: np.ndarray,
    weights: np.ndarray = None,
    num_bins: int = 36,
    bidirectional: bool = True,
    **kwargs,
):
    """
    Plot a rose diagram from 3D direction vectors.

    Computes the horizontal azimuth (bearing) from 3D vectors and creates
    a rose diagram. Useful for visualizing fault slip directions, stress
    orientations, or any 3D directional data projected onto the horizontal plane.

    Parameters:
    -----------
    vectors : np.ndarray
        Array of 3D vectors, either shape (n, 3) or flattened (3*n,).
        Vectors are assumed to be [x, y, z] where x=East, y=North, z=Up.
    weights : np.ndarray, optional
        Weights for each vector (e.g., magnitude, length).
    num_bins : int
        Number of bins around the circle.
    bidirectional : bool
        If True, treat directions as bidirectional.
    **kwargs
        Additional arguments passed to plotRoseDiagram.

    Returns:
    --------
    tuple
        (fig, ax) - Matplotlib figure and polar axes.

    Examples:
    ---------
    >>> import numpy as np
    >>> from xali_tools.plots import plotRoseDiagramFromVectors
    >>> # Random 3D vectors
    >>> vectors = np.random.randn(100, 3)
    >>> plotRoseDiagramFromVectors(vectors, title="Vector Orientations")
    """
    vectors = np.asarray(vectors).reshape(-1, 3)

    # Compute azimuth from x (East) and y (North) components
    # Azimuth = angle from North, measured clockwise
    x = vectors[:, 0]  # East component
    y = vectors[:, 1]  # North component

    # atan2(x, y) gives angle from North, but we need to handle the convention
    azimuths = np.degrees(np.arctan2(x, y)) % 360

    return plotRoseDiagram(
        azimuths,
        weights=weights,
        num_bins=num_bins,
        bidirectional=bidirectional,
        **kwargs,
    )


def compute_circular_statistics(azimuths: np.ndarray, weights: np.ndarray = None, bidirectional: bool = True):
    """
    Compute circular statistics for azimuth data.

    Parameters:
    -----------
    azimuths : np.ndarray
        Azimuth angles in degrees.
    weights : np.ndarray, optional
        Weights for each measurement.
    bidirectional : bool
        If True, double the angles before computing statistics (axial data).

    Returns:
    --------
    dict
        Dictionary containing:
        - n: number of measurements
        - mean_direction: mean direction in degrees
        - resultant_length: R value (0-1, measure of concentration)
        - circular_variance: 1 - R
    """
    azimuths = np.asarray(azimuths).flatten()
    n = len(azimuths)

    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights).flatten()

    # For bidirectional data, double the angles
    if bidirectional:
        theta = np.radians(2 * azimuths)
    else:
        theta = np.radians(azimuths)

    # Compute weighted mean direction
    sum_weights = np.sum(weights)
    C = np.sum(weights * np.cos(theta)) / sum_weights
    S = np.sum(weights * np.sin(theta)) / sum_weights

    # Resultant length (R)
    R = np.sqrt(C**2 + S**2)

    # Mean direction
    mean_theta = np.arctan2(S, C)
    if bidirectional:
        mean_direction = (np.degrees(mean_theta) / 2) % 180
    else:
        mean_direction = np.degrees(mean_theta) % 360

    return {
        "n": n,
        "mean_direction": mean_direction,
        "resultant_length": R,
        "circular_variance": 1 - R,
    }
