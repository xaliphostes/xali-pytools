"""
Wulff Stereonet (Stereographic Projection) for structural geology.

A stereographic projection used to analyze the orientation of planes and lines
in 3D space. This is an equal-angle (conformal) projection that preserves
angular relationships.

Convention:
- Lower hemisphere projection (standard in structural geology)
- North at top, East to the right
- Planes can be specified as strike/dip (right-hand rule) or dip_direction/dip
- Lines specified as trend/plunge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


def _stereonet_coords(trend, plunge):
    """
    Convert trend/plunge to stereonet x,y coordinates (Wulff projection).

    Parameters:
    -----------
    trend : float or np.ndarray
        Trend (azimuth) in degrees, measured clockwise from North.
    plunge : float or np.ndarray
        Plunge in degrees, measured downward from horizontal (0-90).

    Returns:
    --------
    tuple
        (x, y) coordinates on the stereonet.
    """
    trend_rad = np.radians(trend)
    plunge_rad = np.radians(plunge)

    # Wulff (equal-angle) projection
    # r = tan((90 - plunge) / 2) for lower hemisphere
    r = np.tan(np.radians(45) - plunge_rad / 2)

    # Convert to Cartesian (North up, East right)
    x = r * np.sin(trend_rad)
    y = r * np.cos(trend_rad)

    return x, y


def _plane_to_pole(strike, dip):
    """
    Convert strike/dip of a plane to trend/plunge of its pole.

    Uses right-hand rule: strike direction with dip to the right.

    Parameters:
    -----------
    strike : float
        Strike azimuth in degrees (right-hand rule).
    dip : float
        Dip angle in degrees (0-90).

    Returns:
    --------
    tuple
        (trend, plunge) of the pole to the plane.
    """
    # Pole trend is 90 degrees clockwise from strike (dip direction)
    pole_trend = (strike + 90) % 360
    # Pole plunge is 90 - dip
    pole_plunge = 90 - dip

    return pole_trend, pole_plunge


def _great_circle_points(strike, dip, n_points=181):
    """
    Generate points along a great circle for a plane.

    The great circle extends from one edge of the stereonet to the other,
    intersecting the primitive circle at azimuths equal to the strike
    and strike + 180 degrees.

    Parameters:
    -----------
    strike : float
        Strike azimuth in degrees (right-hand rule).
    dip : float
        Dip angle in degrees.
    n_points : int
        Number of points along the great circle.

    Returns:
    --------
    tuple
        (x, y) arrays of stereonet coordinates.
    """
    # The great circle intersects the primitive at strike and strike+180
    # We trace along azimuths from strike to strike+180
    # At each azimuth A, the plunge is: arctan(tan(dip) * sin(A - strike))

    # Generate azimuths from strike to strike + 180
    azimuths = np.linspace(strike, strike + 180, n_points)

    # Compute plunge at each azimuth using apparent dip formula
    dip_rad = np.radians(dip)
    angle_from_strike = np.radians(azimuths - strike)

    # Plunge = arctan(tan(dip) * sin(angle_from_strike))
    plunges = np.degrees(np.arctan(np.tan(dip_rad) * np.sin(angle_from_strike)))

    # Ensure plunges are positive (lower hemisphere)
    plunges = np.abs(plunges)

    # Normalize azimuths to 0-360
    trends = azimuths % 360

    return _stereonet_coords(trends, plunges)


def _draw_stereonet_grid(ax, n_circles=9, n_radials=36):
    """
    Draw the stereonet grid (small circles and great circles).

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    n_circles : int
        Number of small circles (dip intervals).
    n_radials : int
        Number of radial lines (strike intervals).
    """
    # Draw the outer circle (primitive)
    circle = Circle((0, 0), 1, fill=False, color="black", linewidth=1.5)
    ax.add_patch(circle)

    # Draw small circles (lines of equal plunge/dip)
    plunge_intervals = np.linspace(0, 90, n_circles + 1)[1:-1]
    theta = np.linspace(0, 2 * np.pi, 361)

    for plunge in plunge_intervals:
        r = np.tan(np.radians(45) - np.radians(plunge) / 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, "gray", linewidth=0.3, alpha=0.5)

    # Draw great circles (lines of equal trend)
    trend_intervals = np.linspace(0, 180, n_radials // 2 + 1)[:-1]

    for trend in trend_intervals:
        # Each great circle is a vertical plane with given strike
        x, y = _great_circle_points(trend, 90, n_points=181)
        ax.plot(x, y, "gray", linewidth=0.3, alpha=0.5)

    # Draw cardinal directions
    for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        rad = np.radians(angle)
        ax.plot([0, np.sin(rad)], [0, np.cos(rad)], "gray", linewidth=0.5, alpha=0.7)
        offset = 1.08
        ax.text(
            offset * np.sin(rad),
            offset * np.cos(rad),
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )


class Stereonet:
    """
    A Wulff stereonet for plotting and analyzing structural geology data.

    This class provides methods to plot:
    - Planes as great circles or poles
    - Lines/lineations as points
    - Density contours of orientations

    Examples:
    ---------
    >>> from xali_tools.plots import Stereonet
    >>> sn = Stereonet(title="Fault Orientations")
    >>> sn.plane(strike=45, dip=60, color='blue')  # Plot as great circle
    >>> sn.pole(strike=120, dip=30, color='red')   # Plot pole to plane
    >>> sn.line(trend=270, plunge=45, marker='s')  # Plot lineation
    >>> sn.show()
    """

    def __init__(self, figsize=(8, 8), title="Stereonet", show_grid=True):
        """
        Initialize a new stereonet.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height).
        title : str
            Plot title.
        show_grid : bool
            If True, draw the stereonet grid.
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axis("off")
        self.ax.set_title(title, pad=20, fontsize=12)

        if show_grid:
            _draw_stereonet_grid(self.ax)

        # Storage for legend
        self._legend_handles = []
        self._legend_labels = []

    def plane(
        self,
        strike=None,
        dip=None,
        dip_direction=None,
        color="blue",
        linewidth=1.5,
        linestyle="-",
        alpha=1.0,
        label=None,
    ):
        """
        Plot a plane as a great circle.

        Parameters:
        -----------
        strike : float or array-like
            Strike azimuth in degrees (right-hand rule). Required if dip_direction not given.
        dip : float or array-like
            Dip angle in degrees (0-90).
        dip_direction : float or array-like, optional
            Dip direction in degrees. If given, strike is computed as dip_direction - 90.
        color : str
            Line color.
        linewidth : float
            Line width.
        linestyle : str
            Line style ('-', '--', ':', etc.).
        alpha : float
            Transparency (0-1).
        label : str, optional
            Label for legend.

        Returns:
        --------
        self
            For method chaining.
        """
        if dip_direction is not None:
            strike = (np.asarray(dip_direction) - 90) % 360

        strike = np.atleast_1d(strike)
        dip = np.atleast_1d(dip)

        if len(strike) == 1 and len(dip) > 1:
            strike = np.full_like(dip, strike[0])
        elif len(dip) == 1 and len(strike) > 1:
            dip = np.full_like(strike, dip[0])

        for s, d in zip(strike, dip):
            x, y = _great_circle_points(s, d)
            line, = self.ax.plot(x, y, color=color, linewidth=linewidth,
                                  linestyle=linestyle, alpha=alpha)

        if label:
            self._legend_handles.append(line)
            self._legend_labels.append(label)

        return self

    def pole(
        self,
        strike=None,
        dip=None,
        dip_direction=None,
        color="red",
        marker="o",
        markersize=6,
        alpha=1.0,
        label=None,
    ):
        """
        Plot pole(s) to plane(s).

        Parameters:
        -----------
        strike : float or array-like
            Strike azimuth in degrees (right-hand rule).
        dip : float or array-like
            Dip angle in degrees.
        dip_direction : float or array-like, optional
            Dip direction. If given, strike = dip_direction - 90.
        color : str
            Marker color.
        marker : str
            Marker style ('o', 's', '^', etc.).
        markersize : float
            Marker size.
        alpha : float
            Transparency.
        label : str, optional
            Label for legend.

        Returns:
        --------
        self
            For method chaining.
        """
        if dip_direction is not None:
            strike = (np.asarray(dip_direction) - 90) % 360

        strike = np.atleast_1d(strike)
        dip = np.atleast_1d(dip)

        if len(strike) == 1 and len(dip) > 1:
            strike = np.full_like(dip, strike[0])
        elif len(dip) == 1 and len(strike) > 1:
            dip = np.full_like(strike, dip[0])

        trends = []
        plunges = []
        for s, d in zip(strike, dip):
            t, p = _plane_to_pole(s, d)
            trends.append(t)
            plunges.append(p)

        trends = np.array(trends)
        plunges = np.array(plunges)
        x, y = _stereonet_coords(trends, plunges)

        scatter = self.ax.scatter(x, y, c=color, marker=marker, s=markersize**2,
                                   alpha=alpha, zorder=5)

        if label:
            self._legend_handles.append(scatter)
            self._legend_labels.append(label)

        return self

    def line(
        self,
        trend,
        plunge,
        color="green",
        marker="o",
        markersize=6,
        alpha=1.0,
        label=None,
    ):
        """
        Plot line(s) / lineation(s).

        Parameters:
        -----------
        trend : float or array-like
            Trend (azimuth) in degrees.
        plunge : float or array-like
            Plunge in degrees (0-90, downward from horizontal).
        color : str
            Marker color.
        marker : str
            Marker style.
        markersize : float
            Marker size.
        alpha : float
            Transparency.
        label : str, optional
            Label for legend.

        Returns:
        --------
        self
            For method chaining.
        """
        trend = np.atleast_1d(trend)
        plunge = np.atleast_1d(plunge)

        x, y = _stereonet_coords(trend, plunge)
        scatter = self.ax.scatter(x, y, c=color, marker=marker, s=markersize**2,
                                   alpha=alpha, zorder=5)

        if label:
            self._legend_handles.append(scatter)
            self._legend_labels.append(label)

        return self

    def vector(
        self,
        vectors,
        color="blue",
        marker="o",
        markersize=6,
        alpha=1.0,
        label=None,
    ):
        """
        Plot 3D vectors as points on the stereonet.

        Vectors pointing upward (positive z) are projected to the lower hemisphere
        by reversing their direction.

        Parameters:
        -----------
        vectors : np.ndarray
            Array of shape (n, 3) with [x, y, z] components (x=East, y=North, z=Up).
        color : str
            Marker color.
        marker : str
            Marker style.
        markersize : float
            Marker size.
        alpha : float
            Transparency.
        label : str, optional
            Label for legend.

        Returns:
        --------
        self
            For method chaining.
        """
        vectors = np.atleast_2d(vectors)
        if vectors.shape[1] != 3:
            raise ValueError("vectors must have shape (n, 3)")

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        # Project upward-pointing vectors to lower hemisphere
        vectors = np.where(vectors[:, 2:3] > 0, -vectors, vectors)

        # Convert to trend/plunge
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

        # Plunge (downward from horizontal)
        plunge = np.degrees(np.arcsin(-z))

        # Trend (azimuth from North)
        trend = np.degrees(np.arctan2(x, y)) % 360

        return self.line(trend, plunge, color=color, marker=marker,
                        markersize=markersize, alpha=alpha, label=label)

    def density(
        self,
        strike=None,
        dip=None,
        trend=None,
        plunge=None,
        data_type="poles",
        gridsize=100,
        cmap="Reds",
        levels=10,
        alpha=0.7,
        colorbar=True,
    ):
        """
        Plot density contours of orientation data.

        Parameters:
        -----------
        strike, dip : array-like, optional
            Plane orientations (for poles or planes).
        trend, plunge : array-like, optional
            Line orientations (for lines).
        data_type : str
            'poles' to contour poles to planes, 'lines' for lineations.
        gridsize : int
            Resolution of the density grid.
        cmap : str
            Colormap for contours.
        levels : int
            Number of contour levels.
        alpha : float
            Transparency of contours.
        colorbar : bool
            Whether to show a colorbar.

        Returns:
        --------
        self
            For method chaining.
        """
        if data_type == "poles" and strike is not None and dip is not None:
            strike = np.atleast_1d(strike)
            dip = np.atleast_1d(dip)
            trends, plunges = [], []
            for s, d in zip(strike, dip):
                t, p = _plane_to_pole(s, d)
                trends.append(t)
                plunges.append(p)
            trend = np.array(trends)
            plunge = np.array(plunges)
        elif trend is not None and plunge is not None:
            trend = np.atleast_1d(trend)
            plunge = np.atleast_1d(plunge)
        else:
            raise ValueError("Must provide either (strike, dip) or (trend, plunge)")

        x, y = _stereonet_coords(trend, plunge)

        # Create a grid
        xi = np.linspace(-1, 1, gridsize)
        yi = np.linspace(-1, 1, gridsize)
        Xi, Yi = np.meshgrid(xi, yi)

        # Count points in each grid cell
        density = np.zeros((gridsize, gridsize))
        for px, py in zip(x, y):
            # Find nearest grid point
            ix = int((px + 1) / 2 * (gridsize - 1))
            iy = int((py + 1) / 2 * (gridsize - 1))
            ix = np.clip(ix, 0, gridsize - 1)
            iy = np.clip(iy, 0, gridsize - 1)
            density[iy, ix] += 1

        # Apply Gaussian smoothing
        sigma = gridsize / 20
        #try:
        #    from scipy import ndimage
        #    density = ndimage.gaussian_filter(density, sigma=sigma)
        #except ImportError:
        #
        # Fallback: simple box filter using numpy
        kernel_size = int(sigma * 2) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        from numpy.lib.stride_tricks import as_strided
        # Pad and convolve manually
        pad = kernel_size // 2
        padded = np.pad(density, pad, mode='constant')
        # Simple convolution
        result = np.zeros_like(density)
        for i in range(density.shape[0]):
            for j in range(density.shape[1]):
                result[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
        density = result

        # Mask outside the stereonet circle
        mask = Xi**2 + Yi**2 > 1
        density = np.ma.masked_where(mask, density)

        # Plot contours
        contour = self.ax.contourf(Xi, Yi, density, levels=levels, cmap=cmap,
                                    alpha=alpha, zorder=1)

        if colorbar:
            cbar = self.fig.colorbar(contour, ax=self.ax, shrink=0.7, pad=0.05)
            cbar.set_label("Density")

        return self

    def legend(self, **kwargs):
        """
        Show legend.

        Parameters:
        -----------
        **kwargs
            Arguments passed to ax.legend().

        Returns:
        --------
        self
            For method chaining.
        """
        if self._legend_handles:
            kwargs.setdefault("loc", "upper left")
            kwargs.setdefault("bbox_to_anchor", (1.05, 1))
            self.ax.legend(self._legend_handles, self._legend_labels, **kwargs)
        return self

    def show(self):
        """Display the stereonet."""
        plt.tight_layout()
        plt.show()
        return self.fig, self.ax

    def savefig(self, filename, **kwargs):
        """
        Save the stereonet to a file.

        Parameters:
        -----------
        filename : str
            Output filename.
        **kwargs
            Arguments passed to fig.savefig().
        """
        kwargs.setdefault("dpi", 150)
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(filename, **kwargs)
        return self


def plotStereonet(
    planes=None,
    poles=None,
    lines=None,
    title="Stereonet",
    figsize=(8, 8),
    show_grid=True,
):
    """
    Quick function to plot a stereonet with multiple datasets.

    Parameters:
    -----------
    planes : list of dict, optional
        List of plane datasets. Each dict should have 'strike', 'dip' (or 'dip_direction', 'dip')
        and optional 'color', 'label'.
    poles : list of dict, optional
        List of pole datasets. Same format as planes.
    lines : list of dict, optional
        List of line datasets. Each dict should have 'trend', 'plunge'
        and optional 'color', 'marker', 'label'.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    show_grid : bool
        Whether to show the stereonet grid.

    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects.

    Examples:
    ---------
    >>> from xali_tools.plots import plotStereonet
    >>> plotStereonet(
    ...     planes=[{'strike': [30, 45, 60], 'dip': [40, 50, 60], 'color': 'blue', 'label': 'Set 1'}],
    ...     poles=[{'strike': [120, 130], 'dip': [70, 75], 'color': 'red', 'label': 'Set 2'}],
    ...     title="Fault Analysis"
    ... )
    """
    sn = Stereonet(figsize=figsize, title=title, show_grid=show_grid)

    if planes:
        for p in planes:
            sn.plane(
                strike=p.get("strike"),
                dip=p.get("dip"),
                dip_direction=p.get("dip_direction"),
                color=p.get("color", "blue"),
                linewidth=p.get("linewidth", 1.5),
                linestyle=p.get("linestyle", "-"),
                label=p.get("label"),
            )

    if poles:
        for p in poles:
            sn.pole(
                strike=p.get("strike"),
                dip=p.get("dip"),
                dip_direction=p.get("dip_direction"),
                color=p.get("color", "red"),
                marker=p.get("marker", "o"),
                markersize=p.get("markersize", 6),
                label=p.get("label"),
            )

    if lines:
        for l in lines:
            sn.line(
                trend=l.get("trend"),
                plunge=l.get("plunge"),
                color=l.get("color", "green"),
                marker=l.get("marker", "o"),
                markersize=l.get("markersize", 6),
                label=l.get("label"),
            )

    if any(p.get("label") for p in (planes or [])) or \
       any(p.get("label") for p in (poles or [])) or \
       any(l.get("label") for l in (lines or [])):
        sn.legend()

    return sn.show()
