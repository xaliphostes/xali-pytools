"""
Test and demonstration of rose diagram plotting for geological orientations.
"""

import numpy as np
from xali_tools.plots import plotRoseDiagram, plotRoseDiagramFromVectors, compute_circular_statistics


def test_basic_rose_diagram():
    """Test basic rose diagram with simulated fracture orientations."""
    # Simulate fracture orientations with a preferred NE-SW trend (45 degrees)
    np.random.seed(42)
    n_fractures = 200

    # Von Mises distribution centered at 45 degrees (NE direction)
    # kappa controls concentration (higher = more concentrated)
    mean_direction = np.radians(45)
    kappa = 3.0
    azimuths = np.random.vonmises(mean_direction, kappa, n_fractures)
    azimuths = np.degrees(azimuths) % 360

    fig, ax = plotRoseDiagram(
        azimuths,
        num_bins=36,
        bidirectional=True,
        title="Fracture Orientations (NE-SW trend)",
        color="steelblue",
    )
    return fig, ax


def test_weighted_rose_diagram():
    """Test rose diagram with weighted data (e.g., fracture lengths)."""
    np.random.seed(123)
    n_fractures = 150

    # Two populations: NE-SW (dominant) and E-W (secondary)
    azimuths_ne = np.random.normal(40, 15, 100) % 180
    azimuths_ew = np.random.normal(90, 10, 50) % 180
    azimuths = np.concatenate([azimuths_ne, azimuths_ew])

    # Weights representing fracture lengths (NE fractures are longer)
    weights_ne = np.random.exponential(10, 100)
    weights_ew = np.random.exponential(5, 50)
    weights = np.concatenate([weights_ne, weights_ew])

    fig, ax = plotRoseDiagram(
        azimuths,
        weights=weights,
        num_bins=36,
        bidirectional=True,
        cmap="plasma",
        title="Fracture Orientations (weighted by length)",
    )
    return fig, ax


def test_unidirectional_rose_diagram():
    """Test rose diagram for unidirectional data (e.g., paleocurrent directions)."""
    np.random.seed(456)
    n_measurements = 100

    # Paleocurrent flowing towards SE (135 degrees)
    azimuths = np.random.normal(135, 20, n_measurements) % 360

    fig, ax = plotRoseDiagram(
        azimuths,
        num_bins=36,
        bidirectional=False,  # Unidirectional data
        color="coral",
        title="Paleocurrent Directions",
    )
    return fig, ax


def test_rose_diagram_from_vectors():
    """Test rose diagram from 3D vectors (e.g., fault slip directions)."""
    np.random.seed(789)
    n_vectors = 100

    # Create vectors trending NE with some spread
    angles = np.random.normal(np.radians(60), np.radians(15), n_vectors)
    plunges = np.random.normal(np.radians(30), np.radians(10), n_vectors)

    # Convert to Cartesian (x=East, y=North, z=Up)
    x = np.cos(plunges) * np.sin(angles)
    y = np.cos(plunges) * np.cos(angles)
    z = np.sin(plunges)
    vectors = np.column_stack([x, y, z])

    fig, ax = plotRoseDiagramFromVectors(
        vectors,
        num_bins=36,
        bidirectional=True,
        cmap="coolwarm",
        title="Fault Slip Directions",
    )
    return fig, ax


def test_circular_statistics():
    """Test circular statistics computation."""
    # Create data with known properties
    azimuths = np.array([40, 45, 50, 220, 225, 230])  # Bidirectional NE-SW

    stats = compute_circular_statistics(azimuths, bidirectional=True)

    print("Circular Statistics Test:")
    print(f"  n = {stats['n']}")
    print(f"  Mean direction = {stats['mean_direction']:.1f}°")
    print(f"  Resultant length (R) = {stats['resultant_length']:.3f}")
    print(f"  Circular variance = {stats['circular_variance']:.3f}")

    # R should be high for concentrated data
    assert stats["resultant_length"] > 0.9, "R should be high for concentrated data"
    # Mean should be around 45 degrees
    assert 40 < stats["mean_direction"] < 50, "Mean should be around 45°"

    return stats


if __name__ == "__main__":
    print("Testing rose diagram plotting functions...\n")

    print("1. Basic rose diagram with fracture orientations")
    test_basic_rose_diagram()

    print("\n2. Weighted rose diagram")
    test_weighted_rose_diagram()

    print("\n3. Unidirectional rose diagram (paleocurrents)")
    test_unidirectional_rose_diagram()

    print("\n4. Rose diagram from 3D vectors")
    test_rose_diagram_from_vectors()

    print("\n5. Circular statistics")
    test_circular_statistics()

    print("\nAll tests completed!")
