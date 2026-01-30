"""
Test and demonstration of Wulff stereonet for structural geology.
"""

import numpy as np
from xali_tools.plots import Stereonet, plotStereonet


def test_basic_stereonet():
    """Test basic stereonet with planes and poles."""
    sn = Stereonet(title="Basic Stereonet Test")

    # Plot a single plane as great circle
    sn.plane(strike=45, dip=30, color="blue", label="Fault plane")

    # Plot pole to the same plane
    sn.pole(strike=45, dip=30, color="red", marker="o", label="Pole")

    # Plot a lineation
    sn.line(trend=90, plunge=45, color="green", marker="s", label="Lineation")

    sn.legend()
    return sn.show()


def test_multiple_planes():
    """Test plotting multiple fault planes."""
    sn = Stereonet(title="Conjugate Fault System")

    # Conjugate fault set (typical Anderson faulting)
    # Set 1: NE-striking, dipping SE
    strikes1 = [40, 45, 50, 42, 48]
    dips1 = [60, 62, 58, 61, 59]
    for s, d in zip(strikes1, dips1):
        sn.plane(strike=s, dip=d, color="blue", alpha=0.7)

    # Set 2: NW-striking, dipping NE (conjugate)
    strikes2 = [130, 135, 128, 132]
    dips2 = [58, 60, 62, 59]
    for s, d in zip(strikes2, dips2):
        sn.plane(strike=s, dip=d, color="red", alpha=0.7)

    return sn.show()


def test_poles_with_density():
    """Test pole density contours (common for joint analysis)."""
    np.random.seed(42)

    # Simulate joint set with preferred orientation
    n_joints = 100
    strikes = np.random.normal(60, 15, n_joints) % 360
    dips = np.random.normal(75, 10, n_joints)
    dips = np.clip(dips, 0, 90)

    sn = Stereonet(title="Joint Pole Density")
    sn.density(strike=strikes, dip=dips, data_type="poles", cmap="Blues")
    sn.pole(strike=strikes, dip=dips, color="black", markersize=3, alpha=0.5)

    return sn.show()


def test_dip_direction_convention():
    """Test using dip direction instead of strike."""
    sn = Stereonet(title="Dip Direction Convention")

    # Same plane specified two ways
    # Strike 45, dip 60 (right-hand rule) = Dip direction 135, dip 60
    sn.plane(strike=45, dip=60, color="blue", linewidth=3, label="Strike/Dip")
    sn.plane(dip_direction=135, dip=60, color="red", linewidth=1.5,
             linestyle="--", label="DipDir/Dip")

    sn.legend()
    return sn.show()


def test_lineations():
    """Test plotting lineations (e.g., slickenlines, fold axes)."""
    np.random.seed(123)

    sn = Stereonet(title="Fault Slickenlines")

    # Fault plane
    sn.plane(strike=30, dip=55, color="gray", linewidth=2, label="Fault plane")

    # Slickenlines on the fault (rake ~30 degrees)
    n_lines = 15
    trends = np.random.normal(75, 8, n_lines) % 360
    plunges = np.random.normal(25, 5, n_lines)
    plunges = np.clip(plunges, 0, 55)  # Must be less than dip

    sn.line(trends, plunges, color="red", marker="^", markersize=8,
            label="Slickenlines")

    sn.legend()
    return sn.show()


def test_vectors_3d():
    """Test plotting 3D vectors (e.g., principal stress directions)."""
    sn = Stereonet(title="Principal Stress Directions")

    # Sigma 1 (maximum compression) - subvertical
    sigma1 = np.array([[0.1, 0.1, -0.98]])
    sn.vector(sigma1, color="red", marker="o", markersize=12, label="σ1")

    # Sigma 2 (intermediate) - subhorizontal NE
    sigma2 = np.array([[0.6, 0.6, -0.2]])
    sn.vector(sigma2, color="green", marker="s", markersize=10, label="σ2")

    # Sigma 3 (minimum) - subhorizontal NW
    sigma3 = np.array([[-0.6, 0.6, -0.1]])
    sn.vector(sigma3, color="blue", marker="^", markersize=10, label="σ3")

    sn.legend()
    return sn.show()


def test_quick_function():
    """Test the quick plotStereonet function."""
    fig, ax = plotStereonet(
        planes=[
            {"strike": [30, 35, 40], "dip": [50, 55, 52], "color": "blue", "label": "Set A"},
            {"strike": [120, 125], "dip": [70, 68], "color": "purple", "label": "Set B"},
        ],
        poles=[
            {"strike": [80, 85, 90], "dip": [40, 42, 38], "color": "red", "label": "Joints"},
        ],
        lines=[
            {"trend": [45, 50], "plunge": [30, 35], "color": "green", "marker": "^", "label": "Fold axes"},
        ],
        title="Quick Stereonet Plot",
    )
    return fig, ax


if __name__ == "__main__":
    print("Testing Wulff stereonet functions...\n")

    print("1. Basic stereonet (plane, pole, line)")
    test_basic_stereonet()

    print("\n2. Multiple conjugate fault planes")
    test_multiple_planes()

    print("\n3. Pole density contours")
    test_poles_with_density()

    print("\n4. Dip direction convention")
    test_dip_direction_convention()

    print("\n5. Lineations on fault plane")
    test_lineations()

    print("\n6. 3D vectors (principal stresses)")
    test_vectors_3d()

    print("\n7. Quick plotStereonet function")
    test_quick_function()

    print("\nAll stereonet tests completed!")
