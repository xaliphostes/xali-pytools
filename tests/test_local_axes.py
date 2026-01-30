"""
Tests for local coordinate system computation.
"""

import numpy as np
import pytest

from xali_tools.geom import compute_local_axes, compute_triangle_normals, set_local_axes
from xali_tools.io import SurfaceData


class TestComputeTriangleNormals:
    """Tests for compute_triangle_normals."""

    def test_single_horizontal_triangle(self):
        """A horizontal triangle should have a vertical normal."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        indices = np.array([[0, 1, 2]])

        normals = compute_triangle_normals(positions, indices)

        assert normals.shape == (1, 3)
        # Normal should point in +z direction (right-hand rule)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-10)

    def test_vertical_triangle(self):
        """A vertical triangle should have a horizontal normal."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        indices = np.array([[0, 1, 2]])

        normals = compute_triangle_normals(positions, indices)

        assert normals.shape == (1, 3)
        # Normal should point in -y direction
        np.testing.assert_allclose(normals[0], [0.0, -1.0, 0.0], atol=1e-10)

    def test_multiple_triangles(self):
        """Test with multiple triangles."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        indices = np.array([
            [0, 1, 2],  # Horizontal
            [0, 1, 3],  # Vertical (facing -y)
        ])

        normals = compute_triangle_normals(positions, indices)

        assert normals.shape == (2, 3)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(normals[1], [0.0, -1.0, 0.0], atol=1e-10)


class TestComputeLocalAxes:
    """Tests for compute_local_axes."""

    def test_horizontal_triangle(self):
        """Horizontal triangle should use conventional axes."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        indices = np.array([[0, 1, 2]])

        dip, strike, normal = compute_local_axes(positions, indices)

        # Normal should be vertical
        np.testing.assert_allclose(normal[0], [0.0, 0.0, 1.0], atol=1e-10)
        # Conventional strike for horizontal plane (eY * n[2])
        np.testing.assert_allclose(strike[0], [0.0, 1.0, 0.0], atol=1e-10)
        # Dip = n × strike = (0,0,1) × (0,1,0) = (-1,0,0)
        np.testing.assert_allclose(dip[0], [-1.0, 0.0, 0.0], atol=1e-10)

    def test_dipping_triangle(self):
        """Triangle dipping toward +x: test local axes computation."""
        # Create a triangle dipping 45 degrees toward +x
        positions = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ])
        indices = np.array([[0, 1, 2]])

        dip, strike, normal = compute_local_axes(positions, indices)

        # Check that axes are orthonormal
        assert np.abs(np.dot(dip[0], strike[0])) < 1e-10
        assert np.abs(np.dot(dip[0], normal[0])) < 1e-10
        assert np.abs(np.dot(strike[0], normal[0])) < 1e-10

        # Check unit length
        np.testing.assert_allclose(np.linalg.norm(dip[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(strike[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(normal[0]), 1.0, atol=1e-10)

        # Strike should be horizontal (z component = 0)
        assert np.abs(strike[0, 2]) < 1e-10

        # With C++ convention: dip = normal × strike
        # For this triangle: normal ≈ (0.707, 0, 0.707), strike = (0, 1, 0)
        # dip = n × strike ≈ (-0.707, 0, 0.707)
        assert dip[0, 0] < 0  # Negative x component per C++ convention

    def test_vertical_triangle_facing_y(self):
        """Vertical triangle facing +y should have horizontal dip axis."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
        ])
        indices = np.array([[0, 1, 2]])

        dip, strike, normal = compute_local_axes(positions, indices)

        # Normal should be horizontal (in y direction)
        assert np.abs(normal[0, 2]) < 1e-10

        # Strike should be horizontal
        assert np.abs(strike[0, 2]) < 1e-10

        # Check orthonormality
        assert np.abs(np.dot(dip[0], strike[0])) < 1e-10
        assert np.abs(np.dot(dip[0], normal[0])) < 1e-10

    def test_flat_array_input(self):
        """Test with flat array inputs."""
        positions = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        indices = np.array([0, 1, 2])

        dip, strike, normal = compute_local_axes(positions, indices)

        assert dip.shape == (1, 3)
        assert strike.shape == (1, 3)
        assert normal.shape == (1, 3)

    def test_right_hand_rule(self):
        """Verify that dip × strike = normal (right-hand rule)."""
        positions = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5],
        ])
        indices = np.array([[0, 1, 2]])

        dip, strike, normal = compute_local_axes(positions, indices)

        # dip × strike should give the normal (approximately)
        cross = np.cross(dip[0], strike[0])
        # May be parallel or anti-parallel
        dot = np.dot(cross, normal[0])
        assert np.abs(np.abs(dot) - 1.0) < 1e-10


class TestSetLocalAxes:
    """Tests for set_local_axes."""

    def test_set_properties(self):
        """Test that properties are correctly set on SurfaceData."""
        positions = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ])
        indices = np.array([0, 1, 2, 0, 1, 3], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            properties={},
            property_sizes={},
            name="test"
        )

        result = set_local_axes(surface)

        # Should return the same object
        assert result is surface

        # Check properties are set
        assert "dip_axis" in surface.properties
        assert "strike_axis" in surface.properties
        assert "normal_axis" in surface.properties

        # Check property sizes
        assert surface.property_sizes["dip_axis"] == 3
        assert surface.property_sizes["strike_axis"] == 3
        assert surface.property_sizes["normal_axis"] == 3

        # Check shapes - properties are now per-vertex (4 vertices), not per-triangle
        assert surface.properties["dip_axis"].shape == (4, 3)
        assert surface.properties["strike_axis"].shape == (4, 3)
        assert surface.properties["normal_axis"].shape == (4, 3)

    def test_custom_property_names(self):
        """Test with custom property names."""
        positions = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        indices = np.array([0, 1, 2], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            properties={},
            property_sizes={},
        )

        set_local_axes(
            surface,
            dip_name="x_local",
            strike_name="y_local",
            normal_name="z_local"
        )

        assert "x_local" in surface.properties
        assert "y_local" in surface.properties
        assert "z_local" in surface.properties

    def test_no_indices_raises(self):
        """Test that missing indices raises an error."""
        positions = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        surface = SurfaceData(
            positions=positions,
            indices=None,
            properties={},
            property_sizes={},
        )

        with pytest.raises(ValueError, match="must have triangle indices"):
            set_local_axes(surface)


class TestOrthonormality:
    """Test that all axes form orthonormal bases."""

    def test_random_triangles(self):
        """Test orthonormality for random triangles."""
        np.random.seed(42)
        n_triangles = 100

        # Generate random triangles
        positions = np.random.randn(n_triangles * 3, 3)
        indices = np.arange(n_triangles * 3).reshape(-1, 3)

        dip, strike, normal = compute_local_axes(positions, indices)

        for i in range(n_triangles):
            # Check unit lengths
            np.testing.assert_allclose(np.linalg.norm(dip[i]), 1.0, atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(strike[i]), 1.0, atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(normal[i]), 1.0, atol=1e-10)

            # Check orthogonality
            assert np.abs(np.dot(dip[i], strike[i])) < 1e-10
            assert np.abs(np.dot(dip[i], normal[i])) < 1e-10
            assert np.abs(np.dot(strike[i], normal[i])) < 1e-10

    def test_strike_is_horizontal(self):
        """Test that strike axis is always horizontal (z=0) for non-horizontal triangles."""
        np.random.seed(123)
        n_triangles = 50

        # Generate random non-horizontal triangles
        positions = np.random.randn(n_triangles * 3, 3)
        indices = np.arange(n_triangles * 3).reshape(-1, 3)

        dip, strike, normal = compute_local_axes(positions, indices)

        # Check if triangles are nearly horizontal
        horizontal_threshold = 1e-6
        is_horizontal = np.abs(np.abs(normal[:, 2]) - 1.0) < horizontal_threshold

        # For non-horizontal triangles, strike should be horizontal
        for i in range(n_triangles):
            if not is_horizontal[i]:
                assert np.abs(strike[i, 2]) < 1e-10, f"Triangle {i}: strike z = {strike[i, 2]}"
