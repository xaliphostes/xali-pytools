"""Tests for Attribute and SurfaceData property handling."""

import numpy as np
import pytest
import tempfile
import os
from xali_tools.core import Attribute
from xali_tools.io import SurfaceData
from xali_tools.io.tsurf_filter import load_tsurf, save_tsurf


class TestAttributeScalar:
    """Tests for scalar Attribute (item_size=1)."""

    def test_scalar_properties(self):
        """Scalar attribute has correct properties."""
        pressure = Attribute([1.0, 2.0, 3.0, 4.0], item_size=1, name='pressure')

        assert pressure.item_size == 1
        assert pressure.n_items == 4
        assert pressure.is_scalar is True
        assert pressure.is_vector is False
        assert pressure.shape == (4,)

    def test_scalar_iteration(self):
        """Can iterate over scalar values."""
        pressure = Attribute([1.0, 2.0, 3.0, 4.0], item_size=1)
        values = list(pressure)
        assert values == [1.0, 2.0, 3.0, 4.0]

    def test_scalar_indexing(self):
        """Can index scalar values."""
        pressure = Attribute([1.0, 2.0, 3.0, 4.0], item_size=1)
        assert pressure[0] == 1.0
        assert pressure[2] == 3.0

    def test_scalar_flatten(self):
        """Flatten returns 1D array."""
        pressure = Attribute([1.0, 2.0, 3.0, 4.0], item_size=1)
        np.testing.assert_array_equal(pressure.flatten(), [1.0, 2.0, 3.0, 4.0])


class TestAttributeVector:
    """Tests for vector Attribute (item_size=3)."""

    def test_vector_properties(self):
        """Vector attribute has correct properties."""
        velocity = Attribute([[1, 2, 3], [4, 5, 6], [7, 8, 9]], name='velocity')

        assert velocity.item_size == 3
        assert velocity.n_items == 3
        assert velocity.is_scalar is False
        assert velocity.is_vector is True
        assert velocity.shape == (3, 3)

    def test_vector_iteration(self):
        """Can iterate over vector items."""
        velocity = Attribute([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        items = list(velocity)

        assert len(items) == 3
        np.testing.assert_array_equal(items[0], [1, 2, 3])
        np.testing.assert_array_equal(items[1], [4, 5, 6])

    def test_vector_indexing(self):
        """Can index vector items."""
        velocity = Attribute([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        np.testing.assert_array_equal(velocity[0], [1, 2, 3])
        np.testing.assert_array_equal(velocity[1:3], [[4, 5, 6], [7, 8, 9]])

    def test_vector_flatten(self):
        """Flatten returns all components in 1D."""
        velocity = Attribute([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(velocity.flatten(), [1, 2, 3, 4, 5, 6, 7, 8, 9])


class TestAttributeTensor:
    """Tests for tensor Attribute (item_size=6)."""

    def test_tensor_properties(self):
        """Tensor attribute has correct properties."""
        stress_data = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12]
        ], dtype=np.float64)
        stress = Attribute(stress_data, name='stress')

        assert stress.item_size == 6
        assert stress.n_items == 2
        assert stress.is_tensor is True
        assert stress.shape == (2, 6)

    def test_tensor_iteration(self):
        """Can iterate over tensor items."""
        stress_data = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12]
        ], dtype=np.float64)
        stress = Attribute(stress_data)
        items = list(stress)

        np.testing.assert_array_equal(items[0], [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(items[1], [7, 8, 9, 10, 11, 12])


class TestAttributeCreation:
    """Tests for Attribute creation methods."""

    def test_from_flat(self):
        """Can create Attribute from flat array."""
        flat_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        vec = Attribute.from_flat(flat_data, item_size=3, name='vec')

        assert vec.n_items == 3
        assert vec.item_size == 3
        np.testing.assert_array_equal(vec[0], [1, 2, 3])
        np.testing.assert_array_equal(vec[1], [4, 5, 6])
        np.testing.assert_array_equal(vec[2], [7, 8, 9])

    def test_as_view_shares_memory(self):
        """as_view creates a true view (no copy)."""
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        view = Attribute.as_view(original, item_size=3, name='velocity')

        assert np.shares_memory(original, view._data)

    def test_as_view_modifications_propagate(self):
        """Modifications via view affect original and vice versa."""
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        view = Attribute.as_view(original, item_size=3)

        # Modify via view
        view[0] = [10, 20, 30]
        np.testing.assert_array_equal(original[0], [10, 20, 30])

        # Modify original
        original[1] = [40, 50, 60]
        np.testing.assert_array_equal(view[1], [40, 50, 60])

    def test_copy_is_independent(self):
        """copy creates an independent copy."""
        original = Attribute([[1, 2, 3], [4, 5, 6]], name='velocity')
        copied = original.copy()

        assert not np.shares_memory(original._data, copied._data)

        # Modify copy, original unchanged
        copied[0] = [10, 20, 30]
        np.testing.assert_array_equal(original[0], [1, 2, 3])


class TestAttributeSetItem:
    """Tests for Attribute __setitem__."""

    def test_setitem_vector(self):
        """Can set vector values."""
        vec = Attribute([[1, 2, 3], [4, 5, 6]], name='vec')
        vec[0] = [10, 20, 30]
        np.testing.assert_array_equal(vec[0], [10, 20, 30])

    def test_setitem_scalar(self):
        """Can set scalar values."""
        scalar = Attribute([1.0, 2.0, 3.0], item_size=1)
        scalar[1] = 99.0
        assert scalar[1] == 99.0


class TestSurfaceDataProperties:
    """Tests for SurfaceData property handling."""

    def test_get_property_returns_view(self):
        """get_property returns a view of the data."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        U_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        a_data = np.array([0.1, 0.2, 0.3], dtype=np.float64)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            properties={'U': U_data, 'a': a_data},
            property_sizes={'U': 3, 'a': 1},
            name='test'
        )

        U = surface.get_property('U')
        assert U.is_vector
        assert U.n_items == 3

        a = surface.get_property('a')
        assert a.is_scalar
        assert a.n_items == 3

        # Verify it's a view
        assert np.shares_memory(surface.properties['U'], U._data)

    def test_set_property_scalar(self):
        """Can set scalar property."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
        surface = SurfaceData(positions=positions, name='test')

        surface.set_property('pressure', [1.0, 2.0, 3.0], item_size=1)

        assert 'pressure' in surface.properties
        assert surface.property_sizes['pressure'] == 1

    def test_set_property_vector(self):
        """Can set vector property."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
        surface = SurfaceData(positions=positions, name='test')

        surface.set_property('velocity', [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert 'velocity' in surface.properties
        assert surface.property_sizes['velocity'] == 3

        vel = surface.get_property('velocity')
        assert vel.is_vector


class TestTSurfWithESizes:
    """Tests for TSurf file loading with ESIZES."""

    def test_load_with_esizes(self):
        """Loading TSurf with ESIZES populates property_sizes."""
        data = load_tsurf('tests/two-triangles.gcd')

        assert 'U' in data.properties
        assert 'a' in data.properties
        assert 'b' in data.properties

        assert data.property_sizes['U'] == 3
        assert data.property_sizes['a'] == 1
        assert data.property_sizes['b'] == 1

    def test_get_property_from_loaded(self):
        """Can get properties as Attributes from loaded file."""
        data = load_tsurf('tests/two-triangles.gcd')

        U = data.get_property('U')
        assert U.is_vector
        assert U.n_items == 4

        a = data.get_property('a')
        assert a.is_scalar

    def test_roundtrip_preserves_property_sizes(self):
        """Save and reload preserves property_sizes."""
        original = load_tsurf('tests/two-triangles.gcd')

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            assert reloaded.property_sizes == original.property_sizes
            np.testing.assert_array_almost_equal(
                reloaded.properties['U'], original.properties['U']
            )
            np.testing.assert_array_almost_equal(
                reloaded.properties['a'], original.properties['a']
            )
        finally:
            os.unlink(filepath)


class TestAttributeIteration:
    """Tests demonstrating iteration over properties."""

    def test_iterate_vector(self):
        """Can iterate over vector property."""
        data = load_tsurf('tests/two-triangles.gcd')
        U = data.get_property('U')

        count = 0
        for vec in U:
            assert len(vec) == 3
            count += 1
        assert count == U.n_items

    def test_iterate_scalar(self):
        """Can iterate over scalar property."""
        data = load_tsurf('tests/two-triangles.gcd')
        a = data.get_property('a')

        values = list(a)
        assert len(values) == a.n_items
