"""Tests for get_tsurf and save_tsurf functions."""

import numpy as np
import pytest
import tempfile
import os
from xali_tools.io.tsurf_filter import get_tsurf, save_tsurf, load_tsurf
from xali_tools.io import SurfaceData


class TestGetTsurf:
    """Tests for get_tsurf function."""

    def test_get_tsurf_returns_string(self):
        """get_tsurf returns a non-empty string."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            name='test'
        )

        result = get_tsurf(surface)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_tsurf_contains_header(self):
        """get_tsurf output contains GOCAD header."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            name='my_surface'
        )

        result = get_tsurf(surface)
        assert 'GOCAD TSurf 1' in result
        assert 'HEADER' in result
        assert 'my_surface' in result
        assert 'END' in result

    def test_get_tsurf_contains_vertices(self):
        """get_tsurf output contains VRTX lines for each vertex."""
        positions = np.array([
            0, 0, 0,
            1, 0, 0,
            0.5, 1, 0
        ], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            name='test'
        )

        result = get_tsurf(surface)
        # Without properties, should use VRTX
        assert 'VRTX 1' in result
        assert 'VRTX 2' in result
        assert 'VRTX 3' in result

    def test_get_tsurf_contains_triangles(self):
        """get_tsurf output contains TRGL lines for each triangle."""
        positions = np.array([
            0, 0, 0,
            1, 0, 0,
            0.5, 1, 0,
            0.5, 0.5, 1
        ], dtype=np.float64)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            name='test'
        )

        result = get_tsurf(surface)
        # Triangles should use 1-based indexing
        assert 'TRGL 1 2 3' in result
        assert 'TRGL 1 3 4' in result

    def test_get_tsurf_with_properties(self):
        """get_tsurf output contains PVRTX with property values."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        scalar_prop = np.array([1.0, 2.0, 3.0])

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            properties={'temperature': scalar_prop},
            property_sizes={'temperature': 1},
            name='test'
        )

        result = get_tsurf(surface)
        # With properties, should use PVRTX
        assert 'PVRTX 1' in result
        assert 'PROPERTIES temperature' in result
        assert 'ESIZES 1' in result

    def test_get_tsurf_vector_property(self):
        """get_tsurf handles vector properties correctly."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        vector_prop = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            properties={'displacement': vector_prop},
            property_sizes={'displacement': 3},
            name='test'
        )

        result = get_tsurf(surface)
        assert 'PROPERTIES displacement' in result
        assert 'ESIZES 3' in result


class TestSaveTsurf:
    """Tests for save_tsurf function."""

    def test_save_tsurf_creates_file(self):
        """save_tsurf creates a file at the specified path."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            name='test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(surface, filepath)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)

    def test_save_tsurf_file_content_matches_get_tsurf(self):
        """save_tsurf writes same content as get_tsurf returns."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            name='test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(surface, filepath)
            with open(filepath, 'r') as f:
                file_content = f.read()

            expected_content = get_tsurf(surface)
            assert file_content == expected_content
        finally:
            os.unlink(filepath)


class TestGetSaveTsurfRoundtrip:
    """Tests for roundtrip: save then load."""

    def test_roundtrip_positions_preserved(self):
        """Positions are preserved after save/load roundtrip."""
        positions = np.array([
            0.5, 1.5, 2.5,
            3.5, 4.5, 5.5,
            6.5, 7.5, 8.5
        ], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        original = SurfaceData(
            positions=positions,
            indices=indices,
            name='roundtrip_test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            np.testing.assert_allclose(reloaded.positions, original.positions, rtol=1e-5)
        finally:
            os.unlink(filepath)

    def test_roundtrip_indices_preserved(self):
        """Indices are preserved after save/load roundtrip."""
        positions = np.array([
            0, 0, 0,
            1, 0, 0,
            1, 1, 0,
            0, 1, 0
        ], dtype=np.float64)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        original = SurfaceData(
            positions=positions,
            indices=indices,
            name='test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            np.testing.assert_array_equal(reloaded.indices, original.indices)
        finally:
            os.unlink(filepath)

    def test_roundtrip_name_preserved(self):
        """Surface name is preserved after save/load roundtrip."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        original = SurfaceData(
            positions=positions,
            indices=indices,
            name='my_custom_surface_name'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            assert reloaded.name == original.name
        finally:
            os.unlink(filepath)

    def test_roundtrip_scalar_property_preserved(self):
        """Scalar property values are preserved after roundtrip."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        scalar_prop = np.array([10.5, 20.5, 30.5])

        original = SurfaceData(
            positions=positions,
            indices=indices,
            properties={'value': scalar_prop},
            property_sizes={'value': 1},
            name='test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            assert 'value' in reloaded.properties
            np.testing.assert_allclose(reloaded.properties['value'], scalar_prop, rtol=1e-5)
        finally:
            os.unlink(filepath)

    def test_roundtrip_vector_property_preserved(self):
        """Vector property values are preserved after roundtrip."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        vector_prop = np.array([
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [7.7, 8.8, 9.9]
        ])

        original = SurfaceData(
            positions=positions,
            indices=indices,
            properties={'velocity': vector_prop},
            property_sizes={'velocity': 3},
            name='test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            assert 'velocity' in reloaded.properties
            np.testing.assert_allclose(reloaded.properties['velocity'], vector_prop, rtol=1e-5)
            assert reloaded.property_sizes.get('velocity') == 3
        finally:
            os.unlink(filepath)

    def test_roundtrip_multiple_properties_preserved(self):
        """Multiple properties of different types are preserved."""
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        scalar_prop = np.array([1.0, 2.0, 3.0])
        vector_prop = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        original = SurfaceData(
            positions=positions,
            indices=indices,
            properties={
                'temperature': scalar_prop,
                'displacement': vector_prop
            },
            property_sizes={
                'temperature': 1,
                'displacement': 3
            },
            name='multi_prop_test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            assert 'temperature' in reloaded.properties
            assert 'displacement' in reloaded.properties
            np.testing.assert_allclose(reloaded.properties['temperature'], scalar_prop, rtol=1e-5)
            np.testing.assert_allclose(reloaded.properties['displacement'], vector_prop, rtol=1e-5)
        finally:
            os.unlink(filepath)
