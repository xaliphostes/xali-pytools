"""Tests for TSurf file loading/saving with ESIZES support."""

import numpy as np
import pytest
import tempfile
import os
from xali_tools.io.tsurf_filter import load_tsurf, save_tsurf
from xali_tools.io import SurfaceData


class TestTSurfLoad:
    """Tests for loading TSurf files."""

    def test_load_properties(self):
        """Load file contains expected properties."""
        data = load_tsurf('tests/two-triangles.gcd')

        assert 'U' in data.properties
        assert 'a' in data.properties
        assert 'b' in data.properties

    def test_vector_property_shape(self):
        """Vector property U has correct shape."""
        data = load_tsurf('tests/two-triangles.gcd')

        U = data.properties['U']
        # U should be 4 vertices x 3 components
        assert U.shape[1] == 3 or len(U.flatten()) % 3 == 0

    def test_scalar_properties_shape(self):
        """Scalar properties have correct shape."""
        data = load_tsurf('tests/two-triangles.gcd')

        a = data.properties['a']
        b = data.properties['b']

        # Scalars should be 1D or have single column
        assert a.ndim == 1 or a.shape[1] == 1
        assert b.ndim == 1 or b.shape[1] == 1


class TestTSurfRoundtrip:
    """Tests for save/reload roundtrip."""

    def test_save_and_reload(self):
        """Saved file can be reloaded."""
        original = load_tsurf('tests/two-triangles.gcd')

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            assert 'U' in reloaded.properties
            assert 'a' in reloaded.properties
            assert 'b' in reloaded.properties
        finally:
            os.unlink(filepath)

    def test_data_preserved(self):
        """Property data is preserved through roundtrip."""
        original = load_tsurf('tests/two-triangles.gcd')

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            np.testing.assert_allclose(
                reloaded.properties['U'], original.properties['U']
            )
            np.testing.assert_allclose(
                reloaded.properties['a'], original.properties['a']
            )
            np.testing.assert_allclose(
                reloaded.properties['b'], original.properties['b']
            )
        finally:
            os.unlink(filepath)


class TestTSurfESIZESFormat:
    """Tests for correct ESIZES format with scalar, vector, and tensor properties."""

    def test_esizes_scalar_vector_tensor(self):
        """Test that ESIZES correctly contains '1 3 6' for scalar, vector, tensor."""
        n_vertices = 4

        # Create surface with scalar (1), vector (3), and tensor (6) properties
        positions = np.array([
            0, 0, 0,
            1, 0, 0,
            1, 1, 0,
            0, 1, 0
        ], dtype=np.float64)

        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Scalar property (item_size=1)
        scalar_prop = np.array([1.0, 2.0, 3.0, 4.0])

        # Vector property (item_size=3)
        vector_prop = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2]
        ])

        # Symmetric tensor property (item_size=6: xx, xy, xz, yy, yz, zz)
        tensor_prop = np.array([
            [1.0, 0.1, 0.2, 2.0, 0.3, 3.0],
            [1.1, 0.4, 0.5, 2.1, 0.6, 3.1],
            [1.2, 0.7, 0.8, 2.2, 0.9, 3.2],
            [1.3, 1.0, 1.1, 2.3, 1.2, 3.3]
        ])

        surface = SurfaceData(
            positions=positions,
            indices=indices,
            properties={
                'temperature': scalar_prop,
                'displacement': vector_prop,
                'stress': tensor_prop
            },
            property_sizes={
                'temperature': 1,
                'displacement': 3,
                'stress': 6
            },
            name='test_surface'
        )
        # surface.set_property("temperature", scalar_prop, 1)
        # surface.set_property("displacement", vector_prop, 3)
        # surface.set_property("stress", tensor_prop, 6)

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False, mode='w') as f:
            filepath = f.name

        try:
            save_tsurf(surface, filepath)

            # Read the raw file content
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')

            # Find and verify ESIZES line
            esizes_line = None
            for line in lines:
                if line.startswith('ESIZES'):
                    esizes_line = line
                    break

            assert esizes_line is not None, "ESIZES line not found in file"

            # Extract ESIZES values
            esizes_values = esizes_line.split()[1:]
            esizes_ints = [int(v) for v in esizes_values]

            # Check that we have sizes 1, 3, 6 (order may vary based on dict iteration)
            assert sorted(esizes_ints) == [1, 3, 6], \
                f"Expected ESIZES to contain 1, 3, 6 but got {esizes_ints}"

            # Verify total attributes per vertex = 1 + 3 + 6 = 10
            expected_total_attrs = 1 + 3 + 6

            # Check each PVRTX line has correct number of attribute values
            pvrtx_lines = [l for l in lines if l.startswith('PVRTX')]
            assert len(pvrtx_lines) == n_vertices, \
                f"Expected {n_vertices} PVRTX lines, got {len(pvrtx_lines)}"

            for pvrtx_line in pvrtx_lines:
                parts = pvrtx_line.split()
                # Format: PVRTX id x y z attr1 attr2 ... attrN
                # parts[0] = 'PVRTX', parts[1] = id, parts[2:5] = x y z
                # parts[5:] = attribute values
                attr_values = parts[5:]
                assert len(attr_values) == expected_total_attrs, \
                    f"Expected {expected_total_attrs} attribute values per vertex, " \
                    f"got {len(attr_values)} in line: {pvrtx_line}"

        finally:
            os.unlink(filepath)

    def test_esizes_roundtrip_preserves_data(self):
        """Test that scalar, vector, tensor data is preserved through save/load."""
        # Create surface with all property types
        positions = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        scalar_prop = np.array([10.0, 20.0, 30.0])
        vector_prop = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        tensor_prop = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18]
        ], dtype=np.float64)

        original = SurfaceData(
            positions=positions,
            indices=indices,
            properties={
                'scalar': scalar_prop,
                'vector': vector_prop,
                'tensor': tensor_prop
            },
            property_sizes={'scalar': 1, 'vector': 3, 'tensor': 6},
            name='roundtrip_test'
        )

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            filepath = f.name

        try:
            save_tsurf(original, filepath)
            reloaded = load_tsurf(filepath)

            # Verify property sizes are preserved
            assert reloaded.property_sizes.get('scalar') == 1
            assert reloaded.property_sizes.get('vector') == 3
            assert reloaded.property_sizes.get('tensor') == 6

            # Verify data is preserved
            np.testing.assert_allclose(reloaded.properties['scalar'], scalar_prop)
            np.testing.assert_allclose(reloaded.properties['vector'], vector_prop)
            np.testing.assert_allclose(reloaded.properties['tensor'], tensor_prop)

        finally:
            os.unlink(filepath)
