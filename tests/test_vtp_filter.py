"""Tests for VTP file loading and conversion to TSurf."""

import numpy as np
import pytest
import tempfile
import os

# Check if pyvista is available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

pytestmark = pytest.mark.skipif(not HAS_PYVISTA, reason="pyvista not installed")


from xali_tools.io.vtp_filter import load_vtp, vtp_to_tsurf, get_vtp_info, load_all_vtp
from xali_tools.io.tsurf_filter import load_tsurf
from xali_tools.io import SurfaceData


def create_test_vtp(filepath: str, with_properties: bool = False) -> None:
    """Create a simple test VTP file."""
    # Create a simple triangulated surface
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=np.float64)

    # Two triangles
    faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])

    mesh = pv.PolyData(points, faces)

    if with_properties:
        # Add scalar property
        mesh.point_data['temperature'] = np.array([10.0, 20.0, 30.0, 40.0])
        # Add vector property
        mesh.point_data['velocity'] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

    mesh.save(filepath)


class TestLoadVtp:
    """Tests for load_vtp function."""

    def test_load_vtp_returns_surface_data(self):
        """load_vtp returns a SurfaceData object."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath)
            surface = load_vtp(filepath)
            assert isinstance(surface, SurfaceData)
        finally:
            os.unlink(filepath)

    def test_load_vtp_positions(self):
        """load_vtp correctly loads vertex positions."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath)
            surface = load_vtp(filepath)

            assert surface.n_vertices == 4
            positions = surface.get_positions_matrix()
            np.testing.assert_allclose(positions[0], [0, 0, 0])
            np.testing.assert_allclose(positions[1], [1, 0, 0])
        finally:
            os.unlink(filepath)

    def test_load_vtp_indices(self):
        """load_vtp correctly loads triangle indices."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath)
            surface = load_vtp(filepath)

            assert surface.n_triangles == 2
            indices = surface.get_indices_matrix()
            # Check triangles are present (order may vary)
            assert len(indices) == 2
        finally:
            os.unlink(filepath)

    def test_load_vtp_with_properties(self):
        """load_vtp correctly loads point data properties."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath, with_properties=True)
            surface = load_vtp(filepath)

            assert 'temperature' in surface.properties
            assert 'velocity' in surface.properties

            np.testing.assert_allclose(
                surface.properties['temperature'],
                [10.0, 20.0, 30.0, 40.0]
            )
        finally:
            os.unlink(filepath)

    def test_load_vtp_custom_name(self):
        """load_vtp uses custom name when provided."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath)
            surface = load_vtp(filepath, name='my_surface')
            assert surface.name == 'my_surface'
        finally:
            os.unlink(filepath)

    def test_load_vtp_file_not_found(self):
        """load_vtp raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_vtp('nonexistent_file.vtp')


class TestVtpToTsurf:
    """Tests for vtp_to_tsurf converter."""

    def test_vtp_to_tsurf_creates_file(self):
        """vtp_to_tsurf creates a TSurf file."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            vtp_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            ts_path = f.name

        try:
            create_test_vtp(vtp_path)
            vtp_to_tsurf(vtp_path, ts_path)

            assert os.path.exists(ts_path)
            assert os.path.getsize(ts_path) > 0
        finally:
            os.unlink(vtp_path)
            os.unlink(ts_path)

    def test_vtp_to_tsurf_roundtrip(self):
        """Data is preserved through VTP -> TSurf conversion."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            vtp_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            ts_path = f.name

        try:
            create_test_vtp(vtp_path, with_properties=True)

            # Load original VTP
            original = load_vtp(vtp_path)

            # Convert to TSurf
            vtp_to_tsurf(vtp_path, ts_path)

            # Load converted TSurf
            converted = load_tsurf(ts_path)

            # Verify positions
            np.testing.assert_allclose(
                converted.positions, original.positions, rtol=1e-5
            )

            # Verify triangle count
            assert converted.n_triangles == original.n_triangles

            # Verify properties
            assert 'temperature' in converted.properties
            np.testing.assert_allclose(
                converted.properties['temperature'],
                original.properties['temperature'],
                rtol=1e-5
            )
        finally:
            os.unlink(vtp_path)
            os.unlink(ts_path)

    def test_vtp_to_tsurf_with_name(self):
        """vtp_to_tsurf uses provided name."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            vtp_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
            ts_path = f.name

        try:
            create_test_vtp(vtp_path)
            vtp_to_tsurf(vtp_path, ts_path, name='fault_surface')

            converted = load_tsurf(ts_path)
            assert converted.name == 'fault_surface'
        finally:
            os.unlink(vtp_path)
            os.unlink(ts_path)


class TestGetVtpInfo:
    """Tests for get_vtp_info function."""

    def test_get_vtp_info_returns_dict(self):
        """get_vtp_info returns a dictionary with expected keys."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath, with_properties=True)
            info = get_vtp_info(filepath)

            assert isinstance(info, dict)
            assert 'n_points' in info
            assert 'n_cells' in info
            assert 'bounds' in info
            assert 'point_data' in info
            assert 'cell_data' in info
        finally:
            os.unlink(filepath)

    def test_get_vtp_info_correct_values(self):
        """get_vtp_info returns correct values."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath, with_properties=True)
            info = get_vtp_info(filepath)

            assert info['n_points'] == 4
            assert info['n_cells'] == 2
            assert 'temperature' in info['point_data']
            assert 'velocity' in info['point_data']
        finally:
            os.unlink(filepath)


class TestLoadAllVtp:
    """Tests for load_all_vtp function."""

    def test_load_all_vtp_returns_list(self):
        """load_all_vtp returns a list with one surface."""
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            filepath = f.name

        try:
            create_test_vtp(filepath)
            surfaces = load_all_vtp(filepath)

            assert isinstance(surfaces, list)
            assert len(surfaces) == 1
            assert isinstance(surfaces[0], SurfaceData)
        finally:
            os.unlink(filepath)
