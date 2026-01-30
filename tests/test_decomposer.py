"""Tests for Attribute decomposition architecture."""

import numpy as np
import pytest
from xali_tools.core import Attribute, AttributeManager, AttributeType


class TestVector3Decomposition:
    """Tests for Vector3 decomposition."""

    @pytest.fixture
    def velocity_manager(self):
        """Create a manager with velocity data."""
        velocity_data = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
        ])
        velocity = Attribute(velocity_data, item_size=3, name="velocity")
        manager = AttributeManager()
        manager.add("velocity", velocity)
        return manager, velocity_data

    def test_scalar_names_available(self, velocity_manager):
        """Vector3 exposes scalar component names."""
        manager, _ = velocity_manager
        scalars = manager.get_scalar_names()

        assert "velocity:x" in scalars
        assert "velocity:y" in scalars
        assert "velocity:z" in scalars
        assert "velocity:norm" in scalars

    def test_vector_names_available(self, velocity_manager):
        """Vector3 exposes original vector name."""
        manager, _ = velocity_manager
        vectors = manager.get_vector3_names()

        assert "velocity" in vectors

    def test_component_x(self, velocity_manager):
        """Can get x component."""
        manager, data = velocity_manager
        vx = manager.get("velocity:x")

        np.testing.assert_array_almost_equal(vx.as_array(), data[:, 0])

    def test_component_y(self, velocity_manager):
        """Can get y component."""
        manager, data = velocity_manager
        vy = manager.get("velocity:y")

        np.testing.assert_array_almost_equal(vy.as_array(), data[:, 1])

    def test_component_z(self, velocity_manager):
        """Can get z component."""
        manager, data = velocity_manager
        vz = manager.get("velocity:z")

        np.testing.assert_array_almost_equal(vz.as_array(), data[:, 2])

    def test_norm_calculation(self, velocity_manager):
        """Norm is calculated correctly."""
        manager, data = velocity_manager
        vnorm = manager.get("velocity:norm")

        expected_norm = np.linalg.norm(data, axis=1)
        np.testing.assert_array_almost_equal(vnorm.as_array(), expected_norm)


class TestSymTensor3Decomposition:
    """Tests for symmetric 3x3 tensor decomposition."""

    @pytest.fixture
    def stress_manager(self):
        """Create a manager with stress tensor data."""
        # [xx, xy, xz, yy, yz, zz]
        stress_data = np.array([
            [-10.0, 0.0, 0.0, -10.0, 0.0, -10.0],  # Hydrostatic
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # Uniaxial xx
            [50.0, 30.0, 0.0, 50.0, 0.0, 50.0],     # With shear
        ])
        stress = Attribute(stress_data, item_size=6, name="stress")
        manager = AttributeManager()
        manager.add("stress", stress)
        return manager, stress_data

    def test_component_names_available(self, stress_manager):
        """SymTensor3 exposes component names."""
        manager, _ = stress_manager
        scalars = manager.get_scalar_names()

        assert "stress:xx" in scalars
        assert "stress:xy" in scalars
        assert "stress:xz" in scalars
        assert "stress:yy" in scalars
        assert "stress:yz" in scalars
        assert "stress:zz" in scalars

    def test_derived_scalars_available(self, stress_manager):
        """SymTensor3 exposes derived scalar names."""
        manager, _ = stress_manager
        scalars = manager.get_scalar_names()

        assert "stress:trace" in scalars
        assert "stress:von_mises" in scalars

    def test_principal_vector_names_available(self, stress_manager):
        """SymTensor3 exposes principal direction vectors."""
        manager, _ = stress_manager
        vectors = manager.get_vector3_names()

        assert "stress:S1_vec" in vectors
        assert "stress:S2_vec" in vectors
        assert "stress:S3_vec" in vectors

    def test_component_xx(self, stress_manager):
        """Can get xx component."""
        manager, data = stress_manager
        sxx = manager.get("stress:xx")

        np.testing.assert_array_almost_equal(sxx.as_array(), data[:, 0])

    def test_component_xy(self, stress_manager):
        """Can get xy component."""
        manager, data = stress_manager
        sxy = manager.get("stress:xy")

        np.testing.assert_array_almost_equal(sxy.as_array(), data[:, 1])

    def test_trace_calculation(self, stress_manager):
        """Trace is calculated correctly."""
        manager, data = stress_manager
        trace = manager.get("stress:trace")

        expected_trace = data[:, 0] + data[:, 3] + data[:, 5]  # xx + yy + zz
        np.testing.assert_array_almost_equal(trace.as_array(), expected_trace)


class TestPrincipalDecomposition:
    """Tests for principal value/vector decomposition."""

    @pytest.fixture
    def diagonal_stress_manager(self):
        """Create manager with diagonal tensor (principal axes aligned)."""
        # [xx, xy, xz, yy, yz, zz] = [100, 0, 0, 50, 0, 25]
        stress_data = np.array([[100.0, 0.0, 0.0, 50.0, 0.0, 25.0]])
        stress = Attribute(stress_data, item_size=6, name="stress")
        manager = AttributeManager()
        manager.add("stress", stress)
        return manager

    def test_principal_values_available(self, diagonal_stress_manager):
        """Principal values are available."""
        manager = diagonal_stress_manager
        scalars = manager.get_scalar_names()

        assert "stress:S1" in scalars
        assert "stress:S2" in scalars
        assert "stress:S3" in scalars

    def test_principal_values_sorted(self, diagonal_stress_manager):
        """Principal values are sorted by magnitude (S1 >= S2 >= S3)."""
        manager = diagonal_stress_manager

        s1 = manager.get("stress:S1")
        s2 = manager.get("stress:S2")
        s3 = manager.get("stress:S3")

        # For diagonal tensor [100, 50, 25], principal values should be sorted
        np.testing.assert_almost_equal(s1.as_array()[0], 100.0)
        np.testing.assert_almost_equal(s2.as_array()[0], 50.0)
        np.testing.assert_almost_equal(s3.as_array()[0], 25.0)

    def test_principal_values_attribute(self, diagonal_stress_manager):
        """Can get all principal values as single attribute."""
        manager = diagonal_stress_manager
        pvals = manager.get("stress:principal_values")

        assert pvals.item_size == 3
        np.testing.assert_array_almost_equal(
            pvals.as_array()[0], [100.0, 50.0, 25.0]
        )

    def test_principal_vectors_available(self, diagonal_stress_manager):
        """Principal vectors are available."""
        manager = diagonal_stress_manager
        vectors = manager.get_vector3_names()

        assert "stress:S1_vec" in vectors
        assert "stress:S2_vec" in vectors
        assert "stress:S3_vec" in vectors


class TestManagerFeatures:
    """Tests for AttributeManager features."""

    def test_summary(self):
        """Manager summary works."""
        manager = AttributeManager()
        manager.add("pressure", Attribute(np.random.rand(100), item_size=1))
        manager.add("velocity", Attribute(np.random.rand(100, 3), item_size=3))
        manager.add("stress", Attribute(np.random.rand(100, 6), item_size=6))

        summary = manager.summary()

        assert "pressure" in summary
        assert "velocity" in summary
        assert "stress" in summary

    def test_caching(self):
        """Derived attributes are cached."""
        velocity = Attribute(np.random.rand(1000, 3), item_size=3)
        manager = AttributeManager()
        manager.add("velocity", velocity)

        vx1 = manager.get("velocity:x")
        vx2 = manager.get("velocity:x")

        assert vx1 is vx2  # Same object returned

    def test_cache_invalidation(self):
        """Cache is invalidated when base attribute changes."""
        velocity = Attribute(np.random.rand(10, 3), item_size=3)
        manager = AttributeManager()
        manager.add("velocity", velocity)

        # Access to populate cache
        vx1 = manager.get("velocity:x")

        # Replace attribute
        new_velocity = Attribute(np.random.rand(10, 3), item_size=3)
        manager.add("velocity", new_velocity)

        vx2 = manager.get("velocity:x")

        # Should be different object (new computation)
        assert vx1 is not vx2


class TestExplicitAttributeType:
    """Tests for explicit AttributeType specification."""

    def test_default_itemsize3_is_vector(self):
        """itemSize=3 defaults to vector3."""
        data = np.array([[1.0, 0.5, 2.0], [3.0, 1.0, 4.0]])
        vec = Attribute(data, item_size=3, name="displacement")

        assert vec.attribute_type == AttributeType.VECTOR3

    def test_explicit_sym_tensor2(self):
        """Can explicitly set itemSize=3 as sym_tensor2."""
        data = np.array([[1.0, 0.5, 2.0], [3.0, 1.0, 4.0]])
        tensor = Attribute(
            data, item_size=3, name="strain_2d",
            attribute_type=AttributeType.SYM_TENSOR2
        )

        assert tensor.attribute_type == AttributeType.SYM_TENSOR2

    def test_vector3_gets_vector_decomposition(self):
        """Vector3 gets vector decomposition, not tensor."""
        data = np.array([[1.0, 0.5, 2.0], [3.0, 1.0, 4.0]])
        vec = Attribute(data, item_size=3, attribute_type=AttributeType.VECTOR3)

        manager = AttributeManager()
        manager.add("displacement", vec)

        scalars = manager.get_scalar_names()
        assert "displacement:x" in scalars
        assert "displacement:trace" not in scalars

    def test_sym_tensor2_gets_tensor_decomposition(self):
        """SymTensor2 gets tensor decomposition, not vector."""
        data = np.array([[1.0, 0.5, 2.0], [3.0, 1.0, 4.0]])
        tensor = Attribute(
            data, item_size=3,
            attribute_type=AttributeType.SYM_TENSOR2
        )

        manager = AttributeManager()
        manager.add("strain_2d", tensor)

        scalars = manager.get_scalar_names()
        assert "strain_2d:xx" in scalars
        assert "strain_2d:trace" in scalars
        assert "strain_2d:x" not in scalars


class TestVisualizationUseCase:
    """Tests demonstrating visualization property selection use case."""

    @pytest.fixture
    def loaded_manager(self):
        """Create manager with typical loaded properties."""
        n_points = 100
        manager = AttributeManager()
        manager.add("temperature", Attribute(np.random.rand(n_points) * 100, item_size=1))
        manager.add("displacement", Attribute(np.random.rand(n_points, 3), item_size=3))
        manager.add("stress", Attribute(np.random.rand(n_points, 6), item_size=6))
        return manager

    def test_scalar_selection_for_contouring(self, loaded_manager):
        """Can get scalar for iso-contouring."""
        manager = loaded_manager
        scalars = manager.get_scalar_names()

        assert "temperature" in scalars
        assert "stress:von_mises" in scalars

        vm = manager.get("stress:von_mises")
        assert vm.item_size == 1

    def test_vector_selection_for_glyphs(self, loaded_manager):
        """Can get vectors for glyph display."""
        manager = loaded_manager
        vectors = manager.get_vector3_names()

        assert "displacement" in vectors

        disp = manager.get("displacement")
        assert disp.item_size == 3

    def test_principal_vectors_for_streamlines(self, loaded_manager):
        """Can get principal vectors for streamlines."""
        manager = loaded_manager
        vectors = manager.get_vector3_names()

        assert "stress:S1_vec" in vectors

        s1_vec = manager.get("stress:S1_vec")
        assert s1_vec.item_size == 3
