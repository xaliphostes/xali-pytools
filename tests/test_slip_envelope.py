"""Tests for slip envelope analysis module."""

import numpy as np
import pytest
from xali_tools.slip_envelope import (
    # Traction
    compute_triangle_normals,
    resolve_stress,
    stress_6_to_matrix,
    # Coulomb
    CoulombParameters,
    evaluate_coulomb,
    coulomb_slip_tendency,
    # Slip analysis
    SlipAnalyzer,
    SlipResult,
    # Parameters
    ParameterType,
    ParameterRange,
    DomainSpecification,
    friction_pore_pressure_domain,
    friction_cohesion_pore_pressure_domain,
    # Domain sweep
    OutputMetric,
    DomainResult,
    DomainSweepAnalyzer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_mesh():
    """Two triangles forming a square in the XY plane."""
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=np.float64)

    indices = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint32)

    return positions, indices


@pytest.fixture
def oriented_mesh():
    """Triangles at various orientations (XY, XZ, YZ planes)."""
    positions = np.array([
        # XY plane triangle (normal = +Z)
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        # XZ plane triangle (normal = +Y)
        [0, 0, 0], [1, 0, 0], [0, 0, 1],
        # YZ plane triangle (normal = +X)
        [0, 0, 0], [0, 1, 0], [0, 0, 1],
    ], dtype=np.float64)

    indices = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ], dtype=np.uint32)

    return positions, indices


@pytest.fixture
def horizontal_plane():
    """Single horizontal triangle (normal = +Z)."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    return positions, indices


# =============================================================================
# Traction computation tests
# =============================================================================

class TestTriangleNormals:
    """Tests for triangle normal computation."""

    def test_xy_plane_normals(self, simple_mesh):
        """Triangles in XY plane have +Z normals."""
        positions, indices = simple_mesh
        normals = compute_triangle_normals(positions, indices)

        assert normals.shape == (2, 3)
        for i in range(2):
            np.testing.assert_array_almost_equal(normals[i], [0, 0, 1], decimal=5)

    def test_oriented_mesh_normals(self, oriented_mesh):
        """Triangles in different planes have correct normals."""
        positions, indices = oriented_mesh
        normals = compute_triangle_normals(positions, indices)

        assert normals.shape == (3, 3)
        # XY plane -> +Z normal
        np.testing.assert_array_almost_equal(normals[0], [0, 0, 1], decimal=5)


class TestStressTensorConversion:
    """Tests for stress tensor conversion."""

    def test_stress_6_to_matrix(self):
        """Converts 6-component stress to 3x3 matrix."""
        stress = np.array([1, 2, 3, 4, 5, 6])  # xx, xy, xz, yy, yz, zz
        matrix = stress_6_to_matrix(stress)

        expected = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ])
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_stress_matrix_symmetric(self):
        """Stress matrix is symmetric."""
        stress = np.array([1, 2, 3, 4, 5, 6])
        matrix = stress_6_to_matrix(stress)

        np.testing.assert_array_almost_equal(matrix, matrix.T)


class TestStressResolution:
    """Tests for stress resolution into components."""

    def test_pure_compression(self, horizontal_plane):
        """Pure vertical compression gives negative normal stress, no shear."""
        positions, indices = horizontal_plane
        normals = compute_triangle_normals(positions, indices)

        stress = np.array([0, 0, 0, 0, 0, -10])  # Szz = -10
        sigma_n, tau = resolve_stress(stress, normals)

        np.testing.assert_almost_equal(sigma_n[0], -10, decimal=5)
        np.testing.assert_almost_equal(tau[0], 0, decimal=5)

    def test_pure_shear(self, horizontal_plane):
        """Pure shear stress gives no normal stress, positive shear."""
        positions, indices = horizontal_plane
        normals = compute_triangle_normals(positions, indices)

        stress = np.array([0, 0, 5, 0, 0, 0])  # Sxz = 5
        sigma_n, tau = resolve_stress(stress, normals)

        np.testing.assert_almost_equal(sigma_n[0], 0, decimal=5)
        np.testing.assert_almost_equal(tau[0], 5, decimal=5)


# =============================================================================
# Coulomb criterion tests
# =============================================================================

class TestCoulombParameters:
    """Tests for CoulombParameters dataclass."""

    def test_basic_properties(self):
        """Parameters store correctly."""
        params = CoulombParameters(friction=0.6, cohesion=5.0, pore_pressure=2.0)

        assert params.friction == 0.6
        assert params.cohesion == 5.0
        assert params.pore_pressure == 2.0

    def test_effective_normal_stress(self):
        """Effective stress accounts for pore pressure."""
        params = CoulombParameters(friction=0.6, cohesion=5.0, pore_pressure=2.0)
        sigma_n = np.array([-10.0])

        sigma_n_eff = params.effective_normal_stress(sigma_n)

        # sigma'_n = sigma_n - p = -10 - 2 = -12
        np.testing.assert_almost_equal(sigma_n_eff[0], -12.0, decimal=5)

    def test_slip_threshold(self):
        """Slip threshold calculation."""
        params = CoulombParameters(friction=0.6, cohesion=5.0, pore_pressure=2.0)
        sigma_n = np.array([-10.0])

        threshold = params.slip_threshold(sigma_n)

        # threshold = c - mu * sigma'_n = 5 - 0.6 * (-12) = 5 + 7.2 = 12.2
        np.testing.assert_almost_equal(threshold[0], 12.2, decimal=5)


class TestCoulombEvaluation:
    """Tests for Coulomb criterion evaluation."""

    def test_no_slip_below_threshold(self):
        """No slip when shear is below threshold."""
        params = CoulombParameters(friction=0.6, cohesion=0.0, pore_pressure=0.0)
        sigma_n = np.array([-10.0])
        tau = np.array([5.0])  # threshold = 0 - 0.6 * (-10) = 6

        slip = evaluate_coulomb(sigma_n, tau, params)

        assert not slip[0]

    def test_slip_above_threshold(self):
        """Slip when shear exceeds threshold."""
        params = CoulombParameters(friction=0.6, cohesion=0.0, pore_pressure=0.0)
        sigma_n = np.array([-10.0])
        tau = np.array([7.0])  # threshold = 6

        slip = evaluate_coulomb(sigma_n, tau, params)

        assert slip[0]


class TestSlipTendency:
    """Tests for slip tendency calculation."""

    def test_slip_tendency_calculation(self):
        """Slip tendency is tau / threshold."""
        params = CoulombParameters(friction=0.6, cohesion=0.0, pore_pressure=0.0)
        sigma_n = np.array([-10.0])
        tau = np.array([3.0])  # threshold = 6

        tendency = coulomb_slip_tendency(sigma_n, tau, params)

        expected = 3.0 / 6.0  # 0.5
        np.testing.assert_almost_equal(tendency[0], expected, decimal=5)


# =============================================================================
# Slip analysis tests
# =============================================================================

class TestSlipAnalyzer:
    """Tests for SlipAnalyzer class."""

    def test_initialization(self, simple_mesh):
        """Analyzer initializes correctly."""
        positions, indices = simple_mesh
        analyzer = SlipAnalyzer(positions, indices)

        assert analyzer.n_triangles == 2
        assert analyzer.normals.shape == (2, 3)

    def test_analyze_returns_result(self, simple_mesh):
        """Analyze returns SlipResult."""
        positions, indices = simple_mesh
        analyzer = SlipAnalyzer(positions, indices)

        stress = np.array([0, 0, 5, 0, 0, -10])
        result = analyzer.analyze(stress, friction=0.6, cohesion=0.0, pore_pressure=0.0)

        assert isinstance(result, SlipResult)
        assert result.n_triangles == 2
        assert len(result.sigma_n) == 2
        assert len(result.tau) == 2


class TestSlipResult:
    """Tests for SlipResult properties."""

    def test_computed_properties(self):
        """SlipResult computes derived properties correctly."""
        result = SlipResult(
            sigma_n=np.array([-10, -5, -8, -12]),
            tau=np.array([5, 8, 3, 10]),
            slip_mask=np.array([False, True, False, True]),
            slip_tendency=np.array([0.5, 1.2, 0.3, 1.5]),
            params=CoulombParameters(),
        )

        assert result.n_triangles == 4
        assert result.slip_count == 2
        assert result.slip_ratio == 0.5
        assert result.binary_slip == 1
        assert result.max_slip_tendency == 1.5

    def test_to_attributes(self):
        """Can convert to attributes dict."""
        result = SlipResult(
            sigma_n=np.array([-10, -5]),
            tau=np.array([5, 8]),
            slip_mask=np.array([False, True]),
            slip_tendency=np.array([0.5, 1.2]),
            params=CoulombParameters(),
        )

        attrs = result.to_attributes()

        assert "sigma_n" in attrs
        assert "slip_tendency" in attrs


# =============================================================================
# Parameter domain tests
# =============================================================================

class TestParameterRange:
    """Tests for ParameterRange class."""

    def test_range_values(self):
        """Range generates correct values."""
        pr = ParameterRange(ParameterType.FRICTION, 0.3, 0.9, 7)

        assert pr.n_steps == 7
        assert len(pr.values) == 7
        np.testing.assert_almost_equal(pr.values[0], 0.3)
        np.testing.assert_almost_equal(pr.values[-1], 0.9)


class TestDomainSpecification:
    """Tests for DomainSpecification class."""

    def test_2d_domain(self):
        """2D domain has correct shape."""
        domain = friction_pore_pressure_domain(
            friction_range=(0.3, 0.6),
            friction_steps=4,
            pore_pressure_range=(0, 10),
            pore_pressure_steps=3,
        )

        assert domain.n_dimensions == 2
        assert domain.shape == (4, 3)
        assert domain.n_points == 12

    def test_domain_iteration(self):
        """Can iterate over domain points."""
        domain = friction_pore_pressure_domain(
            friction_range=(0.3, 0.6),
            friction_steps=4,
            pore_pressure_range=(0, 10),
            pore_pressure_steps=3,
        )

        points = list(domain.iter_grid())

        assert len(points) == 12
        assert "friction" in points[0]
        assert "pore_pressure" in points[0]
        assert "cohesion" in points[0]  # Fixed value

    def test_3d_domain(self):
        """3D domain has correct shape."""
        domain = friction_cohesion_pore_pressure_domain(
            friction_steps=3,
            cohesion_steps=4,
            pore_pressure_steps=5,
        )

        assert domain.n_dimensions == 3
        assert domain.shape == (3, 4, 5)
        assert domain.n_points == 60


# =============================================================================
# Domain sweep tests
# =============================================================================

class TestDomainSweepAnalyzer:
    """Tests for DomainSweepAnalyzer class."""

    def test_sweep_returns_result(self, simple_mesh):
        """Sweep returns DomainResult."""
        positions, indices = simple_mesh
        analyzer = DomainSweepAnalyzer(positions, indices)

        domain = friction_pore_pressure_domain(
            friction_range=(0.3, 0.9),
            friction_steps=5,
            pore_pressure_range=(0, 10),
            pore_pressure_steps=5,
        )
        stress = np.array([0, 0, 5, 0, 0, -15])

        result = analyzer.sweep(domain, stress, OutputMetric.SLIP_RATIO)

        assert isinstance(result, DomainResult)
        assert result.shape == (5, 5)
        assert result.n_dimensions == 2


class TestDomainResult:
    """Tests for DomainResult class."""

    def test_contour_data_extraction(self, simple_mesh):
        """Can extract 2D contour data."""
        positions, indices = simple_mesh
        analyzer = DomainSweepAnalyzer(positions, indices)

        domain = friction_pore_pressure_domain(
            friction_range=(0.3, 0.9),
            friction_steps=4,
            pore_pressure_range=(0, 10),
            pore_pressure_steps=3,
        )
        stress = np.array([0, 0, 5, 0, 0, -15])

        result = analyzer.sweep(domain, stress, OutputMetric.SLIP_RATIO)
        X, Y, Z = result.get_2d_contour_data()

        assert X.shape == (4, 3)
        assert Y.shape == (4, 3)
        assert Z.shape == (4, 3)

    def test_3d_slicing(self, simple_mesh):
        """Can slice 3D result to 2D."""
        positions, indices = simple_mesh
        analyzer = DomainSweepAnalyzer(positions, indices)

        domain = friction_cohesion_pore_pressure_domain(
            friction_steps=4,
            cohesion_steps=3,
            pore_pressure_steps=5,
        )
        stress = np.array([0, 0, 5, 0, 0, -15])

        result = analyzer.sweep(domain, stress, OutputMetric.SLIP_RATIO)

        assert result.shape == (4, 3, 5)

        slice_2d = result.get_slice(axis=2, index=2)

        assert slice_2d.shape == (4, 3)
        assert slice_2d.n_dimensions == 2

        X, Y, Z = slice_2d.get_2d_contour_data()
        assert X.shape == (4, 3)


class TestOutputMetrics:
    """Tests for different output metrics."""

    def test_all_metrics_work(self, simple_mesh):
        """All output metrics produce valid results."""
        positions, indices = simple_mesh
        analyzer = DomainSweepAnalyzer(positions, indices)

        domain = friction_pore_pressure_domain(
            friction_range=(0.3, 0.9),
            friction_steps=3,
            pore_pressure_range=(0, 10),
            pore_pressure_steps=3,
        )
        stress = np.array([0, 0, 5, 0, 0, -15])

        for metric in OutputMetric:
            result = analyzer.sweep(domain, stress, metric)
            assert result.shape == (3, 3)
            assert not np.isnan(result.values).any()


# =============================================================================
# Integration tests
# =============================================================================

class TestFullWorkflow:
    """Integration tests for complete workflow."""

    def test_single_analysis(self, simple_mesh):
        """Complete single analysis workflow."""
        positions, indices = simple_mesh
        analyzer = SlipAnalyzer(positions, indices)

        result = analyzer.analyze(
            stress=[0, 0, 5, 0, 0, -15],
            friction=0.6,
            cohesion=0.0,
            pore_pressure=5.0
        )

        assert 0 <= result.slip_ratio <= 1

    def test_domain_sweep_workflow(self, simple_mesh):
        """Complete domain sweep workflow."""
        positions, indices = simple_mesh
        analyzer = DomainSweepAnalyzer(positions, indices)

        domain = friction_pore_pressure_domain(
            friction_range=(0.2, 0.8),
            friction_steps=10,
            pore_pressure_range=(0, 15),
            pore_pressure_steps=10,
        )

        envelope = analyzer.sweep(
            domain=domain,
            stress=[0, 0, 5, 0, 0, -15],
            metric=OutputMetric.SLIP_RATIO
        )

        assert envelope.shape == (10, 10)
        assert np.min(envelope.values) >= 0
        assert np.max(envelope.values) <= 1

        # Can get contour data for plotting
        X, Y, Z = envelope.get_2d_contour_data()
        assert X.shape == (10, 10)
