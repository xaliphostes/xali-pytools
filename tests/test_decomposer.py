"""
Test for Serie decomposition architecture.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from xali_tools.core import Serie, SerieContainer, SerieType


def test_vector3_decomposition():
    """Test decomposition of a 3D vector into components."""
    print("=" * 60)
    print("Test: Vector3 Decomposition")
    print("=" * 60)

    # Create velocity field
    velocity_data = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
    ])
    velocity = Serie(velocity_data, item_size=3, name="velocity")

    container = SerieContainer()
    container.add("velocity", velocity)

    # Query available scalars
    scalars = container.get_scalar_names()
    print(f"\nAvailable scalars: {scalars}")

    # Query available vectors
    vectors = container.get_vector3_names()
    print(f"Available vector3: {vectors}")

    # Get derived properties
    vx = container.get("velocity:x")
    vy = container.get("velocity:y")
    vz = container.get("velocity:z")
    vnorm = container.get("velocity:norm")

    print(f"\nvelocity:x = {vx.as_array()}")
    print(f"velocity:y = {vy.as_array()}")
    print(f"velocity:z = {vz.as_array()}")
    print(f"velocity:norm = {vnorm.as_array()}")

    # Verify norm calculation
    expected_norm = np.linalg.norm(velocity_data, axis=1)
    assert np.allclose(vnorm.as_array(), expected_norm), "Norm calculation failed"
    print("\n[OK] Vector3 decomposition works correctly!")


def test_sym_tensor_decomposition():
    """Test decomposition of symmetric 3x3 tensor into components."""
    print("\n" + "=" * 60)
    print("Test: Symmetric Tensor Decomposition")
    print("=" * 60)

    # Create stress tensor [xx, xy, xz, yy, yz, zz]
    # Hydrostatic pressure: diagonal = -p, off-diagonal = 0
    stress_data = np.array([
        [-10.0, 0.0, 0.0, -10.0, 0.0, -10.0],  # Hydrostatic
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Uniaxial xx
        [50.0, 30.0, 0.0, 50.0, 0.0, 50.0],    # With shear
    ])
    stress = Serie(stress_data, item_size=6, name="stress")

    container = SerieContainer()
    container.add("stress", stress)

    # Query available properties
    scalars = container.get_scalar_names()
    print(f"\nAvailable scalars: {scalars}")

    vectors = container.get_vector3_names()
    print(f"Available vector3: {vectors}")

    # Get individual components
    sxx = container.get("stress:xx")
    sxy = container.get("stress:xy")
    trace = container.get("stress:trace")
    vm = container.get("stress:von_mises")

    print(f"\nstress:xx = {sxx.as_array()}")
    print(f"stress:xy = {sxy.as_array()}")
    print(f"stress:trace = {trace.as_array()}")
    print(f"stress:von_mises = {vm.as_array()}")

    # Verify trace
    expected_trace = stress_data[:, 0] + stress_data[:, 3] + stress_data[:, 5]
    assert np.allclose(trace.as_array(), expected_trace), "Trace calculation failed"
    print("\n[OK] SymTensor3 decomposition works correctly!")


def test_principal_decomposition():
    """Test principal value/vector decomposition of symmetric tensor."""
    print("\n" + "=" * 60)
    print("Test: Principal Decomposition")
    print("=" * 60)

    # Create a simple diagonal tensor (principal axes aligned with coordinates)
    # [xx, xy, xz, yy, yz, zz] = [100, 0, 0, 50, 0, 25]
    stress_data = np.array([
        [100.0, 0.0, 0.0, 50.0, 0.0, 25.0],
    ])
    stress = Serie(stress_data, item_size=6, name="stress")

    container = SerieContainer()
    container.add("stress", stress)

    # Get principal values
    s1 = container.get("stress:S1")
    s2 = container.get("stress:S2")
    s3 = container.get("stress:S3")
    pvals = container.get("stress:principal_values")

    print(f"\nstress:S1 = {s1.as_array()}")
    print(f"stress:S2 = {s2.as_array()}")
    print(f"stress:S3 = {s3.as_array()}")
    print(f"stress:principal_values = {pvals.as_array()}")

    # Get principal vectors
    s1_vec = container.get("stress:S1_vec")
    s2_vec = container.get("stress:S2_vec")
    s3_vec = container.get("stress:S3_vec")

    print(f"\nstress:S1_vec = {s1_vec.as_array()}")
    print(f"stress:S2_vec = {s2_vec.as_array()}")
    print(f"stress:S3_vec = {s3_vec.as_array()}")

    # For diagonal tensor, principal values should be the diagonal elements
    # sorted by magnitude: 100, 50, 25
    assert np.isclose(s1.as_array()[0], 100.0), f"S1 should be 100, got {s1.as_array()[0]}"
    assert np.isclose(s2.as_array()[0], 50.0), f"S2 should be 50, got {s2.as_array()[0]}"
    assert np.isclose(s3.as_array()[0], 25.0), f"S3 should be 25, got {s3.as_array()[0]}"
    print("\n[OK] Principal decomposition works correctly!")


def test_container_summary():
    """Test the container summary functionality."""
    print("\n" + "=" * 60)
    print("Test: Container Summary")
    print("=" * 60)

    container = SerieContainer()
    container.add("pressure", Serie(np.random.rand(100), item_size=1))
    container.add("velocity", Serie(np.random.rand(100, 3), item_size=3))
    container.add("stress", Serie(np.random.rand(100, 6), item_size=6))

    print(container.summary())
    print("\n[OK] Container summary works!")


def test_caching():
    """Test that derived series are cached."""
    print("\n" + "=" * 60)
    print("Test: Caching")
    print("=" * 60)

    velocity = Serie(np.random.rand(1000, 3), item_size=3)
    container = SerieContainer()
    container.add("velocity", velocity)

    # First access - computes
    vx1 = container.get("velocity:x")
    # Second access - should be cached
    vx2 = container.get("velocity:x")

    assert vx1 is vx2, "Caching not working - should return same object"
    print("First and second access return same cached object")
    print("[OK] Caching works correctly!")


def test_explicit_serie_type():
    """Test explicit serie_type specification for ambiguous itemSize."""
    print("\n" + "=" * 60)
    print("Test: Explicit SerieType for Ambiguous itemSize=3")
    print("=" * 60)

    # itemSize=3 could be vector3 or sym_tensor2
    data = np.array([[1.0, 0.5, 2.0], [3.0, 1.0, 4.0]])

    # Default: itemSize=3 is treated as vector3
    vec = Serie(data, item_size=3, name="displacement")
    print(f"\nDefault (itemSize=3): serie_type = {vec.serie_type}")
    assert vec.serie_type == SerieType.VECTOR3

    # Explicit: itemSize=3 as sym_tensor2 (2D symmetric tensor [xx, xy, yy])
    tensor = Serie(data, item_size=3, name="strain_2d", serie_type=SerieType.SYM_TENSOR2)
    print(f"Explicit: serie_type = {tensor.serie_type}")
    assert tensor.serie_type == SerieType.SYM_TENSOR2

    # Create container with both
    container = SerieContainer()
    container.add("displacement", vec)
    container.add("strain_2d", tensor)

    # Vector3 gets vector decomposition
    vec_scalars = [n for n in container.get_scalar_names() if n.startswith("displacement")]
    print(f"\nVector3 'displacement' scalars: {vec_scalars}")
    assert "displacement:x" in vec_scalars
    assert "displacement:trace" not in vec_scalars  # NOT tensor decomposition

    # SymTensor2 gets tensor decomposition
    tensor_scalars = [n for n in container.get_scalar_names() if n.startswith("strain_2d")]
    print(f"SymTensor2 'strain_2d' scalars: {tensor_scalars}")
    assert "strain_2d:xx" in tensor_scalars
    assert "strain_2d:trace" in tensor_scalars
    assert "strain_2d:x" not in tensor_scalars  # NOT vector decomposition

    print("\n[OK] Explicit SerieType works correctly!")


def test_use_case_visualization():
    """Demonstrate the main use case: visualization property selection."""
    print("\n" + "=" * 60)
    print("Use Case: Visualization Property Selection")
    print("=" * 60)

    # Simulate loading a surface with multiple properties
    n_points = 1000
    container = SerieContainer()
    container.add("temperature", Serie(np.random.rand(n_points) * 100, item_size=1))
    container.add("displacement", Serie(np.random.rand(n_points, 3), item_size=3))
    container.add("stress", Serie(np.random.rand(n_points, 6), item_size=6))

    print("\nLoaded properties:")
    for name in container.names:
        serie = container.get(name)
        print(f"  - {name}: itemSize={serie.item_size}, n={serie.n_items}")

    print("\n--- For Iso-Contouring (need scalars) ---")
    scalars = container.get_scalar_names()
    print(f"Available: {scalars}")
    print(f"User selects: 'stress:von_mises'")
    vm = container.get("stress:von_mises")
    print(f"  → Serie with {vm.n_items} values, range: [{vm.as_array().min():.3f}, {vm.as_array().max():.3f}]")

    print("\n--- For Vector Field Display (need vector3) ---")
    vectors = container.get_vector3_names()
    print(f"Available: {vectors}")
    print(f"User selects: 'displacement'")
    disp = container.get("displacement")
    print(f"  → Serie with {disp.n_items} vectors")

    print("\n--- For Streamlines (need vector3) ---")
    print(f"User selects: 'stress:S1_vec' (max principal direction)")
    s1_vec = container.get("stress:S1_vec")
    print(f"  → Serie with {s1_vec.n_items} direction vectors")

    print("\n[OK] Visualization use case demonstrated!")


if __name__ == "__main__":
    test_vector3_decomposition()
    test_sym_tensor_decomposition()
    test_principal_decomposition()
    test_container_summary()
    test_caching()
    test_explicit_serie_type()
    test_use_case_visualization()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
