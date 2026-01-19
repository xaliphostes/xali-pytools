"""
Test Serie and SurfaceData property handling.
"""
import numpy as np
from xali_tools.core import Serie
from xali_tools.io import SurfaceData
from xali_tools.io.tsurf_filter import load_tsurf, save_tsurf
import tempfile
import os


def test_scalar_property():
    """Test scalar property (item_size=1)."""
    print("=== Test scalar property ===")

    pressure = Serie([1.0, 2.0, 3.0, 4.0], item_size=1, name='pressure')

    assert pressure.item_size == 1
    assert pressure.n_items == 4
    assert pressure.is_scalar == True
    assert pressure.is_vector == False
    assert pressure.shape == (4,)

    # Iteration
    values = list(pressure)
    assert values == [1.0, 2.0, 3.0, 4.0]

    # Indexing
    assert pressure[0] == 1.0
    assert pressure[2] == 3.0

    # Flatten
    np.testing.assert_array_equal(pressure.flatten(), [1.0, 2.0, 3.0, 4.0])

    print(f"  {repr(pressure)}")
    print("  PASSED")


def test_vector_property():
    """Test 3D vector property (item_size=3)."""
    print("\n=== Test vector property ===")

    velocity = Serie([[1, 2, 3], [4, 5, 6], [7, 8, 9]], name='velocity')

    assert velocity.item_size == 3
    assert velocity.n_items == 3
    assert velocity.is_scalar == False
    assert velocity.is_vector == True
    assert velocity.shape == (3, 3)

    # Iteration
    items = list(velocity)
    assert len(items) == 3
    np.testing.assert_array_equal(items[0], [1, 2, 3])
    np.testing.assert_array_equal(items[1], [4, 5, 6])

    # Indexing
    np.testing.assert_array_equal(velocity[0], [1, 2, 3])
    np.testing.assert_array_equal(velocity[1:3], [[4, 5, 6], [7, 8, 9]])

    # Flatten
    np.testing.assert_array_equal(velocity.flatten(), [1, 2, 3, 4, 5, 6, 7, 8, 9])

    print(f"  {repr(velocity)}")
    print("  PASSED")


def test_tensor_property():
    """Test symmetric tensor property (item_size=6)."""
    print("\n=== Test tensor property ===")

    # 2 items, each with 6 components (xx, xy, xz, yy, yz, zz)
    stress_data = np.array([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12]
    ], dtype=np.float64)

    stress = Serie(stress_data, name='stress')

    assert stress.item_size == 6
    assert stress.n_items == 2
    assert stress.is_tensor == True
    assert stress.shape == (2, 6)

    # Iteration
    items = list(stress)
    np.testing.assert_array_equal(items[0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(items[1], [7, 8, 9, 10, 11, 12])

    print(f"  {repr(stress)}")
    print("  PASSED")


def test_from_flat():
    """Test creating Serie from flat array."""
    print("\n=== Test from_flat ===")

    flat_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    vec = Serie.from_flat(flat_data, item_size=3, name='vec')

    assert vec.n_items == 3
    assert vec.item_size == 3
    np.testing.assert_array_equal(vec[0], [1, 2, 3])
    np.testing.assert_array_equal(vec[1], [4, 5, 6])
    np.testing.assert_array_equal(vec[2], [7, 8, 9])

    print(f"  {repr(vec)}")
    print("  PASSED")


def test_as_view():
    """Test that as_view creates a true view (no copy)."""
    print("\n=== Test as_view (no copy) ===")

    original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    view = Serie.as_view(original, item_size=3, name='velocity')

    # Should share memory
    assert np.shares_memory(original, view._data), "Should share memory"

    # Modify via view
    view[0] = [10, 20, 30]
    np.testing.assert_array_equal(original[0], [10, 20, 30])

    # Modify original
    original[1] = [40, 50, 60]
    np.testing.assert_array_equal(view[1], [40, 50, 60])

    print("  View shares memory with original: True")
    print("  Modifications propagate both ways: True")
    print("  PASSED")


def test_copy():
    """Test that copy creates an independent copy."""
    print("\n=== Test copy ===")

    original = Serie([[1, 2, 3], [4, 5, 6]], name='velocity')
    copied = original.copy()

    # Should not share memory
    assert not np.shares_memory(original._data, copied._data), "Should not share memory"

    # Modify copy
    copied[0] = [10, 20, 30]
    np.testing.assert_array_equal(original[0], [1, 2, 3])  # Original unchanged

    print("  Copy is independent: True")
    print("  PASSED")


def test_setitem():
    """Test setting values."""
    print("\n=== Test __setitem__ ===")

    vec = Serie([[1, 2, 3], [4, 5, 6]], name='vec')

    vec[0] = [10, 20, 30]
    np.testing.assert_array_equal(vec[0], [10, 20, 30])

    scalar = Serie([1.0, 2.0, 3.0], item_size=1)
    scalar[1] = 99.0
    assert scalar[1] == 99.0

    print("  PASSED")


def test_surface_data_get_property():
    """Test SurfaceData.get_property() returns a view."""
    print("\n=== Test SurfaceData.get_property ===")

    # Create SurfaceData with properties
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

    # Get property as view
    U = surface.get_property('U')
    assert U.is_vector
    assert U.n_items == 3

    a = surface.get_property('a')
    assert a.is_scalar
    assert a.n_items == 3

    # Check it's a view
    assert np.shares_memory(surface.properties['U'], U._data)

    print(f"  U: {repr(U)}")
    print(f"  a: {repr(a)}")
    print("  Properties are views: True")
    print("  PASSED")


def test_surface_data_set_property():
    """Test SurfaceData.set_property()."""
    print("\n=== Test SurfaceData.set_property ===")

    positions = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
    surface = SurfaceData(positions=positions, name='test')

    # Set scalar property
    surface.set_property('pressure', [1.0, 2.0, 3.0], item_size=1)
    assert 'pressure' in surface.properties
    assert surface.property_sizes['pressure'] == 1

    # Set vector property
    surface.set_property('velocity', [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert 'velocity' in surface.properties
    assert surface.property_sizes['velocity'] == 3

    # Get back as Serie
    vel = surface.get_property('velocity')
    assert vel.is_vector

    print("  PASSED")


def test_tsurf_with_esizes():
    """Test loading TSurf file with ESIZES populates property_sizes."""
    print("\n=== Test TSurf with ESIZES ===")

    data = load_tsurf('tests/two-triangles.gcd')

    assert 'U' in data.properties
    assert 'a' in data.properties
    assert 'b' in data.properties

    assert data.property_sizes['U'] == 3
    assert data.property_sizes['a'] == 1
    assert data.property_sizes['b'] == 1

    # Get as Serie
    U = data.get_property('U')
    assert U.is_vector
    assert U.n_items == 4

    a = data.get_property('a')
    assert a.is_scalar

    print(f"  Properties: {list(data.properties.keys())}")
    print(f"  Property sizes: {data.property_sizes}")
    print(f"  U: {repr(U)}")
    print("  PASSED")


def test_tsurf_roundtrip():
    """Test save and reload preserves property_sizes."""
    print("\n=== Test TSurf roundtrip ===")

    # Load original
    original = load_tsurf('tests/two-triangles.gcd')

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as f:
        filepath = f.name

    try:
        save_tsurf(original, filepath)

        # Reload
        reloaded = load_tsurf(filepath)

        # Check property_sizes preserved
        assert reloaded.property_sizes == original.property_sizes

        # Check data preserved
        np.testing.assert_array_almost_equal(
            reloaded.properties['U'], original.properties['U']
        )
        np.testing.assert_array_almost_equal(
            reloaded.properties['a'], original.properties['a']
        )

        print(f"  Property sizes preserved: {reloaded.property_sizes}")
        print("  Data preserved: True")
        print("  PASSED")

    finally:
        os.unlink(filepath)


def test_iteration_examples():
    """Example of iterating over properties."""
    print("\n=== Iteration examples ===")

    data = load_tsurf('tests/two-triangles.gcd')

    print("  Iterating over vector U:")
    U = data.get_property('U')
    for i, vec in enumerate(U):
        print(f"    vertex {i}: [{vec[0]:.1f}, {vec[1]:.1f}, {vec[2]:.1f}]")

    print("  Iterating over scalar a:")
    a = data.get_property('a')
    for i, val in enumerate(a):
        print(f"    vertex {i}: {val:.1f}")

    print("  PASSED")


if __name__ == "__main__":
    test_scalar_property()
    test_vector_property()
    test_tensor_property()
    test_from_flat()
    test_as_view()
    test_copy()
    test_setitem()
    test_surface_data_get_property()
    test_surface_data_set_property()
    test_tsurf_with_esizes()
    test_tsurf_roundtrip()
    test_iteration_examples()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
