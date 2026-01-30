"""Tests for weighted sum functions."""

import numpy as np
import pytest
from xali_tools.core import (
    Attribute,
    AttributeManager,
    AttributeType,
    weighted_sum,
    normalized_weighted_sum,
    weighted_sum_from_manager,
)


class TestWeightedSum:
    """Tests for weighted_sum function."""

    def test_scalar_weighted_sum(self):
        """Weighted sum of scalar attributes."""
        a1 = Attribute([1, 2, 3], item_size=1)
        a2 = Attribute([4, 5, 6], item_size=1)

        result = weighted_sum([a1, a2], [0.3, 0.7])

        expected = 0.3 * np.array([1, 2, 3]) + 0.7 * np.array([4, 5, 6])
        np.testing.assert_array_almost_equal(result.as_array(), expected)
        assert result.item_size == 1

    def test_vector3_weighted_sum(self):
        """Weighted sum of 3D vector attributes."""
        v1 = Attribute([[1, 0, 0], [0, 1, 0]], item_size=3)
        v2 = Attribute([[0, 1, 0], [1, 0, 0]], item_size=3)

        result = weighted_sum([v1, v2], [0.5, 0.5])

        expected = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0]])
        np.testing.assert_array_almost_equal(result.as_array(), expected)
        assert result.item_size == 3

    def test_tensor_weighted_sum(self):
        """Weighted sum of symmetric tensor attributes."""
        t1 = Attribute(np.ones((5, 6)), item_size=6)
        t2 = Attribute(np.ones((5, 6)) * 2, item_size=6)

        result = weighted_sum([t1, t2], [0.4, 0.6])

        expected = np.ones((5, 6)) * 1.6
        np.testing.assert_array_almost_equal(result.as_array(), expected)
        assert result.item_size == 6

    def test_three_attributes(self):
        """Weighted sum of three attributes."""
        a1 = Attribute([1, 1, 1], item_size=1)
        a2 = Attribute([2, 2, 2], item_size=1)
        a3 = Attribute([3, 3, 3], item_size=1)

        result = weighted_sum([a1, a2, a3], [0.2, 0.3, 0.5])

        expected = 0.2 * 1 + 0.3 * 2 + 0.5 * 3  # = 2.3
        np.testing.assert_array_almost_equal(result.as_array(), [2.3, 2.3, 2.3])

    def test_single_attribute(self):
        """Weighted sum of single attribute (just scales it)."""
        a = Attribute([1, 2, 3], item_size=1)

        result = weighted_sum([a], [2.0])

        np.testing.assert_array_almost_equal(result.as_array(), [2, 4, 6])

    def test_preserves_attribute_type(self):
        """Weighted sum preserves attribute_type from first attribute."""
        v1 = Attribute([[1, 0, 0]], item_size=3, attribute_type=AttributeType.VECTOR3)
        v2 = Attribute([[0, 1, 0]], item_size=3, attribute_type=AttributeType.VECTOR3)

        result = weighted_sum([v1, v2], [0.5, 0.5])

        assert result.attribute_type == AttributeType.VECTOR3

    def test_result_name(self):
        """Weighted sum can set result name."""
        a1 = Attribute([1, 2], item_size=1)
        a2 = Attribute([3, 4], item_size=1)

        result = weighted_sum([a1, a2], [0.5, 0.5], name="combined")

        assert result.name == "combined"

    def test_zero_weights(self):
        """Weighted sum with zero weight ignores that attribute."""
        a1 = Attribute([1, 2, 3], item_size=1)
        a2 = Attribute([100, 200, 300], item_size=1)

        result = weighted_sum([a1, a2], [1.0, 0.0])

        np.testing.assert_array_almost_equal(result.as_array(), [1, 2, 3])

    def test_negative_weights(self):
        """Weighted sum with negative weights (subtraction)."""
        a1 = Attribute([10, 20, 30], item_size=1)
        a2 = Attribute([1, 2, 3], item_size=1)

        result = weighted_sum([a1, a2], [1.0, -1.0])

        np.testing.assert_array_almost_equal(result.as_array(), [9, 18, 27])


class TestWeightedSumErrors:
    """Tests for weighted_sum error handling."""

    def test_empty_attributes(self):
        """Raises error for empty attributes list."""
        with pytest.raises(ValueError, match="At least one Attribute"):
            weighted_sum([], [])

    def test_mismatched_weights_count(self):
        """Raises error when weights count doesn't match attributes."""
        a1 = Attribute([1, 2], item_size=1)
        a2 = Attribute([3, 4], item_size=1)

        with pytest.raises(ValueError, match="must match number of weights"):
            weighted_sum([a1, a2], [0.5])

    def test_mismatched_item_size(self):
        """Raises error when attributes have different item_size."""
        a1 = Attribute([1, 2], item_size=1)
        a2 = Attribute([[1, 0, 0], [0, 1, 0]], item_size=3)

        with pytest.raises(ValueError, match="item_size=3, expected 1"):
            weighted_sum([a1, a2], [0.5, 0.5])

    def test_mismatched_n_items(self):
        """Raises error when attributes have different n_items."""
        a1 = Attribute([1, 2, 3], item_size=1)
        a2 = Attribute([4, 5], item_size=1)

        with pytest.raises(ValueError, match="n_items=2, expected 3"):
            weighted_sum([a1, a2], [0.5, 0.5])


class TestNormalizedWeightedSum:
    """Tests for normalized_weighted_sum function."""

    def test_normalizes_weights(self):
        """Weights are normalized to sum to 1."""
        a1 = Attribute([10, 20, 30], item_size=1)
        a2 = Attribute([20, 30, 40], item_size=1)

        # Weights 1:3 should become 0.25:0.75
        result = normalized_weighted_sum([a1, a2], [1, 3])

        expected = 0.25 * np.array([10, 20, 30]) + 0.75 * np.array([20, 30, 40])
        np.testing.assert_array_almost_equal(result.as_array(), expected)

    def test_equal_weights(self):
        """Equal weights give simple average."""
        a1 = Attribute([0, 0, 0], item_size=1)
        a2 = Attribute([10, 20, 30], item_size=1)

        result = normalized_weighted_sum([a1, a2], [1, 1])

        np.testing.assert_array_almost_equal(result.as_array(), [5, 10, 15])

    def test_already_normalized(self):
        """Already normalized weights work correctly."""
        a1 = Attribute([1, 2], item_size=1)
        a2 = Attribute([3, 4], item_size=1)

        result = normalized_weighted_sum([a1, a2], [0.5, 0.5])

        np.testing.assert_array_almost_equal(result.as_array(), [2, 3])

    def test_large_weights(self):
        """Large weights are normalized correctly."""
        a1 = Attribute([1, 1], item_size=1)
        a2 = Attribute([2, 2], item_size=1)

        result = normalized_weighted_sum([a1, a2], [100, 100])

        np.testing.assert_array_almost_equal(result.as_array(), [1.5, 1.5])

    def test_zero_sum_weights_error(self):
        """Raises error when weights sum to zero."""
        a1 = Attribute([1, 2], item_size=1)
        a2 = Attribute([3, 4], item_size=1)

        with pytest.raises(ValueError, match="Sum of weights cannot be zero"):
            normalized_weighted_sum([a1, a2], [1, -1])


class TestWeightedSumFromManager:
    """Tests for weighted_sum_from_manager function."""

    def test_basic_usage(self):
        """Get attributes by name from manager."""
        manager = AttributeManager()
        manager.add("a", Attribute([1, 2, 3], item_size=1))
        manager.add("b", Attribute([4, 5, 6], item_size=1))

        result = weighted_sum_from_manager(manager, ["a", "b"], [0.5, 0.5])

        np.testing.assert_array_almost_equal(result.as_array(), [2.5, 3.5, 4.5])

    def test_with_result_name(self):
        """Can set result name."""
        manager = AttributeManager()
        manager.add("x", Attribute([1, 2], item_size=1))
        manager.add("y", Attribute([3, 4], item_size=1))

        result = weighted_sum_from_manager(
            manager, ["x", "y"], [0.5, 0.5], result_name="combined"
        )

        assert result.name == "combined"

    def test_with_derived_attributes(self):
        """Works with derived attribute names."""
        manager = AttributeManager()
        manager.add("velocity", Attribute([[3, 0, 0], [0, 4, 0]], item_size=3))
        manager.add("force", Attribute([[6, 0, 0], [0, 8, 0]], item_size=3))

        # Get x components
        result = weighted_sum_from_manager(
            manager, ["velocity:x", "force:x"], [0.5, 0.5]
        )

        np.testing.assert_array_almost_equal(result.as_array(), [4.5, 0])

    def test_missing_attribute_error(self):
        """Raises error for missing attribute."""
        manager = AttributeManager()
        manager.add("a", Attribute([1, 2], item_size=1))

        with pytest.raises(KeyError, match="not found"):
            weighted_sum_from_manager(manager, ["a", "missing"], [0.5, 0.5])

    def test_tensor_attributes(self):
        """Works with tensor attributes from manager."""
        manager = AttributeManager()
        manager.add("stress1", Attribute(np.ones((3, 6)), item_size=6))
        manager.add("stress2", Attribute(np.ones((3, 6)) * 2, item_size=6))

        result = weighted_sum_from_manager(
            manager, ["stress1", "stress2"], [0.4, 0.6]
        )

        expected = np.ones((3, 6)) * 1.6
        np.testing.assert_array_almost_equal(result.as_array(), expected)
        assert result.item_size == 6
