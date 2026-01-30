"""
Mathematical operations on Attributes.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, TYPE_CHECKING
from .attribute import Attribute

if TYPE_CHECKING:
    from .attribute_manager import AttributeManager


def weighted_sum(
    attributes: Sequence[Attribute],
    weights: Sequence[float],
    name: str = ""
) -> Attribute:
    """
    Compute the weighted sum of multiple Attributes.

    All Attributes must have the same item_size and n_items.

    Args:
        attributes: Sequence of Attributes to sum.
        weights: Sequence of weights (same length as attributes).
        name: Optional name for the resulting Attribute.

    Returns:
        New Attribute containing the weighted sum.

    Raises:
        ValueError: If attributes/weights are empty, lengths don't match,
                    or Attributes have incompatible shapes.

    Examples:
        ```python
        # Weighted sum of scalar attributes
        a1 = Attribute([1, 2, 3], item_size=1)
        a2 = Attribute([4, 5, 6], item_size=1)
        result = weighted_sum([a1, a2], [0.3, 0.7])
        # result = [0.3*1 + 0.7*4, 0.3*2 + 0.7*5, 0.3*3 + 0.7*6]

        # Weighted sum of vector attributes
        v1 = Attribute([[1, 0, 0], [0, 1, 0]], item_size=3)
        v2 = Attribute([[0, 1, 0], [1, 0, 0]], item_size=3)
        result = weighted_sum([v1, v2], [0.5, 0.5])
        # result = [[0.5, 0.5, 0], [0.5, 0.5, 0]]
        ```
    """
    if len(attributes) == 0:
        raise ValueError("At least one Attribute is required")

    if len(attributes) != len(weights):
        raise ValueError(
            f"Number of attributes ({len(attributes)}) must match "
            f"number of weights ({len(weights)})"
        )

    # Validate all attributes have same item_size and n_items
    first = attributes[0]
    item_size = first.item_size
    n_items = first.n_items
    attr_type = first.attribute_type

    for i, attr in enumerate(attributes[1:], start=1):
        if attr.item_size != item_size:
            raise ValueError(
                f"Attribute {i} has item_size={attr.item_size}, "
                f"expected {item_size}"
            )
        if attr.n_items != n_items:
            raise ValueError(
                f"Attribute {i} has n_items={attr.n_items}, "
                f"expected {n_items}"
            )

    # Compute weighted sum
    weights = np.asarray(weights, dtype=np.float64)
    result = np.zeros_like(first.as_array())

    for attr, w in zip(attributes, weights):
        result += w * attr.as_array()

    return Attribute(result, item_size=item_size, name=name, attribute_type=attr_type)


def normalized_weighted_sum(
    attributes: Sequence[Attribute],
    weights: Sequence[float],
    name: str = ""
) -> Attribute:
    """
    Compute the weighted sum with weights normalized to sum to 1.

    This is equivalent to weighted_sum but automatically normalizes the weights,
    useful for computing weighted averages.

    Args:
        attributes: Sequence of Attributes to sum.
        weights: Sequence of weights (will be normalized).
        name: Optional name for the resulting Attribute.

    Returns:
        New Attribute containing the normalized weighted sum.

    Raises:
        ValueError: If sum of weights is zero.

    Examples:
        ```python
        a1 = Attribute([10, 20, 30], item_size=1)
        a2 = Attribute([20, 30, 40], item_size=1)

        # These are equivalent:
        result1 = normalized_weighted_sum([a1, a2], [1, 3])
        result2 = weighted_sum([a1, a2], [0.25, 0.75])
        ```
    """
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = np.sum(weights)

    if np.isclose(weight_sum, 0.0):
        raise ValueError("Sum of weights cannot be zero")

    normalized = weights / weight_sum
    return weighted_sum(attributes, normalized, name=name)


def weighted_sum_from_manager(
    manager: AttributeManager,
    names: Sequence[str],
    weights: Sequence[float],
    result_name: str = ""
) -> Attribute:
    """
    Compute weighted sum of Attributes from an AttributeManager.

    Args:
        manager: AttributeManager containing the Attributes.
        names: Names of Attributes to sum (can include derived names like "velocity:x").
        weights: Sequence of weights.
        result_name: Optional name for the resulting Attribute.

    Returns:
        New Attribute containing the weighted sum.

    Examples:
        ```python
        manager = AttributeManager()
        manager.add("stress1", Attribute(data1, item_size=6))
        manager.add("stress2", Attribute(data2, item_size=6))

        # Weighted sum of full tensors
        combined = weighted_sum_from_manager(
            manager, ["stress1", "stress2"], [0.4, 0.6]
        )

        # Weighted sum of derived scalars
        combined_vonmises = weighted_sum_from_manager(
            manager, ["stress1:vonMises", "stress2:vonMises"], [0.5, 0.5]
        )
        ```
    """
    attributes = []
    for name in names:
        attr = manager.get(name)
        if attr is None:
            raise KeyError(f"Attribute '{name}' not found in manager")
        attributes.append(attr)

    return weighted_sum(attributes, weights, name=result_name)
