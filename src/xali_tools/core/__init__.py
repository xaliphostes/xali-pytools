# xali_tools.core subpackage

from .attribute import Attribute, AttributeType
from .attribute_manager import AttributeManager
from .decomposer import (
    Decomposer,
    DecomposerRegistry,
    Vector2Decomposer,
    Vector3Decomposer,
    SymTensor2Decomposer,
    SymTensor3Decomposer,
    PrincipalDecomposer,
    Tensor3Decomposer,
    register_default_decomposers,
)
from .math import weighted_sum, normalized_weighted_sum, weighted_sum_from_manager

__all__ = [
    "Attribute",
    "AttributeType",
    "AttributeManager",
    "Decomposer",
    "DecomposerRegistry",
    "Vector2Decomposer",
    "Vector3Decomposer",
    "SymTensor2Decomposer",
    "SymTensor3Decomposer",
    "PrincipalDecomposer",
    "Tensor3Decomposer",
    "register_default_decomposers",
    "weighted_sum",
    "normalized_weighted_sum",
    "weighted_sum_from_manager",
]
