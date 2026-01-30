"""
Decomposer: Architecture for decomposing Attributes into derived properties.

An Attribute with itemSize=3 (vector) can be decomposed into scalars (x, y, z, norm).
An Attribute with itemSize=6 (symmetric tensor) can be decomposed into components,
principal values, principal vectors, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Type, Callable
import numpy as np

from .attribute import Attribute, AttributeType


class Decomposer(ABC):
    """
    Abstract base class for Attribute decomposers.

    A decomposer takes an Attribute of a specific type and can produce
    derived Attributes of potentially different item_sizes.
    """

    @property
    @abstractmethod
    def input_attribute_type(self) -> str:
        """The attribute_type this decomposer can process."""
        pass

    @property
    @abstractmethod
    def input_item_size(self) -> int:
        """The item_size this decomposer can process."""
        pass

    @abstractmethod
    def get_available(self, base_name: str) -> Dict[str, int]:
        """
        Get available derived properties.

        Args:
            base_name: Name of the source Attribute.

        Returns:
            Dict mapping derived names to their item_size.
            E.g., {"T:x": 1, "T:y": 1, "T:z": 1, "T:norm": 1}
        """
        pass

    @abstractmethod
    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        """
        Compute a derived Attribute.

        Args:
            attribute: Source Attribute to decompose.
            derived_name: Name of the derived property (e.g., "T:x").

        Returns:
            The computed derived Attribute.
        """
        pass

    def get_suffix(self, derived_name: str) -> str:
        """Extract suffix from derived name (e.g., 'T:x' -> 'x')."""
        if ':' in derived_name:
            return derived_name.split(':', 1)[1]
        return derived_name


class DecomposerRegistry:
    """
    Registry of decomposers indexed by attribute_type.

    Allows registration of new decomposers and lookup by attribute_type.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._decomposers: Dict[str, List[Decomposer]] = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "DecomposerRegistry":
        """Get the singleton registry instance."""
        return cls()

    def register(self, decomposer: Decomposer) -> None:
        """Register a decomposer."""
        attribute_type = decomposer.input_attribute_type
        if attribute_type not in self._decomposers:
            self._decomposers[attribute_type] = []
        # Avoid duplicates
        for existing in self._decomposers[attribute_type]:
            if type(existing) == type(decomposer):
                return
        self._decomposers[attribute_type].append(decomposer)

    def get_decomposers(self, attribute_type: str) -> List[Decomposer]:
        """Get all decomposers for a given attribute_type."""
        return self._decomposers.get(attribute_type, [])

    def get_all_decomposers(self) -> Dict[str, List[Decomposer]]:
        """Get all registered decomposers."""
        return self._decomposers.copy()

    def clear(self) -> None:
        """Clear all registered decomposers (useful for testing)."""
        self._decomposers.clear()


# =============================================================================
# Built-in Decomposers
# =============================================================================

class Vector3Decomposer(Decomposer):
    """
    Decompose a 3D vector (itemSize=3) into scalar components and derived values.

    Produces:
    - x, y, z: individual components (itemSize=1)
    - norm: vector magnitude (itemSize=1)
    """

    @property
    def input_attribute_type(self) -> str:
        return AttributeType.VECTOR3

    @property
    def input_item_size(self) -> int:
        return 3

    def get_available(self, base_name: str) -> Dict[str, int]:
        return {
            f"{base_name}:x": 1,
            f"{base_name}:y": 1,
            f"{base_name}:z": 1,
            f"{base_name}:norm": 1,
        }

    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        suffix = self.get_suffix(derived_name)
        data = attribute.as_array()

        if suffix == 'x':
            result = data[:, 0]
        elif suffix == 'y':
            result = data[:, 1]
        elif suffix == 'z':
            result = data[:, 2]
        elif suffix == 'norm':
            result = np.linalg.norm(data, axis=1)
        else:
            raise ValueError(f"Unknown suffix: {suffix}")

        return Attribute(result, item_size=1, name=derived_name)


class Vector2Decomposer(Decomposer):
    """
    Decompose a 2D vector (itemSize=2) into scalar components and derived values.

    Produces:
    - x, y: individual components (itemSize=1)
    - norm: vector magnitude (itemSize=1)
    """

    @property
    def input_attribute_type(self) -> str:
        return AttributeType.VECTOR2

    @property
    def input_item_size(self) -> int:
        return 2

    def get_available(self, base_name: str) -> Dict[str, int]:
        return {
            f"{base_name}:x": 1,
            f"{base_name}:y": 1,
            f"{base_name}:norm": 1,
        }

    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        suffix = self.get_suffix(derived_name)
        data = attribute.as_array()

        if suffix == 'x':
            result = data[:, 0]
        elif suffix == 'y':
            result = data[:, 1]
        elif suffix == 'norm':
            result = np.linalg.norm(data, axis=1)
        else:
            raise ValueError(f"Unknown suffix: {suffix}")

        return Attribute(result, item_size=1, name=derived_name)


class SymTensor3Decomposer(Decomposer):
    """
    Decompose a symmetric 3x3 tensor (itemSize=6) into scalar components.

    Storage order: [xx, xy, xz, yy, yz, zz]

    Produces:
    - xx, xy, xz, yy, yz, zz: individual components (itemSize=1)
    - trace: tensor trace (itemSize=1)
    - von_mises: von Mises equivalent stress (itemSize=1)
    """

    # Component indices in storage order [xx, xy, xz, yy, yz, zz]
    COMPONENTS = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    INDICES = {name: i for i, name in enumerate(COMPONENTS)}

    @property
    def input_attribute_type(self) -> str:
        return AttributeType.SYM_TENSOR3

    @property
    def input_item_size(self) -> int:
        return 6

    def get_available(self, base_name: str) -> Dict[str, int]:
        result = {}
        for comp in self.COMPONENTS:
            result[f"{base_name}:{comp}"] = 1
        result[f"{base_name}:trace"] = 1
        result[f"{base_name}:von_mises"] = 1
        return result

    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        suffix = self.get_suffix(derived_name)
        data = attribute.as_array()  # shape (n, 6)

        if suffix in self.INDICES:
            result = data[:, self.INDICES[suffix]]
        elif suffix == 'trace':
            # trace = xx + yy + zz
            result = data[:, 0] + data[:, 3] + data[:, 5]
        elif suffix == 'von_mises':
            # von Mises: sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2 + 6*(t12^2+t23^2+t31^2)))
            # For symmetric tensor [xx, xy, xz, yy, yz, zz]:
            xx, xy, xz, yy, yz, zz = [data[:, i] for i in range(6)]
            result = np.sqrt(0.5 * (
                (xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2 +
                6 * (xy**2 + yz**2 + xz**2)
            ))
        else:
            raise ValueError(f"Unknown suffix: {suffix}")

        return Attribute(result, item_size=1, name=derived_name)


class SymTensor2Decomposer(Decomposer):
    """
    Decompose a symmetric 2x2 tensor (itemSize=3) into scalar components.

    Storage order: [xx, xy, yy]

    Produces:
    - xx, xy, yy: individual components (itemSize=1)
    - trace: tensor trace (itemSize=1)
    """

    COMPONENTS = ['xx', 'xy', 'yy']
    INDICES = {name: i for i, name in enumerate(COMPONENTS)}

    @property
    def input_attribute_type(self) -> str:
        return AttributeType.SYM_TENSOR2

    @property
    def input_item_size(self) -> int:
        return 3

    def get_available(self, base_name: str) -> Dict[str, int]:
        result = {}
        for comp in self.COMPONENTS:
            result[f"{base_name}:{comp}"] = 1
        result[f"{base_name}:trace"] = 1
        return result

    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        suffix = self.get_suffix(derived_name)
        data = attribute.as_array()  # shape (n, 3)

        if suffix in self.INDICES:
            result = data[:, self.INDICES[suffix]]
        elif suffix == 'trace':
            result = data[:, 0] + data[:, 2]  # xx + yy
        else:
            raise ValueError(f"Unknown suffix: {suffix}")

        return Attribute(result, item_size=1, name=derived_name)


class PrincipalDecomposer(Decomposer):
    """
    Decompose a symmetric 3x3 tensor (itemSize=6) into principal values and vectors.

    Storage order: [xx, xy, xz, yy, yz, zz]

    Produces:
    - S1, S2, S3: principal values sorted by magnitude (itemSize=1)
    - S1_vec, S2_vec, S3_vec: principal vectors (itemSize=3)
    - principal_values: all 3 principal values as vector (itemSize=3)
    """

    @property
    def input_attribute_type(self) -> str:
        return AttributeType.SYM_TENSOR3

    @property
    def input_item_size(self) -> int:
        return 6

    def get_available(self, base_name: str) -> Dict[str, int]:
        return {
            f"{base_name}:S1": 1,
            f"{base_name}:S2": 1,
            f"{base_name}:S3": 1,
            f"{base_name}:S1_vec": 3,
            f"{base_name}:S2_vec": 3,
            f"{base_name}:S3_vec": 3,
            f"{base_name}:principal_values": 3,
        }

    def _to_matrix(self, row: np.ndarray) -> np.ndarray:
        """Convert [xx, xy, xz, yy, yz, zz] to 3x3 symmetric matrix."""
        xx, xy, xz, yy, yz, zz = row
        return np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz]
        ])

    def _compute_principals(self, data: np.ndarray):
        """Compute principal values and vectors for all items."""
        n = data.shape[0]
        values = np.zeros((n, 3))
        vectors = np.zeros((n, 3, 3))

        for i in range(n):
            mat = self._to_matrix(data[i])
            eigvals, eigvecs = np.linalg.eigh(mat)
            # Sort by descending eigenvalue magnitude
            order = np.argsort(np.abs(eigvals))[::-1]
            values[i] = eigvals[order]
            vectors[i] = eigvecs[:, order].T  # Each row is a principal vector

        return values, vectors

    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        suffix = self.get_suffix(derived_name)
        data = attribute.as_array()

        values, vectors = self._compute_principals(data)

        if suffix == 'S1':
            result = values[:, 0]
            return Attribute(result, item_size=1, name=derived_name)
        elif suffix == 'S2':
            result = values[:, 1]
            return Attribute(result, item_size=1, name=derived_name)
        elif suffix == 'S3':
            result = values[:, 2]
            return Attribute(result, item_size=1, name=derived_name)
        elif suffix == 'S1_vec':
            result = vectors[:, 0, :]
            return Attribute(result, item_size=3, name=derived_name)
        elif suffix == 'S2_vec':
            result = vectors[:, 1, :]
            return Attribute(result, item_size=3, name=derived_name)
        elif suffix == 'S3_vec':
            result = vectors[:, 2, :]
            return Attribute(result, item_size=3, name=derived_name)
        elif suffix == 'principal_values':
            return Attribute(values, item_size=3, name=derived_name)
        else:
            raise ValueError(f"Unknown suffix: {suffix}")


class Tensor3Decomposer(Decomposer):
    """
    Decompose a full 3x3 tensor (itemSize=9) into scalar components.

    Storage order (row-major): [xx, xy, xz, yx, yy, yz, zx, zy, zz]

    Produces:
    - xx, xy, xz, yx, yy, yz, zx, zy, zz: individual components (itemSize=1)
    - trace: tensor trace (itemSize=1)
    - symmetric: symmetric part (itemSize=6)
    - antisymmetric: antisymmetric part (itemSize=3, as vector [wx, wy, wz])
    """

    COMPONENTS = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
    INDICES = {name: i for i, name in enumerate(COMPONENTS)}

    @property
    def input_attribute_type(self) -> str:
        return AttributeType.TENSOR3

    @property
    def input_item_size(self) -> int:
        return 9

    def get_available(self, base_name: str) -> Dict[str, int]:
        result = {}
        for comp in self.COMPONENTS:
            result[f"{base_name}:{comp}"] = 1
        result[f"{base_name}:trace"] = 1
        result[f"{base_name}:symmetric"] = 6
        result[f"{base_name}:antisymmetric"] = 3
        return result

    def compute(self, attribute: Attribute, derived_name: str) -> Attribute:
        suffix = self.get_suffix(derived_name)
        data = attribute.as_array()  # shape (n, 9)

        if suffix in self.INDICES:
            result = data[:, self.INDICES[suffix]]
            return Attribute(result, item_size=1, name=derived_name)
        elif suffix == 'trace':
            # trace = xx + yy + zz (indices 0, 4, 8)
            result = data[:, 0] + data[:, 4] + data[:, 8]
            return Attribute(result, item_size=1, name=derived_name)
        elif suffix == 'symmetric':
            # Symmetric part: S_ij = 0.5 * (T_ij + T_ji)
            # [xx, xy, xz, yy, yz, zz]
            xx = data[:, 0]
            xy = 0.5 * (data[:, 1] + data[:, 3])
            xz = 0.5 * (data[:, 2] + data[:, 6])
            yy = data[:, 4]
            yz = 0.5 * (data[:, 5] + data[:, 7])
            zz = data[:, 8]
            result = np.column_stack([xx, xy, xz, yy, yz, zz])
            return Attribute(result, item_size=6, name=derived_name)
        elif suffix == 'antisymmetric':
            # Antisymmetric part as axial vector: w = [yz-zy, zx-xz, xy-yx] / 2
            wx = 0.5 * (data[:, 5] - data[:, 7])  # yz - zy
            wy = 0.5 * (data[:, 6] - data[:, 2])  # zx - xz
            wz = 0.5 * (data[:, 1] - data[:, 3])  # xy - yx
            result = np.column_stack([wx, wy, wz])
            return Attribute(result, item_size=3, name=derived_name)
        else:
            raise ValueError(f"Unknown suffix: {suffix}")


# =============================================================================
# Registry initialization
# =============================================================================

def register_default_decomposers():
    """Register all built-in decomposers."""
    registry = DecomposerRegistry.get_instance()
    registry.register(Vector2Decomposer())
    registry.register(Vector3Decomposer())
    registry.register(SymTensor2Decomposer())
    registry.register(SymTensor3Decomposer())
    registry.register(PrincipalDecomposer())
    registry.register(Tensor3Decomposer())


# Auto-register on module import
register_default_decomposers()
