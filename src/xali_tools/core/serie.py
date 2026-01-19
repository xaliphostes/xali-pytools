"""
Serie: Data series with item size metadata for scalar/vector/tensor properties.
"""

import numpy as np
from typing import Iterator, Union


class SerieType:
    """Constants for Serie semantic types."""
    SCALAR = "scalar"
    VECTOR2 = "vector2"
    VECTOR3 = "vector3"
    SYM_TENSOR2 = "sym_tensor2"
    SYM_TENSOR3 = "sym_tensor3"
    TENSOR3 = "tensor3"
    UNKNOWN = "unknown"


# Map item_size to default type (when ambiguous, prefer vector over tensor)
_DEFAULT_TYPES = {
    1: SerieType.SCALAR,
    2: SerieType.VECTOR2,
    3: SerieType.VECTOR3,  # Could also be SYM_TENSOR2
    6: SerieType.SYM_TENSOR3,
    9: SerieType.TENSOR3,
}


class Serie:
    """
    Data series with item size metadata for scalar/vector/tensor properties.

    This class encapsulates a numpy array along with its item size (number of
    components per item), providing convenient iteration and access methods.

    Examples:
    ```python
    # Scalar property (item_size=1)
    pressure = Serie([1.0, 2.0, 3.0], item_size=1)
    for p in pressure:
        print(p)  # prints: 1.0, 2.0, 3.0

    # Vector property (item_size=3)
    velocity = Serie([[1, 2, 3], [4, 5, 6]], item_size=3)
    for v in velocity:
        print(v)  # prints: [1, 2, 3], [4, 5, 6]

    # Symmetric tensor (item_size=6: xx, xy, xz, yy, yz, zz)
    stress = Serie(np.zeros((10, 6)), item_size=6)

    # Explicit type for ambiguous item_size=3
    strain_2d = Serie(data, item_size=3, serie_type=SerieType.SYM_TENSOR2)
    ```

    Attributes:
        item_size: Number of components per item (1=scalar, 3=vector, 6=sym tensor)
        n_items: Number of items in the array
        serie_type: Semantic type (vector3, sym_tensor3, etc.)
    """

    def __init__(
        self,
        data: Union[np.ndarray, list],
        item_size: int = None,
        name: str = "",
        serie_type: str = None
    ):
        """
        Create a Serie.

        Args:
            data: Input data as numpy array or list.
            item_size: Number of components per item. If None, inferred from shape.
            name: Optional name for the property.
            serie_type: Semantic type (e.g., SerieType.VECTOR3). If None, inferred.
        """
        self._data = np.asarray(data, dtype=np.float64)
        self._name = name

        # Infer item_size from shape if not provided
        if item_size is None:
            item_size = self._data.shape[1] if self._data.ndim == 2 else 1
        self._item_size = item_size

        # Infer or set serie_type
        if serie_type is None:
            self._serie_type = _DEFAULT_TYPES.get(item_size, SerieType.UNKNOWN)
        else:
            self._serie_type = serie_type

        # Reshape to (n_items, item_size) or (n_items,) for scalars
        if self._item_size == 1:
            self._data = self._data.flatten()
        else:
            self._data = self._data.reshape(-1, self._item_size)

    @property
    def name(self) -> str:
        """Property name."""
        return self._name

    @property
    def serie_type(self) -> str:
        """Semantic type (e.g., 'vector3', 'sym_tensor3')."""
        return self._serie_type

    @property
    def item_size(self) -> int:
        """Number of components per item."""
        return self._item_size

    @property
    def n_items(self) -> int:
        """Number of items in the array."""
        return len(self._data) if self._item_size == 1 else self._data.shape[0]

    @property
    def is_scalar(self) -> bool:
        """True if this is a scalar property (item_size=1)."""
        return self._item_size == 1

    @property
    def is_vector(self) -> bool:
        """True if this is a 3D vector property (item_size=3)."""
        return self._item_size == 3

    @property
    def is_tensor(self) -> bool:
        """True if this is a symmetric 3D tensor property (item_size=6)."""
        return self._item_size == 6

    @property
    def shape(self) -> tuple:
        """Shape of the underlying array."""
        return self._data.shape

    @property
    def dtype(self):
        """Data type of the underlying array."""
        return self._data.dtype

    def __len__(self) -> int:
        """Return number of items."""
        return self.n_items

    def __getitem__(self, idx) -> Union[float, np.ndarray]:
        """
        Get item(s) by index.

        Returns:
            For scalars: float value(s)
            For vectors/tensors: array of shape (item_size,) or (n, item_size)
        """
        return self._data[idx]

    def __setitem__(self, idx, value):
        """Set item(s) by index."""
        self._data[idx] = value

    def __iter__(self) -> Iterator[Union[float, np.ndarray]]:
        """Iterate over items, yielding scalar or array of item_size."""
        for i in range(self.n_items):
            yield self._data[i]

    def __repr__(self) -> str:
        name_str = f"'{self._name}'" if self._name else ""
        return (
            f"Serie({name_str}, n_items={self.n_items}, "
            f"item_size={self._item_size})"
        )

    def flatten(self) -> np.ndarray:
        """Return flattened 1D array of all values."""
        return self._data.flatten()

    def as_array(self) -> np.ndarray:
        """
        Return as numpy array.

        Returns:
            For scalars: shape (n_items,)
            For vectors/tensors: shape (n_items, item_size)
        """
        return self._data

    def copy(self) -> "Serie":
        """Return a copy of this Serie."""
        return Serie(
            self._data.copy(),
            item_size=self._item_size,
            name=self._name,
            serie_type=self._serie_type
        )

    @classmethod
    def from_flat(
        cls,
        data: np.ndarray,
        item_size: int,
        name: str = "",
        serie_type: str = None
    ) -> "Serie":
        """
        Create from a flat array (makes a copy).

        Args:
            data: Flat 1D array of values.
            item_size: Number of components per item.
            name: Optional property name.
            serie_type: Semantic type (e.g., SerieType.VECTOR3). If None, inferred.

        Returns:
            Serie with data reshaped according to item_size.
        """
        data = np.asarray(data, dtype=np.float64)
        if item_size > 1:
            data = data.reshape(-1, item_size)
        return cls(data, item_size=item_size, name=name, serie_type=serie_type)

    @classmethod
    def as_view(
        cls,
        data: np.ndarray,
        item_size: int = None,
        name: str = "",
        serie_type: str = None
    ) -> "Serie":
        """
        Create a view of an existing numpy array (no copy).

        This is useful for wrapping existing data without duplicating memory.
        Modifications to the Serie will affect the original array.

        Args:
            data: Numpy array to wrap (not copied).
            item_size: Number of components per item. If None, inferred from shape.
            name: Optional property name.
            serie_type: Semantic type (e.g., SerieType.VECTOR3). If None, inferred.

        Returns:
            Serie view of the data.
        """
        instance = cls.__new__(cls)
        instance._name = name

        # Infer item_size from shape if not provided
        if item_size is None:
            item_size = data.shape[1] if data.ndim == 2 else 1
        instance._item_size = item_size

        # Infer or set serie_type
        if serie_type is None:
            instance._serie_type = _DEFAULT_TYPES.get(item_size, SerieType.UNKNOWN)
        else:
            instance._serie_type = serie_type

        # Create view with proper shape (no copy)
        if item_size == 1:
            instance._data = data.ravel()  # view when possible
        else:
            instance._data = data.reshape(-1, item_size)  # view when possible

        return instance
