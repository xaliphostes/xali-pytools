"""
SerieContainer: Container for Series with decomposition support.

Provides query interface to discover available scalar, vector, and tensor
properties, including those derived from decomposers.
"""

from typing import Dict, List, Optional
from .serie import Serie
from .decomposer import Decomposer, DecomposerRegistry


class SerieContainer:
    """
    Container for Series with automatic decomposition support.

    Holds original Series and provides a query interface to discover all
    available properties, including those derived from decomposers.

    Examples:
    ```python
    container = SerieContainer()

    # Add original series
    container.add("pressure", Serie([1, 2, 3], item_size=1))
    container.add("velocity", Serie([[1,2,3], [4,5,6]], item_size=3))
    container.add("stress", Serie(np.zeros((10, 6)), item_size=6))

    # Query available properties
    scalars = container.get_scalar_names()
    # Returns: ["pressure", "velocity:x", "velocity:y", "velocity:z",
    #           "velocity:norm", "stress:xx", "stress:xy", ...]

    vectors = container.get_vector3_names()
    # Returns: ["velocity", "stress:S1_vec", "stress:S2_vec", "stress:S3_vec"]

    # Get a derived property (computed lazily)
    vx = container.get("velocity:x")  # Returns Serie with itemSize=1
    ```
    """

    def __init__(self, registry: Optional[DecomposerRegistry] = None):
        """
        Create a SerieContainer.

        Args:
            registry: Optional custom DecomposerRegistry. If None, uses default.
        """
        self._series: Dict[str, Serie] = {}
        self._registry = registry or DecomposerRegistry.get_instance()
        self._cache: Dict[str, Serie] = {}  # Cache for computed derived series

    def add(self, name: str, serie: Serie) -> "SerieContainer":
        """
        Add a Serie to the container.

        Args:
            name: Name for the Serie.
            serie: The Serie to add.

        Returns:
            self for chaining.
        """
        self._series[name] = serie
        # Invalidate cache for this base name
        self._invalidate_cache(name)
        return self

    def remove(self, name: str) -> "SerieContainer":
        """
        Remove a Serie from the container.

        Args:
            name: Name of the Serie to remove.

        Returns:
            self for chaining.
        """
        if name in self._series:
            del self._series[name]
            self._invalidate_cache(name)
        return self

    def _invalidate_cache(self, base_name: str) -> None:
        """Invalidate cached derived series for a base name."""
        keys_to_remove = [k for k in self._cache if k.startswith(f"{base_name}:")]
        for k in keys_to_remove:
            del self._cache[k]

    def clear_cache(self) -> None:
        """Clear all cached derived series."""
        self._cache.clear()

    @property
    def names(self) -> List[str]:
        """Get names of all original (non-derived) series."""
        return list(self._series.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a name (original or derived) is available."""
        if name in self._series:
            return True
        # Check if it's a derived name
        if ':' in name:
            base_name = name.split(':', 1)[0]
            if base_name in self._series:
                available = self._get_all_derived(base_name)
                return name in available
        return False

    def __len__(self) -> int:
        """Return number of original series."""
        return len(self._series)

    def __iter__(self):
        """Iterate over original series names."""
        return iter(self._series)

    def _get_all_derived(self, base_name: str) -> Dict[str, int]:
        """Get all derived names and their item_sizes for a base serie."""
        if base_name not in self._series:
            return {}

        serie = self._series[base_name]
        decomposers = self._registry.get_decomposers(serie.serie_type)

        result = {}
        for decomposer in decomposers:
            result.update(decomposer.get_available(base_name))
        return result

    def _find_decomposer(self, base_name: str, derived_name: str) -> Optional[Decomposer]:
        """Find the decomposer that can compute a derived name."""
        if base_name not in self._series:
            return None

        serie = self._series[base_name]
        decomposers = self._registry.get_decomposers(serie.serie_type)

        for decomposer in decomposers:
            available = decomposer.get_available(base_name)
            if derived_name in available:
                return decomposer
        return None

    def get(self, name: str) -> Optional[Serie]:
        """
        Get a Serie by name (original or derived).

        For derived names (e.g., "velocity:x"), computes and caches the result.

        Args:
            name: Name of the Serie (original or derived).

        Returns:
            The Serie, or None if not found.
        """
        # Check original series
        if name in self._series:
            return self._series[name]

        # Check cache
        if name in self._cache:
            return self._cache[name]

        # Try to compute derived
        if ':' in name:
            base_name = name.split(':', 1)[0]
            decomposer = self._find_decomposer(base_name, name)
            if decomposer:
                serie = self._series[base_name]
                derived = decomposer.compute(serie, name)
                self._cache[name] = derived
                return derived

        return None

    def __getitem__(self, name: str) -> Serie:
        """Get a Serie by name, raising KeyError if not found."""
        result = self.get(name)
        if result is None:
            raise KeyError(f"Serie '{name}' not found")
        return result

    def get_available(self, item_size: Optional[int] = None) -> Dict[str, int]:
        """
        Get all available property names and their item_sizes.

        Args:
            item_size: If specified, filter to only this item_size.

        Returns:
            Dict mapping property names to their item_size.
        """
        result = {}

        # Add original series
        for name, serie in self._series.items():
            if item_size is None or serie.item_size == item_size:
                result[name] = serie.item_size

        # Add derived series
        for name in self._series:
            derived = self._get_all_derived(name)
            for derived_name, derived_size in derived.items():
                if item_size is None or derived_size == item_size:
                    result[derived_name] = derived_size

        return result

    def get_scalar_names(self) -> List[str]:
        """Get all available scalar (itemSize=1) property names."""
        return sorted(self.get_available(item_size=1).keys())

    def get_vector2_names(self) -> List[str]:
        """Get all available 2D vector (itemSize=2) property names."""
        return sorted(self.get_available(item_size=2).keys())

    def get_vector3_names(self) -> List[str]:
        """Get all available 3D vector (itemSize=3) property names."""
        return sorted(self.get_available(item_size=3).keys())

    def get_sym_tensor2_names(self) -> List[str]:
        """Get all available 2D symmetric tensor (itemSize=3) property names."""
        # Note: itemSize=3 could be vector2 or sym_tensor2
        # This returns all itemSize=3, user may need to distinguish by context
        return sorted(self.get_available(item_size=3).keys())

    def get_sym_tensor3_names(self) -> List[str]:
        """Get all available 3D symmetric tensor (itemSize=6) property names."""
        return sorted(self.get_available(item_size=6).keys())

    def get_tensor3_names(self) -> List[str]:
        """Get all available full 3D tensor (itemSize=9) property names."""
        return sorted(self.get_available(item_size=9).keys())

    def summary(self) -> str:
        """Get a summary of all available properties."""
        lines = ["SerieContainer Summary", "=" * 40]

        # Original series
        lines.append("\nOriginal Series:")
        for name, serie in self._series.items():
            lines.append(f"  {name}: itemSize={serie.item_size}, n={serie.n_items}")

        # Derived by category
        scalars = self.get_scalar_names()
        if scalars:
            lines.append(f"\nScalars ({len(scalars)}):")
            lines.append(f"  {', '.join(scalars)}")

        vec3s = self.get_vector3_names()
        if vec3s:
            lines.append(f"\nVector3 ({len(vec3s)}):")
            lines.append(f"  {', '.join(vec3s)}")

        tensors = self.get_sym_tensor3_names()
        if tensors:
            lines.append(f"\nSymTensor3 ({len(tensors)}):")
            lines.append(f"  {', '.join(tensors)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SerieContainer({len(self._series)} series)"
