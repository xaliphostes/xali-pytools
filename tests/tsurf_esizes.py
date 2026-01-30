"""Tests for TSurf file loading/saving with ESIZES support."""

import numpy as np
import os
from xali_tools.io.tsurf_filter import load_tsurf, save_tsurf
from xali_tools.io import SurfaceData

"""Test that ESIZES correctly contains '1 3 6' for scalar, vector, tensor."""
n_vertices = 4

# Create surface with scalar (1), vector (3), and tensor (6) properties
positions = np.array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0
], dtype=np.float64)

indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

# Scalar property (item_size=1)
scalar_prop = np.array([1.0, 2.0, 3.0, 4.0])

# Vector property (item_size=3)
vector_prop = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

# Symmetric tensor property (item_size=6: xx, xy, xz, yy, yz, zz)
tensor_prop = np.array([1.0, 0.1, 0.2, 2.0, 0.3, 3.0, 1.1, 0.4, 0.5, 2.1, 0.6, 3.1, 1.2, 0.7, 0.8, 2.2, 0.9, 3.2, 1.3, 1.0, 1.1, 2.3, 1.2, 3.3])

surface = SurfaceData(
    positions=positions,
    indices=indices,
    # properties={
    #     'temperature': scalar_prop,
    #     'displacement': vector_prop,
    #     'stress': tensor_prop
    # },
    # property_sizes={
    #     'temperature': 1,
    #     'displacement': 3,
    #     'stress': 6
    # },
    name='test_surface'
)
surface.set_property("stress", tensor_prop, 6)
surface.set_property("temperature", scalar_prop, 1)
surface.set_property("displacement", vector_prop, 3)

save_tsurf(surface, "tmp_file.ts")
