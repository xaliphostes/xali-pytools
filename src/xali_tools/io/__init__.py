# xali_tools.io subpackage

from xali_tools.geom import SurfaceData  # Re-export from geom for backward compatibility
from xali_tools.core import Attribute
from .surface_loader import load_surface, load_surfaces, save_surface, save_surfaces
from .tsurf_filter import get_tsurf
from .vtp_filter import load_vtp, vtp_to_tsurf, get_vtp_info

__all__ = [
    # Unified surface data structure
    "SurfaceData",
    "Attribute",
    # Unified loader (all formats)
    "load_surface",
    "load_surfaces",
    "save_surface",
    "save_surfaces",
    # TSurf
    "get_tsurf",
    # VTP
    "load_vtp",
    "vtp_to_tsurf",
    "get_vtp_info",
]
