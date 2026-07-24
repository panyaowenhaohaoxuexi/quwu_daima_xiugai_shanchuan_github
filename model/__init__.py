"""Public formal model interface.

Legacy classes remain importable only through their explicit legacy module path
for checkpoint inspection; importing :mod:`model` never loads them.
"""

from .fog_routed_dehazer import FogRoutedRGBTIRDehazer
from .monotonic_router import MonotonicFogRouter

__all__ = ["FogRoutedRGBTIRDehazer", "MonotonicFogRouter"]
