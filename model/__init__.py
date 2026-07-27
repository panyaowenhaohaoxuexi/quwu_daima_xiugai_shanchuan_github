"""Public formal model interface."""

from .Teacher import FogRoutedRGBTIRDehazer
from .monotonic_router import MonotonicFogRouter

__all__ = ["FogRoutedRGBTIRDehazer", "MonotonicFogRouter"]
