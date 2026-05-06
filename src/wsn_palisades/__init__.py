"""Terrain/vegetation/solar-aware NSGA-III sensor placement."""

__version__ = "0.1.0"

from .params import RANDOM_TRIALS_DEFAULT, SensorParams, SolarParams, nsga_params_for_k

__all__ = [
    "SensorParams",
    "SolarParams",
    "nsga_params_for_k",
    "RANDOM_TRIALS_DEFAULT",
    "__version__",
]
