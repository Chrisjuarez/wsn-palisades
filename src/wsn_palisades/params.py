"""Dataclasses for sensor and solar configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SensorParams:
    """Geometry and visibility parameters for a single sensor.

    Defaults match the canonical Palisades run.
    """

    R_m: float = 500.0
    az_step_deg: int = 1
    step_m: float = 1.0
    sensor_height_m: float = 2.0
    theta_topo_deg: float = 6.0
    h_thr_m: float = 2.0
    h_ref_m: float = 8.0
    alpha_veg: float = 0.9
    veg_mode: str = "exp"
    min_sep_m: float = 400.0


@dataclass
class SolarParams:
    """Parameters for the per-candidate solar irradiance integration."""

    tz: str = "America/Los_Angeles"
    year: int = 2024
    freq: str = "1h"
    albedo: float = 0.2
    diffuse_model: str = "perez"
    use_svf_for_diffuse: bool = True
    use_horizon_for_direct: bool = True
    use_apparent_zenith: bool = True
    svf_pair_isotropic: bool = False


# Adaptive NSGA-III hyperparameters chosen by problem size.
def nsga_params_for_k(K: int) -> dict:
    if K >= 80:
        return dict(max_gen=600, pop_mult=2.0, partitions=12)
    if K >= 60:
        return dict(max_gen=500, pop_mult=2.0, partitions=12)
    if K >= 40:
        return dict(max_gen=400, pop_mult=2.0, partitions=12)
    return dict(max_gen=350, pop_mult=2.0, partitions=12)


RANDOM_TRIALS_DEFAULT = 120
RANDOM_SEED_DEFAULT = 42
