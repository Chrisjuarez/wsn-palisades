"""SensorParams / SolarParams default sanity checks."""

from wsn_palisades import SensorParams, SolarParams, nsga_params_for_k


def test_sensor_params_defaults():
    sp = SensorParams()
    assert sp.R_m == 500.0
    assert sp.min_sep_m == 400.0
    assert sp.veg_mode in {"exp", "linear"}
    assert sp.az_step_deg >= 1


def test_solar_params_defaults():
    sl = SolarParams()
    assert sl.tz.startswith("America/")
    assert 2000 <= sl.year <= 2100
    assert sl.diffuse_model in {"perez", "isotropic", "haydavies", "reindl", "king"}


def test_nsga_params_for_k_monotone():
    p_small = nsga_params_for_k(10)
    p_big = nsga_params_for_k(100)
    assert p_big["max_gen"] >= p_small["max_gen"]
    assert p_big["partitions"] == p_small["partitions"] >= 1
