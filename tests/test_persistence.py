"""Round-trip save/load for k-sweep dicts."""

import gzip
import pickle
from pathlib import Path

import numpy as np

from wsn_palisades.persistence import load_ksweep, save_ksweep


def test_ksweep_roundtrip(tmp_path: Path):
    res = {
        "FLAT": {
            10: {
                "random": {"idxs": np.array([1, 2, 3]), "F": np.array([-50.0, -0.6, -1500.0, -0.7])},
                "greedy": {"idxs": np.array([4, 5, 6]), "F": np.array([-60.0, -0.65, -1600.0, -0.75])},
            }
        },
        "DEM": {
            10: {"random": {"idxs": np.array([7, 8, 9]), "F": np.array([-55.0, -0.61, -1550.0, -0.71])}}
        },
    }
    p = tmp_path / "res.pkl.gz"
    save_ksweep(res, p)
    loaded = load_ksweep(p)
    assert set(loaded.keys()) == {"FLAT", "DEM"}
    np.testing.assert_array_equal(loaded["FLAT"][10]["random"]["idxs"], [1, 2, 3])
