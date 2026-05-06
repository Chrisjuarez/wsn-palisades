"""Save/load helpers for k-sweep results, packs, and the metrics DataFrame."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that maps numpy._core.* (numpy >= 2.0) back to numpy.core.*.

    Lets older pickles produced under numpy < 2 load on newer numpy and vice versa.
    """

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def save_ksweep(res_k_sweep: dict, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(res_k_sweep, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_ksweep(path: PathLike) -> dict:
    with gzip.open(Path(path), "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def save_metrics_csv(df: pd.DataFrame, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df.columns if c != "idxs"]
    df[cols].to_csv(path, index=False)
    return path


def save_packs(packs: dict, path: PathLike) -> Path:
    """Save a single scenario pack (candidates + masks + metrics + dirpacks)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in packs.items() if not k.startswith("_")}
    with gzip.open(path, "wb") as f:
        pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_packs(path: PathLike) -> dict:
    with gzip.open(Path(path), "rb") as f:
        return _NumpyCompatUnpickler(f).load()


__all__ = ["save_ksweep", "load_ksweep", "save_metrics_csv", "save_packs", "load_packs"]
