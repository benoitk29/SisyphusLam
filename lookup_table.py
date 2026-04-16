"""
lookup_table.py  -  KD-tree LP=>(a,b)  (pure geometry, independent of the material)
The LP [ksi¹_A, ksi²_A, ksi¹_D, ksi²_D] depend only on the angles, not on the material.
A single set of trees per X is sufficient for all materials.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy.spatial import KDTree
from config import Config
from materials import dd_sequence, compute_lamination_parameters


def _build_grid_for_X(X: int, n_angles: int) -> Tuple[np.ndarray, np.ndarray]:
    angles  = np.linspace(0., 90., n_angles)
    AA, BB  = np.meshgrid(angles, angles)
    a_flat  = AA.ravel(); b_flat = BB.ravel()
    N       = len(a_flat)
    lp_grid = np.zeros((N, 4), dtype=np.float32)
    ab_grid = np.column_stack([a_flat, b_flat]).astype(np.float32)
    for i, (a, b) in enumerate(zip(a_flat, b_flat)):
        lp = compute_lamination_parameters(dd_sequence(float(a), float(b), X))
        lp_grid[i] = [lp["a1"], lp["a2"], lp["d1"], lp["d2"]]
    return lp_grid, ab_grid


class DDLookupTable:
    def __init__(self):
        self.trees:    Dict[int, KDTree]     = {}
        self.ab:       Dict[int, np.ndarray] = {}
        self.lp:       Dict[int, np.ndarray] = {}
        self.n_angles: int = 0

    @classmethod
    def build(cls, cfg: Config, n_angles: int = 360,
              verbose: bool = True) -> "DDLookupTable":
        lt = cls(); lt.n_angles = n_angles
        for X in range(cfg.X_min, cfg.X_max + 1):
            if verbose:
                print(f"  Building X={X:2d} ({n_angles}²)...", end="\r", flush=True)
            lp_grid, ab_grid = _build_grid_for_X(X, n_angles)
            lt.lp[X] = lp_grid; lt.ab[X] = ab_grid
            lt.trees[X] = KDTree(lp_grid)
        if verbose:
            print(f"\nOK  LT built X={cfg.X_min}-{cfg.X_max}, res {n_angles}×{n_angles}")
        return lt

    def query(self, lp_raw: np.ndarray, X: int,
              k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        lp_raw = np.atleast_2d(lp_raw).astype(np.float32)
        dists, idxs = self.trees[X].query(lp_raw, k=k)
        if k == 1:
            idxs  = idxs[:, None]  if idxs.ndim  == 1 else idxs
            dists = dists[:, None] if dists.ndim == 1 else dists
        angles    = self.ab[X][idxs.ravel()].reshape(len(lp_raw), k * 2)
        return angles, dists

    def query_one(self, lp_raw: np.ndarray, X: int) -> Tuple[float, float, float]:
        lp_raw = np.asarray(lp_raw, dtype=np.float32).reshape(1, -1)
        angles, res = self.query(lp_raw, X, k=1)
        return float(angles[0, 0]), float(angles[0, 1]), float(res[0, 0])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)
        print(f"LT saved => {path}")

    @staticmethod
    def load(path: str) -> "DDLookupTable":
        with open(path, "rb") as f: lt = pickle.load(f)
        print(f"LT loaded ← {path}  (X={min(lt.trees)}-{max(lt.trees)}, "
              f"res={lt.n_angles}×{lt.n_angles})")
        return lt