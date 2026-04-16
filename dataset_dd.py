"""
dataset_dd.py  –  génération multi-matériaux
Condition 21D = [Nx,Ny,Nxy,plate_a,plate_b,buckling,fpf,Ex_target,n_plies | mat_one_hot(12)]
Feature   4D  = [ξ¹_A, ξ²_A, ξ¹_D, ξ²_D]  (LP géométriques, tanh range ≈ [-1,1])
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import Config, COND_NAMES, N_LOAD_COND, FEATURE_NAMES
from material_registry import (MATERIAL_REGISTRY, MATERIAL_KEYS, N_MAT,
                            mat_one_hot, mat_index)
from materials import dd_properties, compute_lamination_parameters, dd_sequence


# ── Normalisation conditions ─────────────────────────────────────────────────

def _normalise_conditions(load_raw: np.ndarray, cfg: Config) -> np.ndarray:
    """Normalizes the 9 load/geometry channels to [0, 1]."""
    bounds = [
        cfg.Nx_bounds, cfg.Ny_bounds, cfg.Nxy_bounds,
        cfg.plate_a_bounds, cfg.plate_b_bounds,
        (0., 1.), (0., 1.), cfg.Ex_target_bounds, cfg.n_plies_bounds,
    ]
    out = load_raw.copy().astype(np.float32)
    for i, (lo, hi) in enumerate(bounds):
        if i <= 2:   # log-scale symétrique
            max_log = np.log10(max(abs(lo), abs(hi)) + 1.)
            sign    = np.sign(load_raw[i])
            out[i]  = float(np.clip((sign * np.log10(abs(load_raw[i]) + 1.) / max_log + 1.) / 2., 0., 1.))
        
        elif i == 7:
            # x = float(np.clip(load_raw[i], lo, hi))
            # out[i] = float(
            #     (np.log1p(x) - np.log1p(lo)) /
            #     (np.log1p(hi) - np.log1p(lo) + 1e-12)
            # )
            val = np.clip(load_raw[i], 10.0, 150.0) 
            out[i] = (val - 10.0) / (150.0 - 10.0)
        else:
            out[i]  = float(np.clip((load_raw[i] - lo) / (hi - lo + 1e-12), 0., 1.))
    return out


def _denormalise_conditions(norm_load: np.ndarray, cfg: Config) -> np.ndarray:
    bounds = [
        cfg.Nx_bounds, cfg.Ny_bounds, cfg.Nxy_bounds,
        cfg.plate_a_bounds, cfg.plate_b_bounds,
        (0., 1.), (0., 1.), cfg.Ex_target_bounds, cfg.n_plies_bounds,
    ]
    out = norm_load.copy().astype(np.float32)
    for i, (lo, hi) in enumerate(bounds):
        if i <= 2:
            max_log  = np.log10(max(abs(lo), abs(hi)) + 1.)
            v_scaled = float(norm_load[i]) * 2. - 1.
            out[i]   = float(np.sign(v_scaled) * (10 ** (abs(v_scaled) * max_log) - 1.))
        elif i == 7:
            v = float(np.clip(norm_load[i], 0., 1.))
            out[i] = float(np.expm1(
                v * (np.log1p(hi) - np.log1p(lo)) + np.log1p(lo)
            ))
        else:
            out[i] = float(norm_load[i]) * (hi - lo) + lo
    return out


# ── Feature normalisation (LP € [-1,1] — identity) ──────────────────────────

def normalise_features(f: np.ndarray) -> np.ndarray:
    return f.astype(np.float32)

def denormalise_features(f: np.ndarray) -> np.ndarray:
    return f.astype(np.float32)


# ── public API: make_condition_vector ─────────────────────────────────────

def make_condition_vector(
    Nx:             float  = -200.0,
    Ny:             float  =    0.0,
    Nxy:            float  =    0.0,
    plate_a:        float  =  500.0,
    plate_b:        float  =  500.0,
    buckling:       bool   =  True,
    fpf:            bool   =  False,
    Ex_target:      float  =   30.0,
    Nombre_de_plis: float  =   56.0,
    mat_key:        str    = "carbon",
    normalize:      bool   =  True,
    cfg:            Config =  None,
) -> np.ndarray:
    """Construct the normalized or raw 21D condition vector."""
    load_raw = np.array([Nx, Ny, Nxy, plate_a, plate_b,
                         float(buckling), float(fpf), Ex_target, Nombre_de_plis],
                        dtype=np.float32)
    mat_oh   = mat_one_hot(mat_key)
    raw      = np.concatenate([load_raw, mat_oh])

    if not normalize:
        return raw
    if cfg is None:
        raise ValueError("cfg required when normalizing=True.")
    norm = raw.copy()
    norm[:N_LOAD_COND] = _normalise_conditions(load_raw, cfg)
    return norm


# ── Génération du dataset ────────────────────────────────────────────────────

def generate_dd_dataset(cfg: Config, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generates cfg.n_samples pairs (condition 21D, feature 4D).
    Samples uniformly across all materials in the registry.
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    angles_pool = (np.array(cfg.discrete_angles, dtype=float)
                   if cfg.discrete_angles else None)

    rows, attempts = [], 0
    max_att = cfg.n_samples * 40

    while len(rows) < cfg.n_samples and attempts < max_att:
        attempts += 1
        try:
            # ── tirage ──────────────────────────────────────────────────
            mat_key = MATERIAL_KEYS[int(rng.integers(0, N_MAT))]
            mat     = MATERIAL_REGISTRY[mat_key].ply_props
            X       = int(rng.integers(cfg.X_min, cfg.X_max + 1))
            n_plies = 4 * X

            if angles_pool is not None:
                a_r = float(rng.choice(angles_pool))
                b_r = float(rng.choice(angles_pool))
            else:
                a_r = float(rng.uniform(cfg.angle_min_deg, cfg.angle_max_deg))
                b_r = float(rng.uniform(cfg.angle_min_deg, cfg.angle_max_deg))
            a, b = min(a_r, b_r), max(a_r, b_r)   # symétrie DD

            plate_a = float(rng.uniform(*cfg.plate_a_range))
            plate_b = float(rng.uniform(*cfg.plate_b_range))

            # ── CLT ─────────────────────────────────────────────────────
            props = dd_properties(a, b, X, mat, plate_a, plate_b,
                                  n_modes=cfg.n_buckling_modes)
            Ex_r  = props["Ex_MPa"] / 1e3
            Ncr_r = props["Ncr_Npmm"]
            if Ncr_r <= 0 or Ex_r <= 0:
                continue
            lp = props["lp"]

            # ── conditions ──────────────────────────────────────────────
            margin    = float(rng.uniform(*cfg.safety_margin_range))
            Nx_d      = -Ncr_r * margin
            Ex_target = Ex_r * float(rng.uniform(*cfg.ex_ratio_range))

            load_raw  = np.array([Nx_d, 0., 0., plate_a, plate_b,
                                   1., 0., Ex_target, float(n_plies)], dtype=np.float32)
            mat_oh    = mat_one_hot(mat_key)
            cond_raw  = np.concatenate([load_raw, mat_oh])
            cond_norm = cond_raw.copy()
            cond_norm[:N_LOAD_COND] = _normalise_conditions(load_raw, cfg)

            # ── features LP ─────────────────────────────────────────────
            feat_raw  = np.array([lp["a1"], lp["a2"], lp["d1"], lp["d2"]], dtype=np.float32)
            feat_norm = normalise_features(feat_raw)

            # ── stockage ─────────────────────────────────────────────────
            row = {}
            for j, name in enumerate(COND_NAMES):
                row[f"c_{name}"]     = float(cond_norm[j])
                row[f"c_{name}_raw"] = float(cond_raw[j])
            for j, name in enumerate(FEATURE_NAMES):
                row[f"f_{name}"]     = float(feat_norm[j])
                row[f"f_{name}_raw"] = float(feat_raw[j])
            row.update(dict(mat_key=mat_key, mat_idx=mat_index(mat_key),
                            a_deg=a, b_deg=b, X_blocks=X, n_plies=n_plies,
                            h_mm=props["h_mm"], Ex_GPa=Ex_r, Ncr_Npmm=Ncr_r,
                            margin=margin, plate_a=plate_a, plate_b=plate_b,
                            rho=mat.rho))
            rows.append(row)
        except Exception:
            continue

    if len(rows) < cfg.n_samples:
        print(f"! {len(rows)}/{cfg.n_samples} samples ({attempts} attempts).")

    df = pd.DataFrame(rows)
    print(f"OK  Dataset {len(df)} samples.  |  X {df['X_blocks'].min()}–{df['X_blocks'].max()}  "
          f"|  Ex {df['Ex_GPa'].min():.1f}–{df['Ex_GPa'].max():.1f} GPa  "
          f"|  {df['mat_key'].nunique()}/{N_MAT} materials")
    return df


# ── I/O ─────────────────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    print(f"Dataset saved → {path}  ({len(df)} rows)")

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    print(f"Dataset loaded ← {path}  ({len(df)} rows)")
    return df


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class DDLaminateDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        cond_cols = [f"c_{n}" for n in COND_NAMES]
        feat_cols = [f"f_{n}" for n in FEATURE_NAMES]
        self.cond = torch.tensor(df[cond_cols].values, dtype=torch.float32)
        self.feat = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    def __len__(self):
        return len(self.cond)
    def __getitem__(self, idx):
        return self.cond[idx], self.feat[idx]