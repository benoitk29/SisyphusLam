"""
config.py — multi-material DD CVAE
Condition : 9 loads/geometry + 12 material one-hot  = 21D
Feature   : 4 LP (ξ¹_A, ξ²_A, ξ¹_D, ξ²_D)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
from material_registry import MATERIAL_KEYS

LOAD_COND_NAMES = ["Nx","Ny","Nxy","plate_a","plate_b","buckling","fpf","Ex_target","n_plies"]
N_LOAD_COND     = len(LOAD_COND_NAMES)       # 9
COND_NAMES      = LOAD_COND_NAMES + [f"mat_{k}" for k in MATERIAL_KEYS]
COND_DIM        = len(COND_NAMES)            # 21
FEATURE_NAMES   = ["xi1_A","xi2_A","xi1_D","xi2_D"]
FEAT_DIM        = len(FEATURE_NAMES)         # 4

@dataclass
class Config:
    X_min: int   = 1
    X_max: int   = 16
    angle_min_deg: float = 0.0
    angle_max_deg: float = 90.0
    discrete_angles: List[float] = field(default_factory=lambda: None)  

    n_samples:            int   = 100_000
    plate_a_range:        Tuple[float,float] = (200.0, 1200.0)
    plate_b_range:        Tuple[float,float] = (200.0, 1200.0)
    safety_margin_range:  Tuple[float,float] = (0.5,   1.5)
    ex_ratio_range:       Tuple[float,float] = (0.8,   1.2)
    n_buckling_modes:     int   = 6
    val_fraction:         float = 0.10
    seed:                 int   = 42

    Nx_bounds:        Tuple[float,float] = (-4000.0, 4000.0)
    Ny_bounds:        Tuple[float,float] = (-4000.0, 4000.0)
    Nxy_bounds:       Tuple[float,float] = (-2000.0, 2000.0)
    plate_a_bounds:   Tuple[float,float] = (100.0,  1500.0)
    plate_b_bounds:   Tuple[float,float] = (100.0,  1500.0)
    Ex_target_bounds: Tuple[float,float] = (1.0,    220.0)
    n_plies_bounds:   Tuple[float,float] = (4.0,    64.0)

    cond_dim:    int       = COND_DIM
    feat_dim:    int       = FEAT_DIM
    latent_dim:  int       = 16
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 512, 256])
    dropout:     float     = 0.05

    batch_size:       int   = 2048
    n_epochs:         int   = 300
    lr:               float = 1e-3
    weight_decay:     float = 1e-5
    beta_kl_max:      float = 0.2
    kl_anneal_epochs: int   = 150

    data_path:  str = "data/dd_dataset.pkl"
    model_path: str = "models/cvae_best.pt"
    log_dir:    str = "logs/"