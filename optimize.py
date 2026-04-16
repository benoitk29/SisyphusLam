"""
optimize.py  -  multi-material optimization
For each (material, X), the CVAE generates LPs, the KD-tree retrieves (a,b),
the CLT checks Ex ≥ Ex_min  and  Ncr ≥ Ncr_required.
Objective  :  min  m = rho · a_plate · b_plate · n_plies · t_ply
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import Config
from material_registry import MATERIAL_REGISTRY, MATERIAL_KEYS
from materials import dd_properties
from dataset_dd import make_condition_vector, denormalise_features
from cvae import CVAE


# ── Optimization problem ──────────────────────────────────────────────────

@dataclass
class DDProblem:
    Nx_applied:  float = -200.0
    Ny_applied:  float =    0.0
    Nxy_applied: float =    0.0
    plate_a:     float =  500.0
    plate_b:     float =  500.0
    Ex_min_GPa:  float =   30.0
    SF_buckling: float =    1.5
    buckling:    bool  =  True
    fpf:         bool  =  False
    # allowed materials (None = all)
    allowed_mats: Optional[List[str]] = None

    @property
    def Ncr_required(self) -> float:
        return self.SF_buckling * abs(self.Nx_applied)


# ── CVAE-guided search ─────────────────────────────────────────────────────

def cvae_guided_search(
    model: CVAE,
    problem: DDProblem,
    cfg: Config,
    lookup_table,
    n_samples_per_X: int   = 5_000,
    temperature:     float = 1.0,
    lp_residual_tol: float = 0.05,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    For each (material, X) authorized :
      1. Builds the condition vector with the one-hot encoding of the material
      2. Generates n_samples_per_X LPs via the CVAE decoder
      3. Inverts LP => (a,b) via KD-tree
      4. Checks Ex, Ncr with the material properties
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    allowed = problem.allowed_mats or MATERIAL_KEYS
    rows    = []

    for mat_key in allowed:
        entry   = MATERIAL_REGISTRY[mat_key]
        mat     = entry.ply_props

        for X in range(cfg.X_min, cfg.X_max + 1):
            n_plies = 4 * X
            h_mm    = n_plies * mat.t
            mass    = mat.rho * problem.plate_a * problem.plate_b * h_mm

            cond_norm = make_condition_vector(
                Nx             = -problem.Ncr_required,
                Ny             =  problem.Ny_applied,
                Nxy            =  problem.Nxy_applied,
                plate_a        =  problem.plate_a,
                plate_b        =  problem.plate_b,
                buckling       =  problem.buckling,
                fpf            =  problem.fpf,
                Ex_target      =  problem.Ex_min_GPa,
                Nombre_de_plis =  float(n_plies),
                mat_key        =  mat_key,
                normalize      =  True,
                cfg            =  cfg,
            )
            ct = torch.tensor(cond_norm, dtype=torch.float32, device=device)

            with torch.no_grad():
                lp_norm = model.generate(ct, n_samples=n_samples_per_X,
                                          temperature=temperature).cpu().numpy()

            lp_raw = denormalise_features(lp_norm)
            angles, residuals = lookup_table.query(lp_raw, X, k=1)
            valid  = residuals[:, 0] < lp_residual_tol

            feasible_rows = []
            for i in np.where(valid)[0]:
                a, b = float(angles[i, 0]), float(angles[i, 1])
                try:
                    p  = dd_properties(a, b, X, mat,
                                        problem.plate_a, problem.plate_b,
                                        n_modes=cfg.n_buckling_modes)
                    ex = p["Ex_MPa"] / 1e3
                    nc = p["Ncr_Npmm"]
                    ok = ex >= problem.Ex_min_GPa and nc >= problem.Ncr_required
                    feasible_rows.append(dict(
                        mat_key=mat_key, mat_name=entry.name,
                        X=X, n_plies=n_plies, h_mm=h_mm,
                        a_deg=round(a,2), b_deg=round(b,2),
                        Ex_GPa=round(ex,3), Ncr_Npmm=round(nc,3),
                        mass_kg=round(mass,8),
                        lp_res=round(float(residuals[i,0]),6),
                        feasible=ok,
                    ))
                except Exception:
                    pass

            n_f = sum(r["feasible"] for r in feasible_rows)
            print(f"  {mat_key:10s} X={X:2d} (n={n_plies:2d})  "
                  f"{valid.sum():5d}/{n_samples_per_X} matched  {n_f} feasible")
            rows.extend(feasible_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        print("!  No feasible design found.")
        return df
    return (df[df["feasible"]]
            .drop_duplicates(subset=["mat_key","a_deg","b_deg","X"])
            .sort_values("mass_kg")
            .reset_index(drop=True))


# ── Brute force ──────────────────────────────────────────────────────────────

def brute_force_grid(
    problem: DDProblem,
    cfg: Config,
    n_angles: int = 90,
    verbose: bool = True,
) -> pd.DataFrame:
    """ exhaustive search over (material, a, b, X)."""
    angles   = np.linspace(0., 90., n_angles)
    X_vals   = np.arange(cfg.X_min, cfg.X_max + 1)
    allowed  = problem.allowed_mats or MATERIAL_KEYS
    total    = len(allowed) * n_angles**2 * len(X_vals)
    if verbose:
        print(f"Brute-force : {len(allowed)} mat × {n_angles}² × {len(X_vals)} "
              f"= {total:,} evaluations")

    rows = []
    for mat_key in allowed:
        entry = MATERIAL_REGISTRY[mat_key]
        mat   = entry.ply_props
        for X in X_vals:
            n_plies = 4 * X; h_mm = n_plies * mat.t
            mass    = mat.rho * problem.plate_a * problem.plate_b * h_mm
            for a in angles:
                for b in angles:
                    if a > b: continue   # symmetry DD
                    try:
                        p  = dd_properties(a, b, X, mat,
                                            problem.plate_a, problem.plate_b,
                                            n_modes=cfg.n_buckling_modes)
                        ex = p["Ex_MPa"] / 1e3; nc = p["Ncr_Npmm"]
                        if ex >= problem.Ex_min_GPa and nc >= problem.Ncr_required:
                            rows.append(dict(
                                mat_key=mat_key, mat_name=entry.name,
                                X=X, n_plies=n_plies, h_mm=h_mm,
                                a_deg=round(a,3), b_deg=round(b,3),
                                Ex_GPa=round(ex,3), Ncr_Npmm=round(nc,3),
                                mass_kg=round(mass,8),
                            ))
                    except Exception:
                        pass

    df = (pd.DataFrame(rows).sort_values("mass_kg").reset_index(drop=True)
          if rows else pd.DataFrame())
    if verbose:
        print(f"Brute force : {len(df)} designs feasible.")
        if not df.empty:
            print(df[["mat_key","n_plies","a_deg","b_deg",
                       "Ex_GPa","Ncr_Npmm","mass_kg"]].head(5).to_string())
    return df


# ── Main solver ─────────────────────────────────────────────────────────
import time
from typing import Optional, Dict


def solve_dd_problem(
    cfg:             Config,
    problem:         DDProblem,
    model:           Optional[CVAE]  = None,
    run_brute_force: bool            = True,
    n_samples_per_X: int             = 5_000,
    temperature:     float           = 1.0,
    lookup_table                     = None,
) -> Dict:
    print("=" * 65)
    print("MULTI-MATERIAL DD OPTIMIZATION")
    print(f"  Nx = {problem.Nx_applied} N/mm  |  Plaque {problem.plate_a}×{problem.plate_b} mm")
    print(f"  Ex ≥ {problem.Ex_min_GPa} GPa  |  Ncr ≥ {problem.Ncr_required} N/mm (SF={problem.SF_buckling})")
    allowed = problem.allowed_mats or MATERIAL_KEYS
    print(f"  Materials ({len(allowed)}) : {', '.join(allowed)}")
    print("=" * 65)

    result: Dict = {}

    t_cvae = None
    t_bf = None

    # ── CVAE ────────────────────────────────────────────────────────────
    if model is not None:
        print("\n=> CVAE guided search")
        start_cvae = time.time()

        result["cvae"] = cvae_guided_search(
            model, problem, cfg, lookup_table,
            n_samples_per_X=n_samples_per_X, temperature=temperature,
        )

        t_cvae = time.time() - start_cvae

        if not result["cvae"].empty:
            b = result["cvae"].iloc[0]
            print(f"\n  CVAE best : [{b['a_deg']}°/-{b['a_deg']}°/"
                  f"{b['b_deg']}°/-{b['b_deg']}°]_{b['X']}  "
                  f"mat={b['mat_key']}  "
                  f"Ex={b['Ex_GPa']} GPa  Ncr={b['Ncr_Npmm']} N/mm  "
                  f"m={b['mass_kg']*1e3:.2f} g")

    # ── BRUTE FORCE ─────────────────────────────────────────────────────
    if run_brute_force:
        print("\n=> Brute-force grid search")
        start_bf = time.time()

        result["brute_force"] = brute_force_grid(problem, cfg)

        t_bf = time.time() - start_bf

        if not result["brute_force"].empty:
            b = result["brute_force"].iloc[0]
            result["best"] = b.to_dict()

    elif model is not None and not result.get("cvae", pd.DataFrame()).empty:
        result["best"] = result["cvae"].iloc[0].to_dict()
    else:
        result["best"] = None

    # ── Design Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if result.get("best"):
        b = result["best"]
        print("DESIGN OPTIMAL :")
        print(f"  Material  : {b.get('mat_name', b.get('mat_key'))}")
        print(f"  Sequence  : [{b['a_deg']}°/-{b['a_deg']}°/"
              f"{b['b_deg']}°/-{b['b_deg']}°]_{b['X']}")
        print(f"  n_plies   : {b['n_plies']}  |  h = {b['h_mm']} mm")
        print(f"  Ex        : {b['Ex_GPa']} GPa  (req ≥ {problem.Ex_min_GPa})")
        print(f"  Ncr       : {b['Ncr_Npmm']} N/mm  (req ≥ {problem.Ncr_required})")
        print(f"  Mass      : {b['mass_kg']*1e3:.3f} g")

    # ── Performance Summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PERFORMANCES :")

    if t_bf is not None:
        print(f"  Time Brute-Force : {t_bf:.4f} s")
    else:
        print("  Time Brute-Force : not executed")

    if t_cvae is not None:
        print(f"  Time CVAE        : {t_cvae:.4f} s")
    else:
        print("  Time CVAE        : not executed")

    if (t_bf is not None) and (t_cvae is not None) and t_cvae > 0:
        speedup = t_bf / max(t_cvae, 1e-8)
        print(f"  Speedup      : x{speedup:.1f}")
    else:
        speedup = None
        print("  Speedup      : N/A")

    # ── Advanced benchmark: time + number of solutions ─────────────────
    print("\n" + "=" * 65)
    print("ADVANCED BENCHMARK :")

    bf_df = result.get("brute_force", None)
    cvae_df = result.get("cvae", None)

    n_bf = len(bf_df) if isinstance(bf_df, pd.DataFrame) else 0
    n_cvae = len(cvae_df) if isinstance(cvae_df, pd.DataFrame) else 0

    print(f"  Feasible laminates BF   : {n_bf}")
    print(f"  Feasible laminates CVAE : {n_cvae}")

    if n_bf > 0:
        coverage = n_cvae / n_bf
        print(f"  Coverage CVAE        : {coverage:.2%}")
    else:
        coverage = None
        print("  Coverage CVAE        : N/A")

    if t_bf is not None and t_bf > 0:
        eff_bf = n_bf / t_bf
        print(f"  Efficiency BF          : {eff_bf:.1f} sol/s")
    else:
        eff_bf = None
        print("  Efficiency BF          : N/A")

    if t_cvae is not None and t_cvae > 0:
        eff_cvae = n_cvae / t_cvae
        print(f"  Efficiency CVAE        : {eff_cvae:.1f} sol/s")
    else:
        eff_cvae = None
        print("  Efficiency CVAE        : N/A")

    def best_mass(df):
        if not isinstance(df, pd.DataFrame) or df.empty or "mass_kg" not in df.columns:
            return None
        return float(df["mass_kg"].min())

    m_bf = best_mass(bf_df)
    m_cvae = best_mass(cvae_df)

    if m_bf is not None:
        print(f"  Best mass BF     : {m_bf*1e3:.3f} g")
    else:
        print("  Best mass BF     : N/A")

    if m_cvae is not None:
        print(f"  Best mass CVAE   : {m_cvae*1e3:.3f} g")
    else:
        print("  Best mass CVAE   : N/A")

    if (m_bf is not None) and (m_cvae is not None) and m_bf > 0:
        mass_gain = (m_bf - m_cvae) / m_bf
        print(f"  Weight gain CVAE vs BF  : {mass_gain:.2%}")
    else:
        mass_gain = None
        print("  Weight gain CVAE vs BF  : N/A")

    print("=" * 65)

    result["timings"] = {
        "cvae": t_cvae,
        "brute_force": t_bf,
        "speedup": speedup,
    }

    result["benchmark"] = {
        "n_feasible_bf": n_bf,
        "n_feasible_cvae": n_cvae,
        "coverage_cvae_vs_bf": coverage,
        "efficiency_bf_sol_per_s": eff_bf,
        "efficiency_cvae_sol_per_s": eff_cvae,
        "best_mass_bf_kg": m_bf,
        "best_mass_cvae_kg": m_cvae,
        "mass_gain_cvae_vs_bf": mass_gain,
    }

    return result