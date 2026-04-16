"""
evaluate.py  –  multi-material metrics
R² reconstruction, feasibility rate by (material, X), angular diversity.
"""
from __future__ import annotations
from typing import Dict
import numpy as np, pandas as torch
from torch.utils.data import DataLoader
from config import FEATURE_NAMES
from dataset_dd import DDLaminateDataset, denormalise_features, make_condition_vector
from material_registry import MATERIAL_KEYS, MATERIAL_REGISTRY
from materials import dd_properties


def encoder_r2(model, df, cfg, batch_size=2048, device=None):
    if device is None: device = torch.device("cpu")
    model.eval()
    ds = DDLaminateDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    true_l, recon_l, mu_l = [], [], []
    with torch.no_grad():
        for cond, feat in loader:
            cond, feat = cond.to(device), feat.to(device)
            x_hat, mu, _ = model(feat, cond)
            true_l.append(feat.cpu().numpy())
            recon_l.append(x_hat.cpu().numpy())
            mu_l.append(mu[:, :cfg.feat_dim].cpu().numpy())
    t = np.vstack(true_l); r = np.vstack(recon_l); m = np.vstack(mu_l)
    res: Dict = {}
    for i, name in enumerate(FEATURE_NAMES):
        ss = np.var(t[:, i]) * len(t)
        if ss < 1e-10: res[f"{name}_recon"] = res[f"{name}_mu"] = 0.; continue
        res[f"{name}_recon"] = float(1 - np.sum((t[:,i]-r[:,i])**2)/ss)
        res[f"{name}_mu"]    = float(1 - np.sum((t[:,i]-m[:,i])**2)/ss)
    res["mean_recon_r2"] = float(np.mean([res[f"{n}_recon"] for n in FEATURE_NAMES]))
    res["mean_mu_r2"]    = float(np.mean([res[f"{n}_mu"]    for n in FEATURE_NAMES]))
    return res


def reconstruction_mse(model, df, cfg, batch_size=2048, device=None):
    if device is None: device = torch.device("cpu")
    model.eval()
    ds = DDLaminateDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    sq = np.zeros(cfg.feat_dim); n = 0
    with torch.no_grad():
        for cond, feat in loader:
            cond, feat = cond.to(device), feat.to(device)
            x_hat, _, _ = model(feat, cond)
            sq += ((x_hat - feat).cpu().numpy()**2).sum(axis=0)
            n  += len(feat)
    mse = sq / n
    res = {name: float(mse[i]) for i, name in enumerate(FEATURE_NAMES)}
    res["mean_mse"] = float(mse.mean()); res["mean_rmse"] = float(np.sqrt(mse.mean()))
    return res


def generation_success_rate(model, problem, cfg, lookup_table,
                             n_samples_per_X=500, temperature=1.0,
                             lp_tol=0.05, device=None):
    if device is None: device = next(model.parameters()).device
    model.eval()
    allowed = problem.allowed_mats or MATERIAL_KEYS
    per_mat: Dict = {}
    totals = dict(n_gen=0, n_matched=0, n_feasible=0)

    for mat_key in allowed:
        mat = MATERIAL_REGISTRY[mat_key].ply_props
        per_X: Dict = {}
        for X in range(max(cfg.X_min, 10), cfg.X_max + 1):
            n_plies = 4 * X
            cn = make_condition_vector(
                Nx=-problem.Ncr_required*1.5, plate_a=problem.plate_a,
                plate_b=problem.plate_b, Ex_target=problem.Ex_min_GPa*1.5,
                Nombre_de_plis=float(n_plies), mat_key=mat_key,
                normalize=True, cfg=cfg,
            )
            ct = torch.tensor(cn, dtype=torch.float32, device=device)
            with torch.no_grad():
                lp_n = model.generate(ct, n_samples_per_X, temperature).cpu().numpy()
            lp_r   = denormalise_features(lp_n)
            ang, res = lookup_table.query(lp_r, X, k=1)
            valid  = res[:, 0] < lp_tol
            n_f    = 0
            for i in np.where(valid)[0]:
                try:
                    p = dd_properties(float(ang[i,0]), float(ang[i,1]),
                                       X, mat, problem.plate_a, problem.plate_b)
                    if p["Ex_MPa"]/1e3 >= problem.Ex_min_GPa and p["Ncr_Npmm"] >= problem.Ncr_required:
                        n_f += 1
                except Exception: pass
            nm = int(valid.sum())
            per_X[X] = dict(n_gen=n_samples_per_X, n_matched=nm, n_feasible=n_f,
                             match_rate=nm/n_samples_per_X,
                             feasibility_rate=n_f/n_samples_per_X)
            totals["n_gen"] += n_samples_per_X
            totals["n_matched"] += nm
            totals["n_feasible"] += n_f
        per_mat[mat_key] = per_X

    totals["overall_match_rate"]   = totals["n_matched"]  / totals["n_gen"]
    totals["overall_feasibility"]  = totals["n_feasible"] / totals["n_gen"]
    return dict(per_mat=per_mat, totals=totals)


def design_diversity(model, problem, cfg, lookup_table,
                     n_samples=3000, temperature=1.0, X=14,
                     mat_key="carbon", device=None):
    if device is None: device = next(model.parameters()).device
    model.eval()
    cn = make_condition_vector(Nx=-problem.Ncr_required*1.3, plate_a=problem.plate_a,
                                plate_b=problem.plate_b, Ex_target=problem.Ex_min_GPa*1.5,
                                Nombre_de_plis=float(4*X), mat_key=mat_key,
                                normalize=True, cfg=cfg)
    ct = torch.tensor(cn, dtype=torch.float32, device=device)
    with torch.no_grad():
        lp_n = model.generate(ct, n_samples, temperature).cpu().numpy()
    ang, _ = lookup_table.query(denormalise_features(lp_n), X, k=1)
    return dict(X=X, mat_key=mat_key,
                a_mean=float(ang[:,0].mean()), a_std=float(ang[:,0].std()),
                b_mean=float(ang[:,1].mean()), b_std=float(ang[:,1].std()),
                a_range=float(ang[:,0].max()-ang[:,0].min()),
                b_range=float(ang[:,1].max()-ang[:,1].min()))


def run_full_evaluation(model, df_train, df_val, problem, cfg,
                         lookup_table, device=None, verbose=True):
    if device is None: device = next(model.parameters()).device
    report: Dict = {}
    if verbose: print("=> R² encodeur ...")
    report["encoder_r2"] = encoder_r2(model, df_val, cfg, device=device)
    if verbose: print("=> MSE reconstruction ...")
    report["recon_mse"]  = reconstruction_mse(model, df_val, cfg, device=device)
    if verbose: print("=> Feasibility rate ...")
    report["generation"] = generation_success_rate(
        model, problem, cfg, lookup_table,
        n_samples_per_X=200,temperature=1, device=device)
    if verbose: print("=> Angular diversity ...")
    # Diversity for the carbon material at X=14
    report["diversity"] = design_diversity(
        model, problem, cfg, lookup_table,
        n_samples=2000, X=14, mat_key="carbon", device=device)
    if verbose: _print_report(report)
    return report


def _print_report(report):
    print("\n" + "="*58); print("CVAE EVALUATION REPORT"); print("="*58)
    r2 = report["encoder_r2"]
    print(f"\nR² reconstruction = {r2['mean_recon_r2']:.4f}  |  "
          f"R² latent μ = {r2['mean_mu_r2']:.4f}")
    print(f"  {'LP':8s}  {'R²_recon':>10s}  {'R²_μ':>10s}")
    for n in FEATURE_NAMES:
        print(f"  {n:8s}  {r2.get(n+'_recon',0):10.4f}  {r2.get(n+'_mu',0):10.4f}")
    mse = report["recon_mse"]
    print(f"\nReconstruction  MSE={mse['mean_mse']:.5f}  RMSE={mse['mean_rmse']:.4f}")
    tot = report["generation"]["totals"]
    print(f"\nMatching rate LP   : {tot['overall_match_rate']*100:.1f}%")
    print(f"Feasibility rate   : {tot['overall_feasibility']*100:.1f}%")
    div = report["diversity"]
    print(f"\nDiversity (X={div['X']}, {div['mat_key']}) :")
    print(f"  a : {div['a_mean']:.1f}° ± {div['a_std']:.1f}°  range {div['a_range']:.1f}°")
    print(f"  b : {div['b_mean']:.1f}° ± {div['b_std']:.1f}°  range {div['b_range']:.1f}°")
    print("="*58)