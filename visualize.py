"""
visualize.py  -  multi-material visualization
Graphs, Miki chart, LP parity, Pareto chart, heatmaps, material comparison.

New functions :
  plot_miki_trajectory          - LP generated at different temperatures on the Miki diagram
  plot_polar_stiffness          - Polar stiffness Ex(phi) for one or more designs
  plot_convergence_mass_penalty - Convergence of the physical penalty (Miki) per epoch
  plot_top_materials_projection - 2D projection of the best materials

Bonus features:
  plot_feasibility_heatmap      - Feasibility rate (X × material) as a heatmap
  plot_lp_generation_quality    - Distribution of generated LPs vs. training data
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np, pandas as pd, torch

from config import Config, FEATURE_NAMES
from cvae import CVAE
from dataset_dd import DDLaminateDataset, denormalise_features, make_condition_vector
from material_registry import MATERIAL_KEYS, get_mat
from optimize import DDProblem


def _dir(path):
    if path: Path(path).parent.mkdir(parents=True, exist_ok=True)


# ── 1. Training curves ────────────────────────────────────────────────

def plot_training_curves(log_csv, save_path=None):
    df  = pd.read_csv(log_csv)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(df["epoch"], df["train_loss"], label="Train")
    ax[0].plot(df["epoch"], df["val_loss"],   label="Val", ls="--")
    ax[0].set_title("Total Loss"); ax[0].legend()
    ax[1].plot(df["epoch"], df["train_recon"], label="Train")
    ax[1].plot(df["epoch"], df["val_recon"],   label="Val", ls="--")
    ax[1].set_title("Recon Loss"); ax[1].legend()
    ax[2].plot(df["epoch"], df["train_kl"],  label="Train KL")
    ax[2].plot(df["epoch"], df["val_kl"],    label="Val KL",   ls="--")
    ax[2].plot(df["epoch"], df["beta"],      label="β",        ls=":")
    ax[2].set_title("KL & β"); ax[2].legend()
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# ── 2. LP Parity ─────────────────────────────────────────────────────────────

def plot_lp_parity(model, df, cfg, save_path=None, n_samples=3000):
    device  = next(model.parameters()).device
    dataset = DDLaminateDataset(df.sample(min(n_samples, len(df)), random_state=0))
    model.eval()
    with torch.no_grad():
        x_hat, mu, _ = model(dataset.feat.to(device), dataset.cond.to(device))
    x_hat_np = x_hat.cpu().numpy(); feat_np = dataset.feat.numpy()
    n = len(FEATURE_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        t  = feat_np[:, i]; p = x_hat_np[:, i]
        ax.scatter(t, p, s=3, alpha=0.3, c=feat_np[:, 0], cmap="viridis")
        lim = max(abs(t).max(), abs(p).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        r2 = 1 - np.var(t-p) / (np.var(t) + 1e-9)
        ax.set_title(f"{name}  R²={r2:.4f}"); ax.set_xlabel("True"); ax.set_ylabel("Pred")
    fig.suptitle("LP Parity Plot", fontsize=13); fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# ── 3. Miki Diagram ─────────────────────────────────────────────────────

def plot_miki_coverage(df_train, df_generated=None, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    xi1 = (df_train["f_xi1_A_raw"] if "f_xi1_A_raw" in df_train.columns
           else df_train["f_xi1_A"])
    xi2 = (df_train["f_xi2_A_raw"] if "f_xi2_A_raw" in df_train.columns
           else df_train["f_xi2_A"])
    sc = ax.scatter(xi1, xi2, s=3, c=df_train["n_plies"], cmap="plasma",
                    alpha=0.3, label="Training data", zorder=1)
    plt.colorbar(sc, ax=ax, label="n_plies")
    if df_generated is not None and not df_generated.empty:
        xi1g = df_generated.get("xi1_A", pd.Series(np.zeros(len(df_generated))))
        xi2g = df_generated.get("xi2_A", pd.Series(np.zeros(len(df_generated))))
        ax.scatter(xi1g, xi2g, s=25, c="red", marker="*",
                   label="Generated feasible", zorder=3)
    th = np.linspace(-1, 1, 500)
    ax.fill_between(th, 2*th**2-1, np.ones(500), alpha=0.08, color="green")
    ax.plot(th, 2*th**2-1, "g-", lw=1, label="Miki boundary")
    ax.set_xlabel(r"$\xi^A_1$", fontsize=13); ax.set_ylabel(r"$\xi^A_2$", fontsize=13)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_title("Miki Diagram - LP Coverage"); ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# ── 4. Pareto Front multi-material ──────────────────────────────────────

def plot_pareto_front(df_feasible, problem, save_path=None):
    if df_feasible.empty: print("No feasible design."); return
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, df_feasible["mat_key"].nunique()))
    for (key, grp), col in zip(df_feasible.groupby("mat_key"), colors):
        ax.scatter(grp["mass_kg"]*1e3, grp["Ncr_Npmm"], s=10, alpha=0.6,
                   color=col, label=key)
    ax.axhline(problem.Ncr_required, color="r", ls="--", lw=1.5,
               label=f"Ncr req {problem.Ncr_required} N/mm")
    ax.set_xlabel("Masse [g]"); ax.set_ylabel("Ncr [N/mm]")
    ax.set_title("Feasible designs - Mass vs. Buckling Resistance")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# ── 5. Heatmaps Ex / Ncr in space (a, b) ───────────────────────────────

def plot_design_space(X, mat, problem, cfg, n_grid=60, save_path=None):
    from materials import dd_properties as ddp
    a_v = np.linspace(0, 90, n_grid); b_v = np.linspace(0, 90, n_grid)
    AA, BB = np.meshgrid(a_v, b_v)
    Ex_g = np.full_like(AA, np.nan); Ncr_g = np.full_like(AA, np.nan)
    for i in range(n_grid):
        for j in range(n_grid):
            try:
                p = ddp(AA[i,j], BB[i,j], X, mat, problem.plate_a, problem.plate_b)
                Ex_g[i,j] = p["Ex_MPa"]/1e3; Ncr_g[i,j] = p["Ncr_Npmm"]
            except Exception: pass
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, req, unit in zip(axes, [Ex_g, Ncr_g],
        [f"Ex [GPa] (X={X})", f"Ncr [N/mm] (X={X})"],
        [problem.Ex_min_GPa, problem.Ncr_required], ["GPa","N/mm"]):
        im = ax.pcolormesh(AA, BB, data, cmap="RdYlGn", shading="auto")
        cs = ax.contour(AA, BB, data, levels=[req], colors="blue", linewidths=2)
        ax.clabel(cs, fmt=f"req={req}{unit}", fontsize=7)
        plt.colorbar(im, ax=ax); ax.set_xlabel("a [°]"); ax.set_ylabel("b [°]")
        ax.set_title(title)
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# ── 6. Material comparison ─────────────────────────────────────────────

def plot_material_comparison(df_feasible, problem, save_path=None):
    if df_feasible.empty: print("No feasible design."); return

    summary = (df_feasible.groupby("mat_key")
               .agg(n_feasible=("mass_kg","count"),
                    min_mass=("mass_kg","min"),
                    max_Ex=("Ex_GPa","max"),
                    max_Ncr=("Ncr_Npmm","max"))
               .reset_index()
               .sort_values("min_mass"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mats = summary["mat_key"].tolist()
    x    = np.arange(len(mats))

    axes[0].bar(x, summary["min_mass"]*1e3, color="steelblue")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(mats, rotation=35, ha="right")
    axes[0].set_title("Minimum feasible mass [g]")
    axes[0].set_ylabel("Mass [g]")

    axes[1].bar(x, summary["max_Ex"], color="tomato")
    axes[1].axhline(problem.Ex_min_GPa, color="k", ls="--", lw=1,
                    label=f"Ex req {problem.Ex_min_GPa} GPa")
    axes[1].set_xticks(x); axes[1].set_xticklabels(mats, rotation=35, ha="right")
    axes[1].set_title("Ex max reached [GPa]"); axes[1].legend()

    axes[2].bar(x, summary["n_feasible"], color="mediumseagreen")
    axes[2].set_xticks(x); axes[2].set_xticklabels(mats, rotation=35, ha="right")
    axes[2].set_title("Number of feasible designs")

    fig.suptitle("Material comparison - Feasible designs", fontsize=13)
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# ── 7. Latent space PCA ──────────────────────────────────────────────────────

def plot_latent_space(model, df, cfg, color_by="mat_idx",
                      save_path=None, n_samples=5000):
    try: from sklearn.decomposition import PCA
    except ImportError: print("pip install scikit-learn"); return
    device  = next(model.parameters()).device
    sub     = df.sample(min(n_samples, len(df)), random_state=1)
    dataset = DDLaminateDataset(sub); model.eval()
    with torch.no_grad():
        mu, _ = model.encode(dataset.feat.to(device), dataset.cond.to(device))
    z = PCA(n_components=2).fit_transform(mu.cpu().numpy())
    c = sub[color_by].values if color_by in sub.columns else None
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z[:,0], z[:,1], s=4, c=c, cmap="tab20", alpha=0.6)
    if c is not None: plt.colorbar(sc, ax=ax, label=color_by)
    ax.set_title("Latent space (PCA 2D, colored by material)")
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# Others functions 
# ── 8. Miki Trajectory : LP generated at different temperatures ───────────────

def plot_miki_trajectory(
    model: CVAE,
    problem: DDProblem,
    cfg: Config,
    lt,
    temperatures: List[float] = None,
    n_samples: int = 800,
    mat_key: str = None,
    X: int = None,
    device=None,
    save_path: str = None,
):
    """
    Plots on the Miki diagram the (ksi¹_A, ksi²_A) and (ksi¹_D, ksi²_D)
    the LP generated by the CVAE at different temperatures.

    The points are colored as follows:
      - green: inside the Miki boundary (geometrically feasible)
      - red: outside the Miki boundary (geometrically infeasible)

    Parameters
    ----------
    temperatures : list of temperature values (default [0.3, 0.7, 1.0, 1.5, 2.0])
    mat_key      : material to use (default: first allowed material)
    X            : repeated blocks (default: X_max of cfg)
    """
    if device is None:
        device = next(model.parameters()).device
    if temperatures is None:
        temperatures = [0.3, 0.7, 1.0, 1.5, 2.0]
    if mat_key is None:
        mat_key = (problem.allowed_mats or MATERIAL_KEYS)[0]
    if X is None:
        X = cfg.X_max

    model.eval()
    n_T = len(temperatures)

    # Miki's Border : ksi² ≥ 2ksi¹² - 1
    xi1_line = np.linspace(-1, 1, 400)
    miki_bnd  = 2 * xi1_line**2 - 1

    fig, axes = plt.subplots(2, n_T, figsize=(4 * n_T, 8))
    if n_T == 1:
        axes = axes[:, np.newaxis]

    cond_norm = make_condition_vector(
        Nx=-problem.Ncr_required, plate_a=problem.plate_a,
        plate_b=problem.plate_b, Ex_target=problem.Ex_min_GPa,
        Nombre_de_plis=float(4 * X), mat_key=mat_key,
        normalize=True, cfg=cfg,
    )
    ct = torch.tensor(cond_norm, dtype=torch.float32, device=device)

    for col, T in enumerate(temperatures):
        with torch.no_grad():
            lp_norm = model.generate(ct, n_samples=n_samples,
                                      temperature=T).cpu().numpy()
        lp_raw = denormalise_features(lp_norm)  # shape (N, 4): ksi¹A ksi²A ksi¹D ksi²D

        for row, (idx1, idx2, lbl1, lbl2) in enumerate([
            (0, 1, r"$\xi^A_1$", r"$\xi^A_2$"),
            (2, 3, r"$\xi^D_1$", r"$\xi^D_2$"),
        ]):
            ax = axes[row][col]
            xi1 = lp_raw[:, idx1]
            xi2 = lp_raw[:, idx2]

            # Miki Feasibility Study : ksi² ≥ 2ksi¹² - 1
            feasible = xi2 >= (2 * xi1**2 - 1)
            rate = feasible.mean() * 100

            ax.scatter(xi1[~feasible], xi2[~feasible], s=6, alpha=0.3,
                       color="tomato", label=f"Infaisable ({(~feasible).sum()})")
            ax.scatter(xi1[feasible],  xi2[feasible],  s=6, alpha=0.4,
                       color="seagreen", label=f"Faisable ({feasible.sum()})")

            # Miki's Border
            ax.plot(xi1_line, miki_bnd, "k-", lw=1.2, label="Miki")
            ax.fill_between(xi1_line, miki_bnd, 1, alpha=0.06, color="green")

            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(lbl1, fontsize=10); ax.set_ylabel(lbl2, fontsize=10)
            title_row = "In-plane (A)" if row == 0 else "Bending (D)"
            ax.set_title(f"T={T:.1f}  |  {title_row}\nMiki OK: {rate:.0f}%",
                         fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"LP trajectories generated in Miki's space\n"
        f"mat={mat_key}  X={X}  n={n_samples}/T",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150)
    plt.show()


# ── 9. Polar Stiffness Ex(phi) ────────────────────────────────────────────────

def plot_polar_stiffness(
    best_design: Dict,
    problem: DDProblem,
    cfg: Config,
    compare_designs: List[Dict] = None,
    save_path: str = None,
):
    """
    Plots the polar diagram of the stiffness Ex(phi) and Ey(phi) for one or multiple designs.

    Parameters
    ----------
    best_design     : dict with keys a_deg, b_deg, X, mat_key
    compare_designs : list of additional dicts (optional)
    """
    from materials import dd_sequence, compute_ABD

    phi = np.linspace(0, 2 * np.pi, 720)
    cos_p = np.cos(phi); sin_p = np.sin(phi)

    def _ex_phi(A, h):
        """"Ex(phi) in GPa via rotation of the compliance tensor."""
        Ai = np.linalg.inv(A)
        a11, a22, a12, a66 = Ai[0,0], Ai[1,1], Ai[0,1], Ai[2,2]
        c2 = cos_p**2; s2 = sin_p**2
        # Transformed compliance a11'(phi) = Ai[0,0]*c4 + Ai[1,1]*s4 + (2Ai[0,1]+Ai[2,2])*c2*s2
        a11_rot = a11 * c2**2 + a22 * s2**2 + (2*a12 + a66) * c2 * s2
        return 1.0 / (h * a11_rot) / 1e3  # MPa => GPa

    designs = [best_design]
    if compare_designs:
        designs.extend(compare_designs)

    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="polar")

    colors = plt.cm.tab10(np.linspace(0, 1, len(designs)))
    max_ex = 0.0

    for d, col in zip(designs, colors):
        a = d["a_deg"]; b = d["b_deg"]; X = d["X"]
        mk = d.get("mat_key", "carbon")
        mat = get_mat(mk).ply_props
        seq = dd_sequence(a, b, X)
        abd = compute_ABD(seq, mat, mat.t)
        ex  = _ex_phi(abd["A"], abd["h"])
        max_ex = max(max_ex, ex.max())

        label = f"[{a:.0f}°/-{a:.0f}°/{b:.0f}°/-{b:.0f}°]_{X}  {mk}"
        ax.plot(phi, ex, lw=2.0, color=col, label=label)
        ax.fill(phi, ex, alpha=0.06, color=col)

    # Draw a circle at Ex_min_GPa if accessible
    if problem.Ex_min_GPa <= max_ex * 1.05:
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(theta, np.full_like(theta, problem.Ex_min_GPa),
                "r--", lw=1.2, label=f"Ex req. {problem.Ex_min_GPa} GPa")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("Polar rigidity Ex(phi) [GPa]", fontsize=13, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.45, -0.05), fontsize=8)

    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── 10. Convergence of Physical Penalties (Miki) ──────────────────────────

def plot_convergence_mass_penalty(
    log_csv: str,
    save_path: str = None,
):
    """
    Displays the convergence of:
        - total loss (training / validation)
        - Miki physical penalty (training / validation)
        - reconstruction loss (training / validation)
        - β and lam_phys over the epochs

    Useful for diagnosing the effect of enabling penalties after the warm-up.
    """
    df = pd.read_csv(log_csv)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ── a. Total loss ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(df["epoch"], df["train_loss"], label="Train", lw=1.5)
    ax0.plot(df["epoch"], df["val_loss"],   label="Val",   lw=1.5, ls="--")
    ax0.set_title("Total loss"); ax0.set_xlabel("Epoch")
    ax0.legend(); ax0.set_yscale("log")

    # ── b. Reconstruction ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(df["epoch"], df["train_recon"], label="Train", lw=1.5)
    ax1.plot(df["epoch"], df["val_recon"],   label="Val",   lw=1.5, ls="--")
    ax1.set_title("Reconstruction MSE"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.set_yscale("log")

    # ── c. KL divergence ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df["epoch"], df["train_kl"], label="Train KL", lw=1.5)
    ax2.plot(df["epoch"], df["val_kl"],   label="Val KL",   lw=1.5, ls="--")
    ax2_b = ax2.twinx()
    ax2_b.plot(df["epoch"], df["beta"], color="gray", lw=1, ls=":", label="β")
    ax2_b.set_ylabel("β", color="gray"); ax2_b.tick_params(axis="y", colors="gray")
    ax2.set_title("KL + β annealing"); ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper left"); ax2.set_yscale("log")

    # ── d. Physical penalty (Miki) ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if "train_phys" in df.columns:
        # Mask the null values (before warm-up activation)
        tr_phys = df["train_phys"].replace(0, np.nan)
        va_phys = df["val_phys"].replace(0, np.nan) if "val_phys" in df.columns else None
        ax3.plot(df["epoch"], tr_phys, label="Train Miki", lw=1.5, color="darkorange")
        if va_phys is not None:
            ax3.plot(df["epoch"], va_phys, label="Val Miki", lw=1.5, ls="--",
                     color="darkorange", alpha=0.6)

        # Vertical line at the beginning of the warm-up
        first_nonzero = df.loc[tr_phys.notna(), "epoch"]
        if len(first_nonzero) > 0:
            ax3.axvline(first_nonzero.iloc[0], color="red", ls="--", lw=1,
                        label=f"Activation (epoch {first_nonzero.iloc[0]})")
        ax3.set_yscale("log")
    ax3.set_title("Physical penalty (Miki)"); ax3.set_xlabel("Epoch")
    ax3.legend(fontsize=8)

    # ── e. lambda_phys schedule ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if "lam_phys" in df.columns:
        ax4.plot(df["epoch"], df["lam_phys"], color="purple", lw=1.5, label="lambda_phys")
    if "lam_reg" in df.columns:
        ax4.plot(df["epoch"], df["lam_reg"], color="steelblue", lw=1.5,
                 ls="--", label="lambda_reg")
    ax4.set_title("Penalty schedules lambda"); ax4.set_xlabel("Epoch")
    ax4.legend()

    # ── f. Ratio phys / recon (stabilité) ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if "train_phys" in df.columns and "train_recon" in df.columns:
        ratio = (df["train_phys"] / (df["train_recon"] + 1e-10)).replace(0, np.nan)
        ax5.plot(df["epoch"], ratio, color="tomato", lw=1.3, label="phys / recon")
        ax5.axhline(1.0, color="k", lw=0.8, ls="--", label="ratio=1")
        ax5.set_yscale("log")
    ax5.set_title("Penalty balance / reconstruction"); ax5.set_xlabel("Epoch")
    ax5.legend(fontsize=8)

    fig.suptitle("CVAE Training Convergence - Penalty Analysis",
                 fontsize=13, y=1.01)
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── 11. 2D rendering of the best materials ────────────────────────────────

def plot_top_materials_projection(
    df_feasible: pd.DataFrame,
    problem: DDProblem,
    top_n: int = 6,
    save_path: str = None,
):
    """
    Synthetic view of the best materials on 4 axes :
      (a) Scatter minimal mass vs max Ncr — size ∝ n_feasible designs
      (b) Scatter minimal mass vs max Ex
      (c) Radar of normalized properties (mass, Ex, Ncr, n_feasible)
      (d) Distribution of minimal mass by X for the top-N materials

    Parameters
    ----------
    df_feasible: DataFrame returned by solve_dd_problem (brute_force or cvae)
    top_n: number of materials to highlight
    """
    if df_feasible.empty:
        print("No feasible design."); return

    # ── Aggregation by material ────────────────────────────────────────────
    summary = (
        df_feasible.groupby("mat_key")
        .agg(
            n_feasible=("mass_kg",  "count"),
            min_mass=  ("mass_kg",  "min"),
            max_Ex=    ("Ex_GPa",   "max"),
            max_Ncr=   ("Ncr_Npmm", "max"),
            min_X=     ("X",        "min"),
        )
        .reset_index()
        .sort_values("min_mass")
        .head(top_n)
    )

    n = len(summary)
    colors = plt.cm.tab10(np.arange(n) / max(n - 1, 1))
    mat_names = summary["mat_key"].tolist()

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── a. Masse vs Ncr ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    sz  = 100 + 500 * (summary["n_feasible"] / summary["n_feasible"].max())
    sc0 = ax0.scatter(summary["min_mass"] * 1e3, summary["max_Ncr"],
                      s=sz, c=colors, alpha=0.85, edgecolors="k", lw=0.5, zorder=3)
    for _, row in summary.iterrows():
        ax0.annotate(row["mat_key"],
                     (row["min_mass"] * 1e3, row["max_Ncr"]),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8.5, fontweight="bold")
    ax0.axhline(problem.Ncr_required, color="r", ls="--", lw=1.2,
                label=f"Ncr req. {problem.Ncr_required:.0f} N/mm")
    ax0.set_xlabel("Minimal mass [g]"); ax0.set_ylabel("Max Ncr [N/mm]")
    ax0.set_title("Minimal mass vs. Buckling resistance\n(size = n feasible designs)")
    ax0.legend(fontsize=8)

    # ── b. Mass vs Ex ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(summary["min_mass"] * 1e3, summary["max_Ex"],
                s=sz, c=colors, alpha=0.85, edgecolors="k", lw=0.5, zorder=3)
    for _, row in summary.iterrows():
        ax1.annotate(row["mat_key"],
                     (row["min_mass"] * 1e3, row["max_Ex"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax1.axhline(problem.Ex_min_GPa, color="b", ls="--", lw=1.2,
                label=f"Ex req. {problem.Ex_min_GPa} GPa")
    ax1.set_xlabel("Minimal mass [g]"); ax1.set_ylabel("Max Ex [GPa]")
    ax1.set_title("Minimal mass vs. Longitudinal stiffness\n(size = n feasible designs)")
    ax1.legend(fontsize=8)

    # ── c. Standard radar ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0], projection="polar")

    metrics = ["min_mass_inv", "max_Ex", "max_Ncr", "n_feasible"]
    labels  = ["Lightweight\n(1/mass)", "Max Ex\n[GPa]",
                "Max Ncr\n[N/mm]", "Feasible designs\n(count)"]
    n_met   = len(metrics)
    angles  = np.linspace(0, 2 * np.pi, n_met, endpoint=False).tolist()
    angles += angles[:1]

    # Normalization on [0, 1]
    radar_df = summary.copy()
    radar_df["min_mass_inv"] = 1.0 / (radar_df["min_mass"] + 1e-12)
    for m in metrics:
        col_max = radar_df[m].max()
        radar_df[m] = radar_df[m] / (col_max + 1e-12)

    for (_, row), col in zip(radar_df.iterrows(), colors):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        ax2.plot(angles, vals, lw=1.8, color=col, label=row["mat_key"])
        ax2.fill(angles, vals, alpha=0.08, color=col)

    ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0]); ax2.set_yticklabels(["25%","50%","75%","100%"],
                                                                    fontsize=6)
    ax2.set_title("Radar des performances normalisées\n(top matériaux)", fontsize=10, pad=18)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.55, -0.05), fontsize=8)

    # ── d. Min mass by X (rows by material) ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    top_keys = summary["mat_key"].tolist()
    sub = df_feasible[df_feasible["mat_key"].isin(top_keys)]

    for mk, col in zip(top_keys, colors):
        grp = sub[sub["mat_key"] == mk].groupby("X")["mass_kg"].min() * 1e3
        if grp.empty: continue
        ax3.plot(grp.index, grp.values, marker="o", ms=4,
                 lw=1.8, color=col, label=mk)

    ax3.set_xlabel("Blocs X (n_plies = 4X)"); ax3.set_ylabel("Minimum mass [g]")
    ax3.set_title("Evolution of the minimum feasible mass vs X")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"Projection multi-matériaux — Top {top_n} designs faisables",
        fontsize=13, y=1.01,
    )
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()



# ── 12. Feasibility Heatmap: X × Material ──────────────────────────────────

def plot_feasibility_heatmap(
    df_feasible: pd.DataFrame,
    cfg: Config,
    metric: str = "n_feasible",
    save_path: str = None,
):
    """
    Heatmap (material × X) of the number of feasible designs or the minimum mass.

    Parameters
    ----------
    metric : "n_feasible"  => number of feasible designs per cell
             "min_mass_g"  => minimum mass [g] per cell
             "max_Ncr"     => Ncr max per cell
    """
    if df_feasible.empty:
        print("No feasible design.s"); return

    all_X    = list(range(cfg.X_min, cfg.X_max + 1))
    all_mats = df_feasible["mat_key"].unique().tolist()

    # Complete grid (NaN if no designs)
    if metric == "n_feasible":
        pivot = (df_feasible.groupby(["mat_key", "X"])
                 .size().reset_index(name="val")
                 .pivot(index="mat_key", columns="X", values="val"))
        cmap  = "YlGn"; fmt = ".0f"; title_suffix = "Number of feasible designs"
    elif metric == "min_mass_g":
        pivot = (df_feasible.groupby(["mat_key", "X"])["mass_kg"]
                 .min().reset_index()
                 .assign(val=lambda d: d["mass_kg"] * 1e3)
                 .pivot(index="mat_key", columns="X", values="val"))
        cmap  = "RdYlGn_r"; fmt = ".1f"; title_suffix = "Minimum mass [g]"
    elif metric == "max_Ncr":
        pivot = (df_feasible.groupby(["mat_key", "X"])["Ncr_Npmm"]
                 .max().reset_index()
                 .pivot(index="mat_key", columns="X", values="Ncr_Npmm"))
        cmap  = "Blues"; fmt = ".0f"; title_suffix = "Ncr max [N/mm]"
    else:
        raise ValueError(f"metric inconnu : {metric}")

    # Re-indexing to cover all Xs in the configuration
    pivot = pivot.reindex(columns=all_X)

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(10, len(all_X) * 0.6 + 2),
                                    max(5, len(all_mats) * 0.5 + 1.5)))
    mask = pivot.isna()
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, mask=mask,
        annot=True, fmt=fmt, linewidths=0.5, linecolor="white",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("Blocs X (n_plies = 4X)", fontsize=11)
    ax.set_ylabel("Material", fontsize=11)
    ax.set_title(f"Heatmap of feasibility — {title_suffix}", fontsize=13)
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── 13. Quality of LP generation: generated distributions vs. dataset ──────────

def plot_lp_generation_quality(
    model: CVAE,
    df_train: pd.DataFrame,
    problem: DDProblem,
    cfg: Config,
    mat_key: str = None,
    X: int = None,
    n_generated: int = 4000,
    temperature: float = 1.0,
    device=None,
    save_path: str = None,
):
    """
    For each LP (ksi¹_A, ksi²_A, ksi¹_D, ksi²_D), compare the following in an overlay:
      - the actual distribution in the training dataset
      - the distribution generated by the CVAE

    Also display the KS-test statistics (degree of similarity).
    """
    from scipy.stats import ks_2samp

    if device is None:
        device = next(model.parameters()).device
    if mat_key is None:
        mat_key = (problem.allowed_mats or MATERIAL_KEYS)[0]
    if X is None:
        X = cfg.X_max

    model.eval()
    cond_norm = make_condition_vector(
        Nx=-problem.Ncr_required, plate_a=problem.plate_a,
        plate_b=problem.plate_b, Ex_target=problem.Ex_min_GPa,
        Nombre_de_plis=float(4 * X), mat_key=mat_key,
        normalize=True, cfg=cfg,
    )
    ct = torch.tensor(cond_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        lp_gen = model.generate(ct, n_samples=n_generated,
                                  temperature=temperature).cpu().numpy()
    lp_gen = denormalise_features(lp_gen)

    # Actual data filtered by mat_key and X
    sub = df_train[(df_train["mat_key"] == mat_key) & (df_train["X_blocks"] == X)]
    if len(sub) < 20:
        sub = df_train[df_train["mat_key"] == mat_key]  # X-filter-less fallback
    if sub.empty:
        sub = df_train  # fallback total

    raw_cols = [f"f_{n}_raw" for n in FEATURE_NAMES]
    lp_real  = sub[[c for c in raw_cols if c in sub.columns]].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    lp_names = [r"$\xi^A_1$", r"$\xi^A_2$", r"$\xi^D_1$", r"$\xi^D_2$"]
    palettes  = [("steelblue", "tomato"), ("mediumseagreen", "darkorange"),
                 ("mediumpurple", "gold"), ("teal", "crimson")]

    for i, (ax, name, (c_real, c_gen)) in enumerate(
        zip(axes.flat, lp_names, palettes)
    ):
        real_vals = lp_real[:, i] if lp_real.shape[1] > i else np.array([])
        gen_vals  = lp_gen[:, i]

        bins = np.linspace(-1.05, 1.05, 60)
        if len(real_vals) > 0:
            ax.hist(real_vals, bins=bins, density=True, alpha=0.55,
                    color=c_real, label=f"Dataset ({len(real_vals)})")
        ax.hist(gen_vals, bins=bins, density=True, alpha=0.55,
                color=c_gen, label=f"Generated ({len(gen_vals)})")

        if len(real_vals) > 5:
            stat, pval = ks_2samp(real_vals, gen_vals)
            ks_txt = f"KS={stat:.3f}  p={pval:.3f}"
        else:
            ks_txt = "(not enough real data)"

        ax.set_title(f"{name}  |  {ks_txt}", fontsize=10)
        ax.set_xlabel("Valeur LP"); ax.set_ylabel("Densité")
        ax.legend(fontsize=8)
        ax.axvline(0, color="k", lw=0.6, ls=":")

    fig.suptitle(
        f"Generation quality LP — mat={mat_key}  X={X}  T={temperature:.1f}",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path: _dir(save_path); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()