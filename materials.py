"""
materials.py
============
Material stiffness invariants, CLT (ABD matrix),
lamination parameters, and effective engineering constants.

All functions work with plain NumPy - no PyTorch dependency.
Compatible with VAE_v1.py conventions (normalised z-coordinates).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

@dataclass
class PlyProperties:
    """
    Unidirectional ply properties.

    Parameters
    ----------
    E1, E2, G12 : float
        Moduli in MPa.
    nu12 : float
        Major Poisson's ratio.
    rho : float
        Density in kg/mm³  (e.g. 1600 kg/m³ => 1.6e-6 kg/mm³).
    t : float
        Ply thickness in mm.
    """

    E1:   float
    E2:   float
    G12:  float
    nu12: float
    rho:  float
    t:    float

    def __post_init__(self):
        nu21 = self.nu12 * self.E2 / self.E1
        D = 1.0 - self.nu12 * nu21
        self.Q11 = self.E1  / D
        self.Q22 = self.E2  / D
        self.Q12 = self.nu12 * self.E2 / D
        self.Q66 = self.G12

        # Tsai-Pagano material invariants (MPa)
        Q = self.Q11, self.Q12, self.Q22, self.Q66
        self.U1 = (3*Q[0] + 2*Q[1] + 3*Q[2] + 4*Q[3]) / 8
        self.U2 = (    Q[0]          -     Q[2]        ) / 2
        self.U3 = (    Q[0] - 2*Q[1] +     Q[2] - 4*Q[3]) / 8
        self.U4 = (    Q[0] + 6*Q[1] +     Q[2] - 4*Q[3]) / 8
        self.U5 = (    Q[0] - 2*Q[1] +     Q[2] + 4*Q[3]) / 8


# ---------------------------------------------------------------------------
# Transformed reduced stiffness  Q̄
# ---------------------------------------------------------------------------

def _Qbar(mat: PlyProperties, theta_deg: float) -> np.ndarray:
    """3×3 transformed reduced stiffness matrix for a ply at *theta_deg*."""
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    c2, s2 = c**2, s**2
    cs = c * s
    c4, s4 = c2**2, s2**2
    c2s2 = c2 * s2

    Q11, Q22, Q12, Q66 = mat.Q11, mat.Q22, mat.Q12, mat.Q66

    Qb = np.array([
        [Q11*c4 + 2*(Q12 + 2*Q66)*c2s2 + Q22*s4,
         (Q11 + Q22 - 4*Q66)*c2s2 + Q12*(c4 + s4),
         (Q11 - Q12 - 2*Q66)*c2*cs - (Q22 - Q12 - 2*Q66)*s2*cs],
        [(Q11 + Q22 - 4*Q66)*c2s2 + Q12*(c4 + s4),
         Q11*s4 + 2*(Q12 + 2*Q66)*c2s2 + Q22*c4,
         (Q11 - Q12 - 2*Q66)*s2*cs - (Q22 - Q12 - 2*Q66)*c2*cs],
        [(Q11 - Q12 - 2*Q66)*c2*cs - (Q22 - Q12 - 2*Q66)*s2*cs,
         (Q11 - Q12 - 2*Q66)*s2*cs - (Q22 - Q12 - 2*Q66)*c2*cs,
         (Q11 + Q22 - 2*Q12 - 2*Q66)*c2s2 + Q66*(c4 + s4)],
    ])
    return Qb


# ---------------------------------------------------------------------------
# ABD matrix
# ---------------------------------------------------------------------------

def compute_ABD(angles_deg: List[float],
                mat: PlyProperties,
                t: float) -> Dict[str, np.ndarray | float]:
    """
    Full ABD stiffness matrix for a laminate.

    Returns dict with keys "A", "B", "D" (3×3 arrays, MPa·mm, MPa,
    MPa/mm respectively) and "h" (total thickness, mm).
    """
    n = len(angles_deg)
    h = n * t
    z_edges = np.array([-h / 2 + k * t for k in range(n + 1)])

    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    for k, theta in enumerate(angles_deg):
        Qb = _Qbar(mat, theta)
        zk,  zk1 = z_edges[k], z_edges[k + 1]
        A += Qb * (zk1 - zk)
        B += Qb * (zk1**2 - zk**2) / 2
        D += Qb * (zk1**3 - zk**3) / 3

    return {"A": A, "B": B, "D": D, "h": h}


# ---------------------------------------------------------------------------
# Lamination parameters  (compatible with VAE_v1.py normalised formula)
# ---------------------------------------------------------------------------

def compute_lamination_parameters(angles_deg: List[float]) -> Dict[str, float]:
    """
    Lamination parameters using normalised (t=1) z-coordinates,
    consistent with VAE_v1.py:

        layer_heights = cumsum(ones(n)) - n/2  => [1-n/2, …, n/2]

    Returns dict with keys a1…a4, d1…d4 (ksi¹_A…ksi⁴_A, ksi¹_D…ksi⁴_D).
    """
    n = len(angles_deg)
    if n == 0:
        return {k: 0.0 for k in ("a1","a2","a3","a4","d1","d2","d3","d4")}

    lh = np.arange(1, n + 1, dtype=float) - n / 2          # layer_heights
    th = np.radians(np.array(angles_deg, dtype=float))

    cos2 = np.cos(2 * th)
    cos4 = np.cos(4 * th)
    sin2 = np.sin(2 * th)
    sin4 = np.sin(4 * th)

    dz  = lh - (lh - 1)                                      # always 1
    dz3 = lh**3 - (lh - 1)**3

    a1 = float(np.sum(cos2 * dz)  / n)
    a2 = float(np.sum(cos4 * dz)  / n)
    a3 = float(np.sum(sin2 * dz)  / n)
    a4 = float(np.sum(sin4 * dz)  / n)

    d1 = float(4 / n**3 * np.sum(cos2 * dz3))
    d2 = float(4 / n**3 * np.sum(cos4 * dz3))
    d3 = float(4 / n**3 * np.sum(sin2 * dz3))
    d4 = float(4 / n**3 * np.sum(sin4 * dz3))

    return {"a1": a1, "a2": a2, "a3": a3, "a4": a4,
            "d1": d1, "d2": d2, "d3": d3, "d4": d4}


# ---------------------------------------------------------------------------
# ABD from lamination parameters  (inverse formula, eqs. 2-3 in Sun 2025)
# ---------------------------------------------------------------------------

def ABD_from_LP(mat: PlyProperties,
                lp: Dict[str, float],
                h: float) -> Dict[str, np.ndarray]:
    """
    Reconstruct A and D matrices directly from lamination parameters
    and total laminate thickness h (mm).

    Uses Tsai-Pagano invariants (U1…U5).
    Assumes B = 0 (symmetric or mid-plane symmetric laminate assumed).
    ksi4 terms default to 0 if not provided.
    """
    U1, U2, U3, U4, U5 = mat.U1, mat.U2, mat.U3, mat.U4, mat.U5

    xi1A = lp.get("a1", 0.0); xi2A = lp.get("a2", 0.0)
    xi3A = lp.get("a3", 0.0); xi4A = lp.get("a4", 0.0)
    xi1D = lp.get("d1", 0.0); xi2D = lp.get("d2", 0.0)
    xi3D = lp.get("d3", 0.0); xi4D = lp.get("d4", 0.0)

    def _build(xi1, xi2, xi3, xi4, factor):
        A11 = factor * (U1 + U2*xi1 + U3*xi2)
        A22 = factor * (U1 - U2*xi1 + U3*xi2)
        A12 = factor * (U4           - U3*xi2)
        A66 = factor * (U5           - U3*xi2)
        A16 = factor * (U2*xi3/2     + U3*xi4)
        A26 = factor * (U2*xi3/2     - U3*xi4)
        return np.array([[A11, A12, A16],
                         [A12, A22, A26],
                         [A16, A26, A66]])

    A = _build(xi1A, xi2A, xi3A, xi4A, h)
    D = _build(xi1D, xi2D, xi3D, xi4D, h**3 / 12)
    B = np.zeros((3, 3))
    return {"A": A, "B": B, "D": D, "h": h}


# ---------------------------------------------------------------------------
# Engineering constants
# ---------------------------------------------------------------------------

def effective_engineering_constants(abd: Dict) -> Dict[str, float]:
    """
    Effective engineering constants from the A matrix.

    Under uniaxial stress σ_x:
      ε_x = A⁻¹[0,0] · N_x = A⁻¹[0,0] · σ_x · h
      Ex = σ_x / ε_x = 1 / (h · A⁻¹[0,0])

    Returns values in MPa (same unit as the Q_ij used to build A).
    """
    A = abd["A"]
    h = abd["h"]
    Ai = np.linalg.inv(A)
    Ex   = 1.0 / (h * Ai[0, 0])
    Ey   = 1.0 / (h * Ai[1, 1])
    Gxy  = 1.0 / (h * Ai[2, 2])
    nuxy = -Ai[0, 1] / Ai[0, 0]
    return {"Ex": Ex, "Ey": Ey, "Gxy": Gxy, "nuxy": nuxy}


# ---------------------------------------------------------------------------
# Critical buckling load  Ncr  (simply-supported, pure Nx compression)
# ---------------------------------------------------------------------------

def buckling_Ncr(abd: Dict,
                 plate_a: float,
                 plate_b: float,
                 n_modes: int = 6) -> float:
    """
    Critical buckling load (N/mm, positive value) for a simply-supported
    rectangular plate of dimensions *plate_a × plate_b* (mm) under
    uniaxial compression Nx.

    Uses the closed-form orthotropic formula (D16 ≈ D26 ≈ 0 for balanced
    laminates):

        Ncr(m) = (π/b)² [D11(mb/a)² + 2(D12+2D66) + D22(a/mb)²]

    Returns the minimum over m = 1 … n_modes.
    """
    D = abd["D"]
    D11 = D[0, 0]; D12 = D[0, 1]; D22 = D[1, 1]; D66 = D[2, 2]

    ncr_min = np.inf
    for m in range(1, n_modes + 1):
        r = (m * plate_b) / plate_a
        ncr = (np.pi / plate_b)**2 * (D11 * r**2
                                       + 2*(D12 + 2*D66)
                                       + D22 / r**2)
        if ncr < ncr_min:
            ncr_min = ncr

    return float(ncr_min)


# ---------------------------------------------------------------------------
# Helper: build DD laminate sequence [a/-a/b/-b]_X
# ---------------------------------------------------------------------------

def dd_sequence(a_deg: float, b_deg: float, X: int) -> List[float]:
    """
    Returns the stacking sequence for a Double-Double laminate:
        [a, -a, b, -b] repeated X times  =>  4X plies total.
    """
    block = [a_deg, -a_deg, b_deg, -b_deg]
    return block * int(X)


# ---------------------------------------------------------------------------
# Convenience: all properties of a DD laminate at once
# ---------------------------------------------------------------------------

def dd_properties(a_deg: float, b_deg: float, X: int,
                  mat: PlyProperties,
                  plate_a: float, plate_b: float,
                  n_modes: int = 6) -> Dict:
    """
    Compute all structural properties for [a/-a/b/-b]_X.

    Returns
    -------
    dict with keys: angles, h, lp, A, B, D, Ex_MPa, Ey_MPa, Ncr_Npmm
    """
    angles = dd_sequence(a_deg, b_deg, X)
    t = mat.t

    abd = compute_ABD(angles, mat, t)
    lp  = compute_lamination_parameters(angles)
    eng = effective_engineering_constants(abd)
    ncr = buckling_Ncr(abd, plate_a, plate_b, n_modes)

    return {
        "angles":   angles,
        "h_mm":     abd["h"],
        "lp":       lp,
        "A":        abd["A"],
        "B":        abd["B"],
        "D":        abd["D"],
        "Ex_MPa":   eng["Ex"],
        "Ey_MPa":   eng["Ey"],
        "Ncr_Npmm": ncr,
    }
