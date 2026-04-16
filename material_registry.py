"""
material_registry.py
====================
Bridge between `properties.py` (material catalog with properties)
and `materials.py` (PlyProperties class for CLT).

Provides :
  MATERIAL_REGISTRY : dict  {key: MatEntry}
  MATERIAL_KEYS     : list of ordered keys  => index for the one-hot
  N_MAT             : int  = len(MATERIAL_KEYS)

Each MatEntry exposes :
  .ply_props  : PlyProperties (materials.py) ready for CLT
  .data       : PlyProperties (properties.py) with strengths
  .name       : str
  .t_ply_mm   : float  unit thickness of the ply [mm]
  .rho_kg_mm3 : float  density [kg/mm³]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# CLT PlyProperties (no strength fields, has 't')
from materials import PlyProperties as CltPly

# Materials Catalog (with resistors, without 't'))
import properties as P


# ---------------------------------------------------------------------------
# MatEntry : contains all the information about a material
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatEntry:
    key:        str
    name:       str
    t_ply_mm:   float         # fold thickness [mm]
    ply_props:  CltPly        # for CLT (E1, E2, G12, nu12, rho, t in MPa, kg/mm³)
    data:       object        # properties.PlyProperties (strengths, E in GPa)


def _make_entry(key: str, p, t_ply_mm: float) -> MatEntry:
    """Converts a properties.PlyProperties object to a MatEntry."""
    clt = CltPly(
        E1   = p.E1  * 1e3,        # GPa => MPa
        E2   = p.E2  * 1e3,
        G12  = p.G12 * 1e3,
        nu12 = p.nu12,
        rho  = p.rho * 1e-3,       # g/cm³ => kg/mm³  (1 g/cm³ = 1e-3 kg/mm³)
        t    = t_ply_mm,
    )
    return MatEntry(
        key       = key,
        name      = p.name,
        t_ply_mm  = t_ply_mm,
        ply_props = clt,
        data      = p,
    )

# ---------------------------------------------------------------------------
# Registre  (ordre fixe => index one-hot stable)
# ---------------------------------------------------------------------------

MATERIAL_REGISTRY: Dict[str, MatEntry] = {
    "im7":      _make_entry("im7",     P.IM7_977_3,              0.140),
    "carbon":   _make_entry("carbon",  P.CARBON_EPOXY,           0.125),
    "flax":     _make_entry("flax",    P.FLAX_EPOXY,             0.250),
    "glass":    _make_entry("glass",   P.GLASS_EPOXY,            0.250),
    "flax_pla": _make_entry("flax_pla",P.FLAX_PLA,               0.250),
    "hemp":     _make_entry("hemp",    P.HEMP_EPOXY,             0.250),
    "basalt":   _make_entry("basalt",  P.BASALT_EPOXY,           0.250),
    "rcarbon":  _make_entry("rcarbon", P.RECYCLED_CARBON_EPOXY,  0.125),
    "peek":     _make_entry("peek",    P.CARBON_PEEK,            0.140),
    "aramid":   _make_entry("aramid",  P.ARAMID_EPOXY,           0.130),
    "boron":    _make_entry("boron",   P.BORON_EPOXY,            0.130),
    "bamboo":   _make_entry("bamboo",  P.BAMBOO_EPOXY,           0.300),
}

MATERIAL_KEYS: List[str] = list(MATERIAL_REGISTRY.keys())
N_MAT: int = len(MATERIAL_KEYS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mat_index(key: str) -> int:
    """Key => integer index (position in MATERIAL_KEYS)."""
    return MATERIAL_KEYS.index(key)
def mat_one_hot(key: str) -> "np.ndarray":
    """Key => one-hot vector  shape (N_MAT,)  float32."""
    import numpy as np
    vec = np.zeros(N_MAT, dtype=np.float32)
    vec[mat_index(key)] = 1.0
    return vec
def mat_from_index(idx: int) -> MatEntry:
    return MATERIAL_REGISTRY[MATERIAL_KEYS[idx]]

def get_mat(key: str) -> MatEntry:
    return MATERIAL_REGISTRY[key]

# ---------------------------------------------------------------------------
# Quick summary
# ---------------------------------------------------------------------------
def print_summary() -> None:
    print(f"{'#':>2}  {'Key':10s}  {'Nom':35s}  "
          f"{'E1':>7s}  {'ρ':>5s}  {'t':>6s}")
    print("-" * 75)
    for i, key in enumerate(MATERIAL_KEYS):
        e = MATERIAL_REGISTRY[key]
        print(f"{i:>2}  {key:10s}  {e.name:35s}  "
              f"{e.data.E1:7.1f}  {e.data.rho:5.2f}  {e.t_ply_mm:6.3f}")

if __name__ == "__main__":
    print_summary()