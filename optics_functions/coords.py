"""
Coordinate transformations
"""

import pandas as pd
import numpy as np
from optics_functions import rdts

FACTOR = 0.5

def cmplx_courant_snyder(df: pd.DataFrame,
                         plane: str) -> pd.DataFrame:
    if plane not in ['X', 'Y']:
        raise RuntimeError("`plane` must be one of `'X'`, `'Y'`")
    sqrt_b = np.sqrt(df[f"BET{plane}"])
    a = df[f"ALF{plane}"]
    x = df[plane] / sqrt_b
    px = x * a + sqrt_b * df[f"P{plane}"]

    return x + 1.0j*px


def h_from_z(zxp, zxm, zyp, zym, fterms: pd.DataFrame, order: int) -> pd.DataFrame:
    """Calculates the complex Courant-Snyder coordinates hxp, hyp from normal form coordinates

    Args:
        zxp ([type]): zeta_x plus
        zxm ([type]): zeta_x minus
        zyp ([type]): zeta_y plus
        zym ([type]): zeta_y minus
        fterms ([type]): RDTs f_jklm, use helper functions if you want to populate it
        order ([type]): the order of the calculation. Attention: **not** the order 'j+k+l+m' of the RDTs
        in 'fterms'

    Returns:
        [type]: a 'pd.DataFrame' filled with Courant-Snyder coordinates, position and momentum
    """
    hxp = zxp
    hxp += FACTOR * np.conj(fterms["1001"]) * zyp + FACTOR * np.conj(fterms["1010"]) * zym

    if order > 1:
        hxp -= 1.0 * zxp * np.abs(fterms["1010"])

    #df = pd.DataFrame()
    #df["HXP"] = hxp

    return hxp


def driven_courant_snyder(free_cs, ampl, nat_tunes, drv_tunes, phases, s_ac, plane):
    _plane = 0 if plane == 'X' else 1
    Qplus = nat_tunes[_plane] + drv_tunes[_plane]
    Qminus = nat_tunes[_plane] - drv_tunes[_plane]

    phadv = rdts.phadv(phases, s_ac, 0)[free_cs.index]

    return (free_cs
            + ampl / (4.0 * np.sin(np.pi * Qminus)) * np.exp(-2.0j*np.pi*(phadv + Qminus))
            - ampl / (4.0 * np.sin(np.pi * Qplus)) * np.exp(-2.0j*np.pi*(phadv - Qplus)))


def assemble_fterms(nonzero_values: dict, max_order: int) -> pd.DataFrame:
    fterms = {}
    columns = [f"{j}{k}{l}{m}"
               for j in range(max_order+1)
               for k in range(max_order+1)
               for l in range(max_order+1)
               for m in range(max_order+1)]

    fterms = pd.DataFrame(columns=columns)

    for k, v in nonzero_values.items():
        fterms[k] = v

    return fterms
