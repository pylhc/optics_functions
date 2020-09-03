"""
Coordinate transformations
"""

import pandas as pd
import numpy as np

FACTOR = 0.5

def cmplx_courant_snyder(df: pd.DataFrame,
                         plane: str) -> pd.DataFrame:
    if plane not in ['X', 'Y']:
        raise RuntimeError("`plane` must be one of `'X'`, `'Y'`")
    new_df = pd.DataFrame()
    sqrt_b = np.sqrt(df[f"BET{plane}"])
    a = df[f"ALF{plane}"]
    new_df[f'H{plane}'] = df[plane] / sqrt_b
    new_df[f'HP{plane}'] = new_df[f'H{plane}'] * a + sqrt_b * df[f"P{plane}"]

    return new_df


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

    df = pd.DataFrame()
    df["HXP"] = hxp

    return df


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
