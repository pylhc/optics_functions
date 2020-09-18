"""Resonance driving term module

Calculates the resonance driving terms from twiss (needs beta functions, magnet strengths and phase advances)
"""

import pandas as pd
import numpy as np
from numpy import sqrt, exp, pi
from math import factorial

TWOPI_I = 2.0j * pi


def phadv(phases, i, tune):
    return pd.concat([(phases[i] - phases[:i] - tune)[:-1], phases[i] - phases[i:]])


def calc_fjklm(df: pd.DataFrame,
               j: int,
               k: int,
               l: int,
               m: int,
               tune_x: float,
               tune_y: float,
               inplace: bool = False,
               mask=None) -> pd.DataFrame:
    """Calculates the resonance driving term Fjklm. Low level function, needs a DataFrame with beta
    functions, phases and magnet strengths and, separately, the tunes. Four integers `j,k,l,m` define
    which RDT should be calculated.

    Args:
        df (pd.DataFrame): A dataframe with beta functions, phases and magnet strengths. It must have
        the following columns
            +---------+-----------------------------------------------+
            | `BETX`  | hor. $beta$ function                          |
            +---------+-----------------------------------------------+
            | `BETY`  | ver. $beta$ function                          |
            +---------+-----------------------------------------------+
            | `MUX`   | hor. phase                                    |
            +---------+-----------------------------------------------+
            | `MUY`   | ver. phase                                    |
            +---------+-----------------------------------------------+
            | `KnL`   | normal magnet strength, for appearing order n |
            +---------+-----------------------------------------------+
            | `KnSL`  | skew magnet strength, for appearing order n   |
            +---------+-----------------------------------------------+
        j (int): RDT index j, left moving x coordinate
        k (int): RDT index k, right moving x coordinate $h_x^-$
        l (int): RDT index l, left moving y coordinate $h_y^+$
        m (int): RDT index m, right moving y coordinate $h_y^-$
        tune_x (float): horizontal tune
        tune_y (float): vertical tune

    Raises:
        KeyError: if any of the necessary columns is missing

    Returns:
        pd.DataFrame: a dataframe populated with the requested RDTs
    """
    n = j+k+l+m
    k1sl = df["K1SL"]
    betx = df["BETX"]
    bety = df["BETY"]
    mux = df["MUX"]
    muy = df["MUY"]

    if mask is None:
        mask = df.index

    f = pd.Series(index=mask, dtype=complex)
    for idx in mask:
        summand = (
            0.0j + k1sl*sqrt(np.power(betx, j+k) * np.power(bety, l+m))
            * exp(TWOPI_I * ((j-k)*phadv(mux, idx, tune_x) +
                             (l-m)*phadv(muy, idx, tune_y)))
        )
        f[idx] = (-np.sum(summand) / (factorial(j) * factorial(k) * factorial(l) * factorial(m) * 2**n *
                                      (1.0 - exp(-TWOPI_I * ((j-k)*tune_x + (l-m)*tune_y)))))
    if inplace:
        df[f"F{j}{k}{l}{m}"] = f
        return df
    return f
