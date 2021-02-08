"""
Resonance Driving Terms
***********************

Functions for the calculations of Resonance Driving Terms, as well as
getting lists of valid driving term indices for certain orders.

"""
import itertools
import logging
from math import factorial
from typing import Tuple, Sequence, List, Union

import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from optics_functions.constants import PI2I, X, Y, BETA, TUNE
from optics_functions.utils import (seq2str, timeit, get_all_phase_advances,
                                    dphi_at_element, dphi, i_pow,
                                    split_complex_columns)

LOG = logging.getLogger(__name__)


def calculate_rdts(df: TfsDataFrame, rdts: Sequence[str],
                   qx: float = None, qy: float = None, feeddown: int = 0,
                   complex_columns: bool = True, loop_phases: bool = False,
                   hamiltionian_terms: bool = False) -> TfsDataFrame:
    """ Calculates the Resonance Driving Terms.

    Eq. (A8) in [FranchiAnalyticFormulas2017]_ .

    Args:
        df (TfsDataFrame): Twiss Dataframe.
        rdts (Sequence): List of rdt-names to calculate.
        qx (float): Tune in X-Plane (if not given, header df.Q1 is assumed present).
        qy (float): Tune in Y-Plane (if not given, header df.Q2 is assumed present).
        feeddown (int): Levels of feed-down to include.
        complex_columns (bool): Output complex values in single column of type complex.
                        If ``False``, split complex columns into two real-valued columns.
        loop_phases (bool): Loop over elements when calculating phase-advances.
                            Might be slower for small number of elements, but
                            allows for large (e.g. sliced) optics.
        hamiltionian_terms (bool): Add the hamiltonian terms to the result dataframe.

    Returns:
        New TfsDataFrame with RDT columns.
    """
    LOG.info(f"Calculating RDTs: {seq2str(rdts):s}.")
    with timeit("RDT calculation", print_fun=LOG.debug):
        df_res = TfsDataFrame(index=df.index)
        if qx is None:
            qx = df.headers[f"{TUNE}1"]
        if qy is None:
            qy = df.headers[f"{TUNE}2"]

        if not loop_phases:
            phase_advances = get_all_phase_advances(df)  # might be huge!

        for rdt in rdts:
            rdt = rdt.upper()
            if len(rdt) != 5 or rdt[0] != 'F':
                raise ValueError(f"'{rdt:s}' does not seem to be a valid RDT name.")

            j, k, l, m = [int(i) for i in rdt[1:]]
            conj_rdt = jklm2str(k, j, m, l)

            if conj_rdt in df_res:
                df_res[rdt] = np.conjugate(df_res[conj_rdt])
            else:
                with timeit(f"calculating {rdt}", print_fun=LOG.debug):
                    n = j + k + l + m
                    jk, lm = j + k, l + m

                    if n <= 1:
                        raise ValueError(f"The RDT-order has to be >1 but was {n:d} for {rdt:s}")

                    denom_h = 1./(factorial(j) * factorial(k) * factorial(l) * factorial(m) * (2**n))
                    denom_f = 1./(1. - np.exp(PI2I * ((j-k) * qx + (l-m) * qy)))

                    betax = df[f"{BETA}{X}"]**(jk/2.)
                    betay = df[f"{BETA}{Y}"]**(lm/2.)

                    # Magnetic Field Strengths with Feed-Down
                    dx_idy = df[X] + 1j*df[Y]
                    k_complex = pd.Series(0j, index=df.index)  # Complex sum of strenghts (from K_n + iJ_n) and feeddown to them

                    for q in range(feeddown+1):
                        n_mad = n + q - 1
                        kl_iksl = df[f"K{n_mad:d}L"] + 1j * df[f"K{n_mad:d}SL"]
                        k_complex += (kl_iksl * (dx_idy**q)) / factorial(q)

                    # real(i**lm * k+ij) is equivalent to Omega-function in paper, see Eq.(A11)
                    # pd.Series is needed here, as np.real() returns numpy-array
                    k_real = pd.Series(np.real(i_pow(lm) * k_complex), index=df.index)
                    sources = df.index[k_real != 0]  # other elements do not contribute to integral, speedup summations

                    if not len(sources):
                        LOG.warning(f"No sources found for {rdt}. RDT will be zero.")
                        df_res[rdt] = 0j
                        if hamiltionian_terms:
                            df_res[f2h(rdt)] = 0j
                        continue

                    # calculate hamiltionian-numerator part
                    h_terms = -k_real.loc[sources] * betax.loc[sources] * betay.loc[sources]

                    if loop_phases:
                        # do loop over elements to not have elements x elements Matrix in memory
                        h_jklm = pd.Series(index=df.index, dtype=complex)
                        for element in df.index:
                            # calculate dphi from all sources to the element
                            # index-intersection keeps `element` at correct place in index
                            sources_plus = df.index.intersection(sources.union([element]))
                            dphis = dphi_at_element(df.loc[sources_plus, :], element, qx, qy)
                            phase_term = np.exp(PI2I * ((j-k) * dphis[X].loc[sources] + (l-m) * dphis[Y].loc[sources]))
                            h_jklm[element] = (h_terms * phase_term).sum() * denom_h
                    else:
                        phx = dphi(phase_advances['X'].loc[sources, :], qx)
                        phy = dphi(phase_advances['Y'].loc[sources, :], qy)
                        phase_term = np.exp(PI2I * ((j-k) * phx + (l-m) * phy))
                        h_jklm = phase_term.multiply(h_terms, axis="index").sum(axis="index").transpose() * denom_h

                    df_res[rdt] = h_jklm * denom_f
                    LOG.info(f"Average RDT amplitude |{rdt:s}|: {df_res[rdt].abs().mean():g}")

                    if hamiltionian_terms:
                        df_res[f2h(rdt)] = h_jklm

    if not complex_columns:
        terms = list(rdts)
        if hamiltionian_terms:
            terms += [f2h(rdt) for rdt in rdts]  # F#### -> H####
        df_res = split_complex_columns(df_res, terms)
    return df_res


def get_ac_dipole_rdts(order_or_terms: Union[int, str, Sequence[str]], spectral_line: Tuple[int],
                       plane: str, ac_tunes: Tuple[float, float], acd_name: str):
    """ Calculates the Hamiltonian Terms under Forced Motion.

    Args:
        order_or_terms (Union[int, str, Sequence[str]]): If an int is given all Resonance Driving
            Terms up to this order will be calculated. The strings are assumed to be the desired
            driving term names, e.g. "F1001"
        spectral_line (Tuple[int]): Needed to determine what phase advance is needed before and
            after AC dipole location, depends on detal+ and delta-. Sample input: (2,-1)
        plane (str):  Either 'H' or 'V' to determine phase term of AC dipole before and after
            ACD location.
        ac_tunes (Tuple[float, float]) Contains horizontal and vertical AC dipole tunes,
            i.e. (0.302, 0.33)
        acd_name (str): The AC Dipole element name (?).
    """
    raise NotImplementedError("Todo. Leave it here so it's not forgotten. See (and improve) python2 code!")


# RDT Definition Generation Functions ------------------------------------------

def get_all_to_order(n: int) -> List[Tuple[int, int, int, int]]:
    """ Returns list of all valid RDT jklm-tuple of order 2 to n """
    if n <= 1:
        raise ValueError("'n' must be greater 1 for resonance driving terms.")

    permut = [x for x in itertools.product(range(n + 1), repeat=4)
              if 1 < sum(x) <= n and not (x[0] == x[1] and x[2] == x[3])]
    return list(sorted(permut, key=sum))


def generator(orders: Sequence[int], normal: bool = True,
              skew: bool = True, complex_conj: bool = True) -> dict:
    """ Generates lists of RDT-4-tuples sorted into a dictionary by order.

    Args:
        orders (list): list of orders to be generated. Orders < 2 raise errors.
        normal (bool): generate normal RDTs (default: True).
        skew (bool): generate skew RDTs (default: True).
        complex_conj (bool): Have both, RDT and it's complex conjugate RDT in the list
                            (default: True).

    Returns:
        Dictionary with keys of orders containing lists of 4-Tuples for the RDTs of that order.
    """
    if any([n <= 1 for n in orders]):
        raise ValueError("All order must be greater 1 for resonance driving terms.")

    if not (normal or skew):
        raise ValueError("At least one of 'normal' or 'skew' parameters must be True.")

    permut = {o: [] for o in orders}
    for x in itertools.product(range(max(orders) + 1), repeat=4):
        order = sum(x)
        if ((order in orders)  # check for order
            and not (x[0] == x[1] and x[2] == x[3])  # rdt index rule
            and ((skew and sum(x[2:4]) % 2) or (normal and not sum(x[2:4]) % 2))  # skew or normal
            and (complex_conj or not((x[1], x[0], x[3], x[2]) in permut[order]))  # filter conj
        ):
            permut[order].append(x)
    return permut


# Other ------------------------------------------------------------------------

def jklm2str(j: int, k: int, l: int, m: int) -> str:
    return f"F{j:d}{k:d}{l:d}{m:d}"


def str2jklm(rdt: str) -> Tuple[int, ...]:
    return tuple(int(i) for i in rdt[1:])


def f2h(rdt: str) -> str:
    return f"H{rdt[1:]}"
