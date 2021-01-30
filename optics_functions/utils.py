"""
Utilities
---------

Reusable utilities that might be needed in multiple optics functions.


.. rubric:: References

.. [#FranchiAnalyticformulasrapid2017]
    A. Franchi et al.,
    'Analytic formulas for the rapid evaluation of the orbit response matrix
    and chromatic functions from lattice parameters in circular accelerators'
    https://arxiv.org/abs/1711.06589

"""
import logging
import string
from contextlib import contextmanager
from time import time
from typing import Sequence, Tuple, Iterable

import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from optics_functions.constants import REAL, IMAG, PLANES, PHASE_ADV, X, Y, S, NAME, DELTA_ORBIT

LOG = logging.getLogger(__name__)
D = DELTA_ORBIT

# DataFrames -------------------------------------------------------------------


def prepare_twiss_dataframe(beam: int,
                            df_twiss: TfsDataFrame, df_errors: pd.DataFrame = None,
                            max_order: int = 16, join: str = "inner") -> TfsDataFrame:
    """ Prepare dataframe to use with the optics functions.

    - Adapt Beam 4 signs.
    - Add missing K(S)L and orbit columns.
    - Merge optics and error dataframes (add values).

    Args:
        beam (int): Beam that is being used. 1,2 or 4. Only 4 has an effect.
        df_twiss (TfsDataFrame): Twiss-optics DataFrame
        df_errors (DataFrame): Twiss-errors DataFrame (optional)
        max_order (int): Maximum field order to be still included (1==Dipole)
        join (str): How to join elements of optics and errors. "inner" or "outer".

    Returns:
        TfsDataFrame with necessary columns added. If a merge happened, only the
        neccessary columns are present.
    """
    df_twiss, df_errors = df_twiss.copy(), df_errors.copy()  # As data is moved around
    if beam == 4:
        df_twiss, df_errors = switch_signs_for_beam4(df_twiss, df_errors)

    df_twiss = set_name_index(df_twiss, 'twiss')

    k_columns = [f"K{n}{s}L" for n in range(max_order) for s in ('S', '')]
    orbit_columns = list(PLANES)
    if df_errors is None:
        return add_missing_columns(df_twiss, k_columns + orbit_columns)

    # Merge Dataframes
    df_errors = set_name_index(df_errors, 'error')
    index = df_twiss.index.join(df_errors.index, how=join)
    df = TfsDataFrame(index=index, headers=df_twiss.headers.copy())
    if join == "inner":
        df[S] = df_twiss[S]
    else:
        # Merge S column and set zeros in dfs for addition, where elements are missing
        for df_self, df_other in ((df_twiss, df_errors), (df_errors, df_twiss)):
            df.loc[df_self.index, S] = df_self[S]
            for new_indx in df_other.index.difference(df_self.index):
                df_self.loc[new_indx, :] = 0
        df = df.sort_values(by=S)

    df_twiss = add_missing_columns(df_twiss, k_columns + list(PLANES))
    df_errors = add_missing_columns(df_errors, k_columns + [f"{D}{X}", f"{D}{Y}"])
    df_errors = df_errors.rename(columns={f"{D}{p}": p for p in PLANES})

    add_columns = k_columns + orbit_columns
    df.loc[:, add_columns] = df_twiss[add_columns] + df_errors[add_columns]
    return df


def split_complex_columns(df: pd.DataFrame, columns: Sequence[str],
                          drop: bool = True) -> TfsDataFrame:
    """ Splits the given complex columns into two real-values columns containing the
    real and imaginary parts of the original columns.

    Args:
        df (TfsDataFrame): DataFrame containing the original columns.
        columns (Sequence[str]): List of column names to be replaced.
        drop (bool): Original columns are not present in resulting DataFrame

    Returns:
        Original TfsDataFrame with added columns.
    """
    df = df.copy()
    for column in columns:
        df[f"{column}{REAL}"] = np.real(df[column])
        df[f"{column}{IMAG}"] = np.imag(df[column])

    if drop:
        df = df.drop(columns=columns)
    return df


def switch_signs_for_beam4(df_twiss: pd.DataFrame, df_errors: pd.DataFrame = None) -> Tuple[TfsDataFrame, TfsDataFrame]:
    """ Switch the signs for Beam 4 optics.
    This is due to the switch in direction for this beam and
    (anti-) symmetry after a rotation of 180deg around the y-axis of magnets,
    combined with the fact that the KL values in MAD-X twiss do not change sign,
    but in the errors they do (otherwise it would compensate).
    Magnet orders that show anti-symmetry are: a1 (K0SL), b2 (K1L), a3 (K2SL), b4 (K3L) etc.
    Also the sign for (delta) X is switched back to have the same orientation as beam2."""
    LOG.debug(f"Switching signs for X and K(S)L values when needed, to match Beam 4 to Beam 2.")
    df_twiss, df_errors = df_twiss.copy(), df_errors.copy()
    df_twiss[X] = -df_twiss[X]

    if df_errors is not None:
        df_errors[f"{D}{X}"] = -df_errors[f"{D}{X}"]
        max_order = df_errors.columns.str.extract(r"^K(\d+)S?L$", expand=False).dropna().astype(int).max()
        for order in range(max_order+1):
            name = f"K{order:d}{'' if order % 2 else 'S'}L"  # odd -> '', even -> S
            if name in df_errors.columns:
                df_errors[name] = -df_errors[name]
    return df_twiss, df_errors


# Phase Advance Functions ------------------------------------------------------


def get_all_phase_advances(df: pd.DataFrame) -> dict:
    """
    Calculate phase advances between all elements.
    Will result in a elements x elements matrix, that might be very large!

    Returns:
        Dictionary with DataFrame matrices similar to DPhi(i,j) = Phi(j) - Phi(i)
    """
    LOG.debug("Calculating Phase Advances:")
    phase_advance_dict = dict.fromkeys(PLANES)
    with timeit("Phase Advance calculations"):
        for plane in PLANES:
            phases_mdl = df[f"{PHASE_ADV}{plane}"].to_numpy()
            # Same convention as in [1]: DAdv(i,j) = Phi(j) - Phi(i)
            # Do not calculate dphi and tau here.
            # only slices of phase_advances as otherwise super slow
            phase_advance_dict[plane] = pd.DataFrame(
                (phases_mdl[None, :] - phases_mdl[:, None]),
                index=df.index,
                columns=df.index)
    return phase_advance_dict


def dphi(data, q):
    """ Return dphi from phase advances in data, see Eq. 8 in [#FranchiAnalyticformulasrapid2017]_

    Args:
        data (DataFrame, Series): Phase-Advance data.
        q: Tune
    """
    return data + np.where(data <= 0, q, 0)  # '<=' seems to be what MAD-X does


def tau(data, q):
    """ Return tau from phase advances in data, see Eq. 16 in [#FranchiAnalyticformulasrapid2017]_

    Args:
        data (DataFrame, Series): Phase-Advance data.
        q: Tune
    """
    return data + np.where(data <= 0, q / 2, -q / 2)  # '<=' seems to be what MAD-X does


def dphi_at_element(df, element, qx, qy):
    """ Return dphis for both planes at the given element.
    See Eq. 8 in [#FranchiAnalyticformulasrapid2017]_

    Args:
        df (DataFrame): DataFrame containing the Phase-Advance columns for both planes.
        element: Element at which dphi is calculated (must be in the index of df).
        qx (float): Tune in X-Plane.
        qy (float): Tune in Y-Plane.

    Returns:
        dict of planes with DPhi[i] = Phi[element] - Phi[i] (+ Tune if Phi[i] > Phi[element])
    """
    phase_advance_dict = dict.fromkeys(PLANES)
    for tune, plane in zip((qx, qy), PLANES):
        phases = df[f"{PHASE_ADV}{plane}"]
        phase_advance_dict[plane] = pd.concat([(phases[element] - phases.loc[:element])[:-1],  #  only until element
                                               (phases[element] - phases.loc[element:]+tune)])
    return phase_advance_dict


def add_missing_columns(df, columns):
    """ Check if K_columns are in df and add them all zero if not."""
    for c in columns:
        if c not in df.columns:
            LOG.debug(f"Added {c:s} with all zero to data-frame.")
            df[c] = 0.
    return df


# Timing -----------------------------------------------------------------------

@contextmanager
def timeit(text: str = "Time used {:.3f}s", print_fun=LOG.debug):
    """ Timing Helper with logging/printing output. """
    start_time = time()
    try:
        yield
    finally:
        time_used = time() - start_time
        if len(get_format_keys(text)):
            print_fun(text.format(time_used))
        else:
            print_fun(f"Time used for {text}: {time_used:.3f}s")


def get_format_keys(format_str: str):
    """ Get keys from format string. Unnamed placeholders are returned as empty strings."""
    return [t[1] for t in string.Formatter().parse(format_str) if t[1] is not None]


# Other ------------------------------------------------------------------------

def seq2str(sequence: Iterable) -> str:
    """ Converts an Iterable to string of it's comma separated elements. """
    return ", ".join(str(item) for item in sequence)


def i_pow(n: int) -> complex:
    """ Calculates i**n in a quick and exact way. """
    return 1j**(n % 4)


def set_name_index(df: pd.DataFrame, df_name='') -> pd.DataFrame:
    """ Sets the NAME column as index if present and checks for string index. """
    if NAME in df.columns:
        df = df.set_index(NAME)

    if not all(isinstance(indx, str) for indx in df.index):
        raise TypeError(f"Index of the {df_name} Dataframe should be string (i.e. from {NAME})")

    return df
