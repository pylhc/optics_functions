"""
Utilities
---------

Reusable utilities that might be needed in multiple optics functions.
"""
import logging
import string
from contextlib import contextmanager
from time import time
from typing import Sequence

import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from optics_functions.constants import REAL, IMAG, PLANES, PHASE_ADV

LOG = logging.getLogger(__name__)


def split_complex_columns(df: TfsDataFrame, columns: Sequence[str],
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
    for column in columns:
        df[f"{column}{REAL}"] = np.real(df[column])
        df[f"{column}{IMAG}"] = np.imag(df[column])

    if drop:
        df = df.drop(columns=columns)
    return df


# Phase Advance Functions ------------------------------------------------------


def get_all_phase_advances(twiss_df):
    """
    Calculate phase advances between all elements.
    Will result in a elements x elements matrix, that might be very large!

    Returns:
        Matrices similar to DPhi(i,j) = Phi(j) - Phi(i)
    """
    LOG.debug("Calculating Phase Advances:")
    phase_advance_dict = dict.fromkeys(PLANES)
    with timeit("Phase Advance calculations"):
        for plane in PLANES:
            phases_mdl = twiss_df.loc[twiss_df.index, f"{PHASE_ADV}{plane}"]
            # Same convention as in [1]: DAdv(i,j) = Phi(j) - Phi(i)
            phase_advances = pd.DataFrame((phases_mdl[None, :] - phases_mdl[:, None]),
                                          index=twiss_df.index,
                                          columns=twiss_df.index)
            # Do not calculate dphi and tau here.
            # only slices of phase_advances as otherwise super slow
            phase_advance_dict[plane] = phase_advances
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
        phase_advance_dict[plane] = pd.concat([(phases[element] - phases[:element])[:-1],  #  only until element
                                               (phases[element] - phases[element:]+tune)])
    return phase_advance_dict


def add_missing_columns(df, columns):
    """ Check if K_columns are in df and add them all zero if not."""
    for c in columns:
        if c not in df.columns:
            LOG.debug(f"Added {k:s} with all zero to data-frame.")
            df[c] = 0.
    return df

# Timing -----------------------------------------------------------------------


@contextmanager
def timeit(text: str = "Time used {:.3f}s", print_fun=LOG.debug,):
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

def seq2str(sequence: Sequence):
    return ", ".join(str(item) for item in sequence)


def i_pow(n):
    return 1j**(n % 4)   # more exact and quicker with modulo
