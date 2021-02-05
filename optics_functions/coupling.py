"""
Coupling
********

Functions to calculate coupling from twiss dataframes.

"""
import logging
from contextlib import suppress
from typing import Sequence

import numpy as np
from tfs import TfsDataFrame

from optics_functions.constants import ALPHA, BETA, GAMMA, X, Y, TUNE, DELTA, MINIMUM, PI2, PHASE_ADV, S, LENGTH
from optics_functions.rdt import rdts
from optics_functions.utils import split_complex_columns, timeit

COUPLING_RDTS = ('F1001', 'F1010')


LOG = logging.getLogger(__name__)


def coupling_from_rdts(df: TfsDataFrame, qx: float = None, qy: float = None, feeddown: int = 0, real: bool = False):
    """ Returns the coupling term.

    .. warning::
        This function changes sign of the real part of the RDTs compared to
        [FranchiAnalyticFormulas2017]_ to be consistent with the RDT
        calculations from [CalagaBetatronCoupling2005]_ .


    Args:
        df (TfsDataFrame): Twiss Dataframe
        qx (float): Tune in X-Plane (if not given df.Q1 is assumed present)
        qy (float): Tune in Y-Plane (if not given df.Q2 is assumed present)
        feeddown (int): Levels of feed-down to include
        real (bool): Split complex columns into two real-valued columns.

    Returns:
        New TfsDataFrame with Coupling Columns
    """
    df_res = rdts(df, COUPLING_RDTS, qx=qx, qy=qy, feeddown=feeddown)
    for rdt in COUPLING_RDTS:
        df_res.loc[:, rdt].to_numpy().real *= -1  # definition, also: sets value in dataframe

    if real:
        df_res = split_complex_columns(df_res, COUPLING_RDTS)

    return df_res


def coupling_from_cmatrix(df: TfsDataFrame, real=False, output: Sequence[str] = ("rdts", "gamma", "cmatrix")):
    """ Calculates C matrix and Coupling and Gamma from it.
    See [CalagaBetatronCoupling2005]_

    Args:
        df (TfsDataFrame): Twiss Dataframe
        real (bool): Split complex columns into two real-valued columns.
        output (list[str]): Combination of 'rdts', 'gamma' and 'cmatrix'.
                            Specifies which parameters one wants to output.

    Returns:
        New TfsDataFrame with columns as specified in 'output'.
    """
    LOG.debug("Calculating CMatrix.")
    df_res = TfsDataFrame(index=df.index)
    with timeit("CMatrix calculation", print_fun=LOG.debug):
        n = len(df)
        gx, r, igy = np.zeros((n, 2, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))

        # rs form after -J R^T J == inv(R)*det|R| == C
        r[:, 0, 0] = df["R22"]
        r[:, 0, 1] = -df["R12"]
        r[:, 1, 0] = -df["R21"]
        r[:, 1, 1] = df["R11"]

        r *= 1 / np.sqrt(1 + np.linalg.det(r)[:, None, None])

        # Cbar = Gx * C * Gy^-1
        sqrtbetax = np.sqrt(df[f"{BETA}{X}"])
        sqrtbetay = np.sqrt(df[f"{BETA}{Y}"])

        gx[:, 0, 0] = 1 / sqrtbetax
        gx[:, 1, 0] = df[f"{ALPHA}{X}"] * gx[:, 0, 0]
        gx[:, 1, 1] = sqrtbetax

        igy[:, 1, 1] = 1 / sqrtbetay
        igy[:, 1, 0] = -df[f"{ALPHA}{Y}"] * igy[:, 1, 1]
        igy[:, 0, 0] = sqrtbetay

        c = np.matmul(gx, np.matmul(r, igy))
        gamma = np.sqrt(1 - np.linalg.det(c))

    if "rdts" in output:
        denom = 1 / (4 * gamma)
        df_res.loc[:, "F1001"] = ((c[:, 0, 0] + c[:, 1, 1]) * 1j +
                                  (c[:, 0, 1] - c[:, 1, 0])) * denom
        df_res.loc[:, "F1010"] = ((c[:, 0, 0] - c[:, 1, 1]) * 1j +
                                  (-c[:, 0, 1]) - c[:, 1, 0]) * denom
        LOG.debug(f"Average coupling amplitude |F1001|: {df_res['F1001'].abs().mean():g}")
        LOG.debug(f"Average coupling amplitude |F1010|: {df_res['F1010'].abs().mean():g}")

        if real:
            df_res = split_complex_columns(df_res, COUPLING_RDTS)

    if "cmatrix" in output:
        df_res.loc[:, "C11"] = c[:, 0, 0]
        df_res.loc[:, "C12"] = c[:, 0, 1]
        df_res.loc[:, "C21"] = c[:, 1, 0]
        df_res.loc[:, "C22"] = c[:, 1, 1]

    if "gamma" in output:
        df_res.loc[:, GAMMA] = gamma
        LOG.debug(f"Average gamma: {df_res[GAMMA].mean():g}")

    return df_res


def closest_tune_approach(df: TfsDataFrame, qx: float = None, qy: float = None, method: str = 'calaga'):
    """ Calculates the closest tune approach from coupling resonances.

    A complex F1001 column is assumed to be present in the DataFrame.
    This can be calculated by :func:`~optics_functions.rdt.rdts`
    :func:`~optics_functions.coupling.coupling_from_rdts` or
    :func:`~optics_functions.coupling.coupling_from_cmatrix`.
    If F1010 is also present it is used, otherwise assumed 0.

    The closest tune approach is calculated by means of
    Eq. (27) in [CalagaBetatronCoupling2005]_ ('calaga')
    by default, or approximated by
    Eq. (1) in [PerssonImprovedControlCoupling2014]_ ('franchi')
    or Eq. (2) in [PerssonImprovedControlCoupling2014]_ ('persson')
    or the latter without the exp(i(Qx-Qy)s/R) term ('persson_alt').

    For the 'persson' and 'persson_alt' methods, also MUX and MUY columns
    are needed in the DataFrame as well as LENGTH (of the machine) and S column
    for the 'persson' method.

    Args:
        df (TfsDataFrame): Twiss Dataframe, needs to have complex-valued F1001 column.
        qx (float): Tune in X-Plane (if not given df.Q1 is assumed present)
        qy (float): Tune in Y-Plane (if not given df.Q2 is assumed present)
        method (str): Which method to use for evaluation.
                      Choices: 'calaga', 'franchi', 'perrson' and 'persson_alt'.

    Returns:
        New DataFrame with closest tune approach (DELTAQMIN) column.
        The value is real for 'calaga' and 'franchi' methods,
    """
    method_map = {
        'calaga': _cta_calaga,
        'franchi': _cta_franchi,
        'persson': _cta_persson,
        'persson_alt': _cta_persson_alt,
    }
    if qx is None:
        qx = df.headers[f"{TUNE}1"]

    if qy is None:
        qy = df.headers[f"{TUNE}2"]
    qx, qy = qx % 1, qy % 1

    dqmin = f"{DELTA}{TUNE}{MINIMUM}"
    df_res = TfsDataFrame(index=df.index, columns=[dqmin])
    df_res[dqmin] = method_map[method.lower()](df, qx, qy)

    LOG.info(f"({method}) |C-| = {np.abs(df_res[dqmin].mean())}")
    return df_res


def _cta_franchi(df, qx, qy):
    """ Closest tune approach calculated by Eq. (1) in [PerssonImprovedControlCoupling2014]_ . """
    return 4 * (qx - qy) * df['F1001'].abs()


def _cta_persson_alt(df, qx, qy):
    """ Closest tune approach calculated by Eq. (2) in [PerssonImprovedControlCoupling2014]_ .

    The exp(i(Qx-Qy)s/R) term is omitted. """
    return 4 * (qx - qy) * df['F1001'] * np.exp(-1j*(df[f"{PHASE_ADV}{X}"]-df[f"{PHASE_ADV}{Y}"]))


def _cta_persson(df, qx, qy):
    """ Closest tune approach calculated by Eq. (2) in [PerssonImprovedControlCoupling2014]_ . """
    return 4 * (qx - qy) * df['F1001'] * np.exp(1j *
           (((qx - qy) * df[S] / (df.headers[LENGTH]/PI2)) - (df[f"{PHASE_ADV}{X}"]-df[f"{PHASE_ADV}{Y}"])))


def _cta_calaga(df, qx, qy):
    """ Closest tune approach calculated by Eq. (27) in [CalagaBetatronCoupling2005]_ .

    If F1010 is not given, it is assumed to be zero.
    """
    f_diff = df["F1001"].abs()**2
    with suppress(KeyError):
        f_diff -= df["1010"].abs()**2

    return ((np.cos(PI2*qx) - np.cos(PI2*qy)) / (np.pi*(np.sin(PI2*qx) + np.sin(PI2*qy)))
            * (4 * np.sqrt(f_diff) / (1 + 4*f_diff)))



