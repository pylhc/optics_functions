"""
Coupling
********

Functions to estimate coupling from twiss dataframes and
different methods to calculate the closest tune approach from
the calculated coupling RDTs.

"""
import logging
from contextlib import suppress
from typing import Sequence

import numpy as np
from tfs import TfsDataFrame
from pandas import DataFrame

from optics_functions.constants import (ALPHA, BETA, GAMMA,
                                        X, Y, TUNE, DELTA,
                                        MINIMUM, PI2, PHASE_ADV, S, LENGTH, IMAG, REAL)
from optics_functions.rdt import calculate_rdts
from optics_functions.utils import split_complex_columns, timeit

COUPLING_RDTS = ["F1001", "F1010"]


LOG = logging.getLogger(__name__)


# Coupling ---------------------------------------------------------------------

def coupling_via_rdts(df: TfsDataFrame, complex_columns: bool = True, **kwargs):
    """ Returns the coupling term.

    .. warning::
        This function changes sign of the real part of the RDTs compared to
        [FranchiAnalyticFormulas2017]_ to be consistent with the RDT
        calculations from [CalagaBetatronCoupling2005]_ .

    Args:
        df (TfsDataFrame): Twiss Dataframe
        complex_columns (bool): Output complex values in single column of type complex.
                                If ``False``, split complex columns into
                                two real-valued columns.

    Keyword Args:
        **kwargs: Remaining arguments from :func:`~optics_functions.rdt.rdts`
                  i.e. ``qx``, ``qy``, ``feeddown``, ``loop_phases``
                  and ``hamiltionian_terms``.

    Returns:
        New TfsDataFrame with Coupling Columns.
    """
    df_res = calculate_rdts(df, rdts=COUPLING_RDTS, **kwargs)
    for rdt in COUPLING_RDTS:
        df_res.loc[:, rdt].to_numpy().real *= -1  # definition, also: sets value in dataframe

    if not complex_columns:
        df_res = split_complex_columns(df_res, COUPLING_RDTS)

    return df_res


def coupling_via_cmatrix(df: DataFrame, complex_columns: bool = True,
                         output: Sequence[str] = ("rdts", "gamma", "cmatrix")) -> DataFrame:
    """ Calculates C matrix then Coupling and Gamma from it.
    See [CalagaBetatronCoupling2005]_ .

    Args:
        df (DataFrame): Twiss Dataframe
        complex_columns (bool): Output complex values in single column of type complex.
                                If ``False``, split complex columns into two
                                real-valued columns.
        output (Sequence[str]): Combination of 'rdts', 'gamma' and 'cmatrix'.
                            Specifies which parameters one wants to output.

    Returns:
        New TfsDataFrame with columns as specified in 'output'.
    """
    LOG.info("Calculating coupling from c-matrix.")
    df_res = DataFrame(index=df.index)

    with timeit("CMatrix calculation", print_fun=LOG.debug):
        n = len(df)
        gx, r, inv_gy = np.zeros((n, 2, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))

        # rs form after -J R^T J == inv(R)*det|R| == C
        r[:, 0, 0] = df["R22"]
        r[:, 0, 1] = -df["R12"]
        r[:, 1, 0] = -df["R21"]
        r[:, 1, 1] = df["R11"]

        r *= 1 / np.sqrt(1 + np.linalg.det(r)[:, None, None])

        # Cbar = Gx * C * Gy^-1  (Eq. (5) in reference)
        sqrtbetax = np.sqrt(df[f"{BETA}{X}"])
        sqrtbetay = np.sqrt(df[f"{BETA}{Y}"])

        gx[:, 0, 0] = 1 / sqrtbetax
        gx[:, 1, 0] = df[f"{ALPHA}{X}"] * gx[:, 0, 0]
        gx[:, 1, 1] = sqrtbetax

        inv_gy[:, 1, 1] = 1 / sqrtbetay
        inv_gy[:, 1, 0] = -df[f"{ALPHA}{Y}"] * inv_gy[:, 1, 1]
        inv_gy[:, 0, 0] = sqrtbetay

        c = np.matmul(gx, np.matmul(r, inv_gy))
        gamma = np.sqrt(1 - np.linalg.det(c))

    if "rdts" in output:
        denom = 1 / (4 * gamma)
        df_res.loc[:, "F1001"] = ((c[:, 0, 0] + c[:, 1, 1]) * 1j +
                                  (c[:, 0, 1] - c[:, 1, 0])) * denom
        df_res.loc[:, "F1010"] = ((c[:, 0, 0] - c[:, 1, 1]) * 1j +
                                  (-c[:, 0, 1]) - c[:, 1, 0]) * denom
        LOG.info(f"Average coupling amplitude |F1001|: {df_res['F1001'].abs().mean():g}")
        LOG.info(f"Average coupling amplitude |F1010|: {df_res['F1010'].abs().mean():g}")

        if not complex_columns:
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


# R-Matrix ---------------------------------------------------------------------

def rmatrix_from_coupling(df: DataFrame, complex_columns: bool = True) -> DataFrame:
    """ Calculates the R-matrix from a DataFrame containing the coupling columns
     as well as alpha and beta columns. This is the inverse of
    :func:`optics_functions.coupling.coupling_via_cmatrix`.
    See [CalagaBetatronCoupling2005]_ .

    Args:
        df (TfsDataFrame): Twiss Dataframe
        complex_columns (bool): Tells the function if the coupling input columns
                                are complex-valued or split into real and
                                imaginary parts.

    Returns:
        New TfsDataFrame containing the R-columns.
    """
    LOG.info("Calculating r-matrix from coupling rdts.")
    df_res = DataFrame(index=df.index)

    with timeit("R-Matrix calculation", print_fun=LOG.debug):
        if complex_columns:
            df = split_complex_columns(df, COUPLING_RDTS, drop=False)

        n = len(df)

        # From Eq. (5) in reference:
        inv_gx, jcj, gy = np.zeros((n, 2, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))

        sqrtbetax = np.sqrt(df[f"{BETA}{X}"])
        sqrtbetay = np.sqrt(df[f"{BETA}{Y}"])

        inv_gx[:, 1, 1] = 1 / sqrtbetax
        inv_gx[:, 1, 0] = -df[f"{ALPHA}{X}"] * inv_gx[:, 1, 1]
        inv_gx[:, 0, 0] = sqrtbetax

        gy[:, 0, 0] = 1 / sqrtbetay
        gy[:, 1, 0] = df[f"{ALPHA}{Y}"] * gy[:, 0, 0]
        gy[:, 1, 1] = sqrtbetay

        # Eq. (15)
        gamma = np.sqrt(1. / (1. + 4. * (df["F1001"].abs()**2 - df["F1010"].abs()**2)))

        # Eq. (11) and Eq. (12)
        cbar = np.zeros((n, 2, 2))
        cbar[:, 0, 0] = (df[f"F1001{IMAG}"] + df[f"F1010{IMAG}"]).to_numpy()
        cbar[:, 0, 1] = -(df[f"F1010{REAL}"] - df[f"F1001{REAL}"]).to_numpy()
        cbar[:, 1, 0] = -(df[f"F1010{REAL}"] + df[f"F1001{REAL}"]).to_numpy()
        cbar[:, 1, 1] = (df[f"F1001{IMAG}"] - df[f"F1010{IMAG}"]).to_numpy()
        cbar = 2 * gamma.to_numpy()[:, None, None] * cbar

        # Gx^-1 * Cbar * Gy = C  (Eq. (5) inverted)
        c = np.matmul(inv_gx, np.matmul(cbar, gy))

        # from above: -J R^T J == inv(R)*det|R| == C
        # therefore -J C^T J = R
        jcj[:, 0, 0] = c[:, 1, 1]
        jcj[:, 0, 1] = -c[:, 0, 1]
        jcj[:, 1, 0] = -c[:, 1, 0]
        jcj[:, 1, 1] = c[:, 0, 0]

        rmat = jcj * np.sqrt(1/(1 - np.linalg.det(jcj))[:, None, None])
        df_res["R11"] = rmat[:, 0, 0]
        df_res["R12"] = rmat[:, 0, 1]
        df_res["R21"] = rmat[:, 1, 0]
        df_res["R22"] = rmat[:, 1, 1]

    return df_res


# Closest Tune Approach --------------------------------------------------------

def closest_tune_approach(df: TfsDataFrame, qx: float = None, qy: float = None, method: str = "calaga"):
    """ Calculates the closest tune approach from coupling resonances.

    A complex F1001 column is assumed to be present in the DataFrame.
    This can be calculated by :func:`~optics_functions.rdt.rdts`
    :func:`~optics_functions.coupling.coupling_from_rdts` or
    :func:`~optics_functions.coupling.coupling_from_cmatrix`.
    If F1010 is also present it is used, otherwise assumed 0.

    The closest tune approach is calculated by means of Eq. (27) in
    [CalagaBetatronCoupling2005]_ (method='calaga') by default,
    or approximated by Eq. (1) in [PerssonImprovedControlCoupling2014]_
    (method='franchi') or Eq. (2) in [PerssonImprovedControlCoupling2014]_
    (method='persson') or the latter without the exp(i(Qx-Qy)s/R) term
    (method='persson_alt').

    For the 'persson' and 'persson_alt' methods, also MUX and MUY columns
    are needed in the DataFrame as well as LENGTH (of the machine) and S column
    for the 'persson' method.

    Args:
        df (TfsDataFrame): Twiss Dataframe, needs to have complex-valued F1001 column.
        qx (float): Tune in X-Plane (if not given, header df.Q1 is assumed present).
        qy (float): Tune in Y-Plane (if not given, header df.Q2 is assumed present).
        method (str): Which method to use for evaluation.
                      Choices: 'calaga', 'franchi', 'persson' and 'persson_alt'.

    Returns:
        New DataFrame with closest tune approach (DELTAQMIN) column.
        The value is real for 'calaga' and 'franchi' methods,
    """
    method_map = {
        "calaga": _cta_calaga,
        "franchi": _cta_franchi,
        "persson": _cta_persson,
        "persson_alt": _cta_persson_alt,
    }
    if qx is None:
        qx = df.headers[f"{TUNE}1"]
    if qy is None:
        qy = df.headers[f"{TUNE}2"]

    qx_frac, qy_frac = qx % 1, qy % 1

    dqmin_str = f"{DELTA}{TUNE}{MINIMUM}"
    df_res = TfsDataFrame(index=df.index, columns=[dqmin_str])
    df_res[dqmin_str] = method_map[method.lower()](df, qx_frac, qy_frac)

    LOG.info(f"({method}) |C-| = {np.abs(df_res[dqmin_str].mean())}")
    return df_res


def _cta_franchi(df: TfsDataFrame, qx_frac: float, qy_frac: float):
    """ Closest tune approach calculated by Eq. (1) in [PerssonImprovedControlCoupling2014]_ . """
    return 4 * (qx_frac - qy_frac) * df['F1001'].abs()


def _cta_persson_alt(df: TfsDataFrame, qx_frac: float, qy_frac: float):
    """ Closest tune approach calculated by Eq. (2) in [PerssonImprovedControlCoupling2014]_ .

    The exp(i(Qx-Qy)s/R) term is omitted. """
    deltaq = qx_frac - qy_frac  # fractional tune split
    return 4 * deltaq * df['F1001'] * np.exp(-1j * (df[f"{PHASE_ADV}{X}"] - df[f"{PHASE_ADV}{Y}"]))


def _cta_persson(df: TfsDataFrame, qx_frac: float, qy_frac: float):
    """ Closest tune approach calculated by Eq. (2) in [PerssonImprovedControlCoupling2014]_ . """
    deltaq = qx_frac - qy_frac  # fractional tune split
    return 4 * deltaq * df['F1001'] * np.exp(1j *
           ((deltaq * df[S] / (df.headers[LENGTH] / PI2)) - (df[f"{PHASE_ADV}{X}"] - df[f"{PHASE_ADV}{Y}"])))


def _cta_calaga(df: TfsDataFrame, qx_frac: float, qy_frac: float):
    """ Closest tune approach calculated by Eq. (27) in [CalagaBetatronCoupling2005]_ .

    If F1010 is not given, it is assumed to be zero.
    """
    f_diff = df["F1001"].abs() ** 2
    with suppress(KeyError):
        f_diff -= df["1010"].abs() ** 2

    return ((np.cos(PI2 * qx_frac) - np.cos(PI2 * qy_frac))
            / (np.pi * (np.sin(PI2 * qx_frac) + np.sin(PI2 * qy_frac)))
            * (4 * np.sqrt(f_diff) / (1 + 4*f_diff)))
