"""
Coupling
--------

Functions to calculate coupling from twiss dataframes.

.. rubric:: References

.. [#FranchiAnalyticformulasrapid2017]
    A. Franchi et al.,
    'Analytic formulas for the rapid evaluation of the orbit response matrix
    and chromatic functions from lattice parameters in circular accelerators'
    https://arxiv.org/abs/1711.06589
.. [#CalagaBetatroncouplingMerging2005]
    R. Calaga et al.,
    'Betatron coupling: Merging Hamiltonian and matrix approaches'
    Phys. Rev. ST Accel. Beams 8, 034001 (2005).
    https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.8.034001
.. [#PerssonImprovedControlBetatronCoupling]_
    T. Persson et al.,
   'Improved control of the betatron coupling in the Large Hadron Collider'
    Phys. Rev. ST Accel. Beams 17, 051004 (2014).
    http://cds.cern.ch/record/2135848/files/PhysRevSTAB.17.051004.pdf

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
        [#FranchiAnalyticformulasrapid2017]_ to be consistent with the RDT
        calculations from [#CalagaBetatroncouplingMerging2005]_ .


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
        df_res.loc[:, rdt].real *= -1  # definition

    if real:
        df_res = split_complex_columns(df_res, COUPLING_RDTS)

    return df_res


def coupling_from_cmatrix(df: TfsDataFrame, real=False, output: Sequence[str] = ("rdts", "gamma", "cmatrix")):
    """ Calculates C matrix and Coupling and Gamma from it.
    See [#CalagaBetatroncouplingMerging2005]_

    Args:
        df (TfsDataFrame): Twiss Dataframe
        real (bool): Split complex columns into two real-valued columns.
        output (list[str]): Combination of 'rdts', 'gamma' and 'cmatrix'.
                            Specifies which parameters one wants to output.

    Returns:
        New TfsDataFrame with columns as specified in 'output'.
    """
    LOG.debug("Calculating CMatrix.")
    df_res = TfsDataFrame()
    with timeit("CMatrix calculation", print_fun=LOG.debug):
        j = np.array([[0., 1.], [-1., 0.]])
        rs = np.reshape(df["R11", "R12", "R21", "R22"].to_numpy(), (len(df), 2, 2))
        cs = np.einsum("ij,kjn,no->kio", -j, np.transpose(rs, axes=(0, 2, 1)), j)
        cs = np.einsum("k,kij->kij", (1 / np.sqrt(1 + np.linalg.det(rs))), cs)

        g11a = 1 / np.sqrt(df.loc[:, f"{BETA}{X}"])
        g12a = np.zeros(len(df))
        g21a = df.loc[:, f"{ALPHA}{X}"] / np.sqrt(df.loc[:, f"{BETA}{X}"])
        g22a = np.sqrt(df.loc[:, f"{BETA}{X}"])
        gas = np.reshape(np.array([g11a, g12a, g21a, g22a]).T, (len(df), 2, 2))

        ig11b = np.sqrt(df.loc[:, f"{BETA}{Y}"])
        ig12b = np.zeros(len(df))
        ig21b = -df.loc[:, f"{ALPHA}{Y}"] / np.sqrt(df.loc[:, f"{BETA}{Y}"])
        ig22b = 1. / np.sqrt(df.loc[:, f"{BETA}{Y}"])
        igbs = np.reshape(np.array([ig11b, ig12b, ig21b, ig22b]).T, (len(df), 2, 2))

        cs = np.einsum("kij,kjl,kln->kin", gas, cs, igbs)
        gammas = np.sqrt(1 - np.linalg.det(cs))

    if "rdts" in output:
        df_res.loc[:, "F1001"] = ((cs[:, 0, 0] + cs[:, 1, 1]) * 1j +
                                  (cs[:, 0, 1] - cs[:, 1, 0])) / 4 / gammas
        df_res.loc[:, "F1010"] = ((cs[:, 0, 0] - cs[:, 1, 1]) * 1j +
                                  (-cs[:, 0, 1]) - cs[:, 1, 0]) / 4 / gammas
        LOG.debug(f"Average coupling amplitude |F1001|: {df['F1001'].abs().mean():g}")
        LOG.debug(f"Average coupling amplitude |F1010|: {df['F1010'].abs().mean():g}")

        if real:
            df_res = split_complex_columns(df_res, COUPLING_RDTS)

    if "cmatrix" in output:
        df_res.loc[:, "C11"] = cs[:, 0, 0]
        df_res.loc[:, "C12"] = cs[:, 0, 1]
        df_res.loc[:, "C21"] = cs[:, 1, 0]
        df_res.loc[:, "C22"] = cs[:, 1, 1]

    if "gamma" in output:
        df_res.loc[:, GAMMA] = gammas
        LOG.debug(f"Average gamma: {df[GAMMA].mean():g}")

    return df_res


def closest_tune_approach(df: TfsDataFrame, qx: float = None, qy: float = None, method: str = 'calaga'):
    """ Calculates the closest tune approach from coupling resonances.

    A complex F1001 column is assumed to be present in the DataFrame.
    This can be calculated by :func:`~optics_functions.rdt.rdts`
    :func:`~optics_functions.coupling.coupling_from_rdts` or
    :func:`~optics_functions.coupling.coupling_from_cmatrix`.
    If F1010 is also present it is used, otherwise assumed 0.

    The closest tune approach is calculated by means of
    Eq. (27) in [#CalagaBetatroncouplingMerging2005]_ ('calaga')
    by default, or approximated by
    Eq. (1) in [#PerssonImprovedControlBetatronCoupling]_ ('franchi')
    or Eq. (2) in [#PerssonImprovedControlBetatronCoupling]_ ('persson')
    or the latter without the exp(i(Qx-Qy)s/R) term ('persson_alt').

    Args:
        df (TfsDataFrame): Twiss Dataframe, needs to have complex-valued F1001 column.
        qx (float): Tune in X-Plane (if not given df.Q1 is assumed present)
        qy (float): Tune in Y-Plane (if not given df.Q2 is assumed present)
        method (str): Which method to use for evaluation.
                      Choices: 'calaga', 'franchi', 'perrson' and 'persson_alt'.

    Returns:
        New DataFrame with closest tune approach (DELTAQMIN) column.
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

    df_res = TfsDataFrame(index=df.index)
    df_res[f"{DELTA}{TUNE}{MINIMUM}"] = method_map[method.lower()](df, qx, qy)

    return df_res


def _cta_franchi(df, qx, qy):
    """ Closest tune approach calculated by Eq. (1) in [#PerssonImprovedControlBetatronCoupling]_ . """
    return 4 * ((qx - qy) * df['F1001']).abs()


def _cta_persson_alt(df, qx, qy):
    """ Closest tune approach calculated by Eq. (2) in [#PerssonImprovedControlBetatronCoupling]_ .

    The exp(i(Qx-Qy)s/R) term is omitted. """
    return 4 * (qx - qy) * df['F1001'] * np.exp(-1j*(df[f"{PHASE_ADV}{X}"]-df[f"{PHASE_ADV}{Y}"]))


def _cta_persson(df, qx, qy):
    """ Closest tune approach calculated by Eq. (2) in [#PerssonImprovedControlBetatronCoupling]_ . """
    return 4 * (qx - qy) * df['F1001'] * np.exp(1j *
           ((qx - qy) * df[S] / df.headers[LENGTH]) - (df[f"{PHASE_ADV}{X}"]-df[f"{PHASE_ADV}{Y}"])
           )


def _cta_calaga(df, qx, qy):
    """ Closest tune approach calculated by Eq. (27) in [#CalagaBetatroncouplingMerging2005]_ .

    If F1010 is not given, it is assumed to be zero.
    """
    f_diff = df["F1001"].abs()**2
    with suppress(KeyError):
        f_diff -= df["1010"].abs()**2

    return ((np.cos(PI2*qx) - np.cos(PI2*qy)) / (np.pi*(np.cos(PI2*qx) + np.cos(PI2*qy)))
            * (4 * np.sqrt(f_diff) / (1 + 4*f_diff)))



