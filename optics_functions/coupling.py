"""
Coupling
--------

Functions to calculate coupling from twiss dataframes.
"""

from typing import Sequence

from tfs import TfsDataFrame
import logging

from optics_functions.constants import ALPHA, BETA, GAMMA, X, Y
from optics_functions.utils import split_complex_columns, timeit
import numpy as np

COUPLING_RDTS = ('F1001', 'F1010')


LOG = logging.getLogger(__name__)


def get_coupling_from_rdts(df: TfsDataFrame, qx: float = None, qy: float = None, feeddown: int = 0, real: bool = False):
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

    """
    df_res = get_rdts(df, COUPLING_RDTS, qx=qx, qy=qy, feeddown=feeddown)
    for rdt in COUPLING_RDTS:
        df_res.loc[:, rdt].real *= -1  # definition

    if real:
        df_res = split_complex_columns(df_res, COUPLING_RDTS)

    return df_res


def get_coupling_from_cmatrix(df: TfsDataFrame, real=False, output: Sequence[str] = ("rdts", "gamma", "cmatrix")):
        """ Calculates C matrix and Coupling and Gamma from it.
        See [#CalagaBetatroncouplingMerging2005]_
        """
        LOG.debug("Calculating CMatrix.")
        df_res = TfsDataFrame()
        with timeit("CMatrix calculation"):
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
