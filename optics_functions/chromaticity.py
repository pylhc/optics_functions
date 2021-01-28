"""
Chromaticity
------------

Functions to calculate chromaticity and chromatic beating.
"""
import numpy as np
import logging

from tfs import TfsDataFrame

from optics_functions.constants import PLANES, DISPERSION, X, Y, BETA, CHROM_TERM, CHROMATICITY, DELTA, TUNE
from optics_functions.dispersion import linear_dispersion
from optics_functions.utils import timeit, get_all_phase_advances, tau

LOG = logging.getLogger(__name__)


def linear_chromaticity(df: TfsDataFrame, qx: float = None, qy: float = None,
                        feeddown: int = 0, save_memory=False):
    """ Calculate the Linear Chromaticity

    Eq. 31 in [#FranchiAnalyticformulasrapid2017]_
    """
    LOG.debug("Calculating Linear Chromaticity")
    with timeit("Linear Chromaticity calculations", print_fun=LOG.debug):
        chromatic_colums = [f"{CHROM_TERM}{p}" for p in PLANES]
        chromaticity_headers = [f"{CHROMATICITY}{p}" for p in (1, 2)]  # TODO: X, Y??

        if any(c not in df.columns for c in chromatic_colums):
            df_res = calc_chromatic_term(df, qx=qx, qy=qy, feeddown=feeddown, save_memory=save_memory)
        else:
            LOG.info("Chromatic Terms found in DataFrame. Using these.")
            df_res = df[chromatic_colums]

        for sign_, column, header in zip((-1, 1), chromatic_colums, chromaticity_headers):
            df_res.headers[header] = sign_/(4 * np.pi) * df_res[column].sum(axis="index")

        LOG.debug(f"Q'x: {df_res[chromaticity_headers[0]]:g}")
        LOG.debug(f"Q'y: {df_res[chromaticity_headers[1]]:g}")


def chromatic_beating(df: TfsDataFrame, qx: float = None, qy: float = None,
                      feeddown: int = 0, save_memory=False):
    """ Calculate the Chromatic Beating

    Eq. 36 in [#FranchiAnalyticformulasrapid2017]_
    """
    with timeit("Chromatic Beating calculations", print_fun=LOG.debug):
        chromatic_colums = [f"{CHROM_TERM}{p}" for p in PLANES]
        if qx is None:
            qx = df.headers[f"{TUNE}1"]

        if qy is None:
            qy = df.headers[f"{TUNE}2"]

        if any(c not in df.columns for c in chromatic_colums):
            df_res = calc_chromatic_term(df, qx=qx, qy=qy, feeddown=feeddown, save_memory=save_memory)
        else:
            LOG.info("Chromatic Terms found in DataFrame. Using these.")
            df_res = df[chromatic_colums]

        if not save_memory:
            phase_advances = get_all_phase_advances(df)  # might be huge!
        else:
            raise NotImplementedError  # TODO

        mask = (df_res[f'{CHROM_TERM}{X}'] != 0) | (df_res[f'{CHROM_TERM}{Y}'] != 0)

        for sign_, plane, tune in zip((1, -1), PLANES, (qx, qy)):
            df_res.loc[mask, f'{DELTA}{CHROM_TERM}{plane}'] = sign_ * _chromatic_beating(
                df_res.loc[mask, f'{CHROM_TERM}{plane}'],
                tau(phase_advances[plane].loc[mask, :], tune),
                tune).transpose() - 1

    for plane in PLANES:
        values = df_res[f'{DELTA}{CHROM_TERM}{plane}']
        LOG.debug(f"Pk2Pk chromatic beating in {plane:s}: {values.max()-values.min():g}")
    return df_res


def _chromatic_beating(chrom_term, tau, q):
    """ Chromatic Beating helper function. """
    return (1 / (2 * np.sin(2 * np.pi * q)) *
            np.cos(4 * np.pi * tau).mul(chrom_term, axis='index').sum(axis='index')
            )


def calc_chromatic_term(df: TfsDataFrame, qx: float = None, qy: float = None,
                        feeddown: int = 0, save_memory=False):
    """ Calculates the chromatic term which is common to all chromatic equations """
    LOG.debug("Calculating Chromatic Term.")
    with timeit("Chromatic Term calculation", print_fun=LOG.debug):
        df_res = TfsDataFrame(0, index=df.index, columns=[f'{CHROM_TERM}{X}', f'{CHROM_TERM}{Y}'])

        if feeddown:
            raise NotImplementedError  # TODO
        mask = (df['K1L'] != 0) | (df['K2L'] != 0) | (df['K2SL'] != 0)

        disp_columns = [f"{DISPERSION}{X}", f"{DISPERSION}{Y}"]
        if any(c not in df.columns for c in disp_columns):
            LOG.info("Dispersion values not found in twiss DataFrame. Calculating analytic values.")
            df_disp = linear_dispersion(df, qx=qx, qy=qy, feeddown=feeddown, save_memory=save_memory)
            df_res.loc[:, disp_columns] = df_disp
        else:
            df_disp = df.loc[:, disp_columns]

        sum_term = (
                df.loc[mask, 'K1L'] -
                (df.loc[mask, 'K2L'] * df_disp.loc[mask, f'{DISPERSION}{X}']) +
                (df.loc[mask, 'K2SL'] * df_disp.loc[mask, f'{DISPERSION}{Y}'])
        )

        for p in PLANES:
            df_res.loc[mask, f'{CHROM_TERM}{p}'] = sum_term * df.loc[mask, f'{BETA}{p}']
        return df_res

