"""
Dispersion
**********

Functions to calculate the (linear) dispersion.

"""
import logging

import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from optics_functions.constants import TUNE, DISPERSION, X, Y, BETA, PLANES
from optics_functions.utils import get_all_phase_advances, timeit, tau

LOG = logging.getLogger(__name__)


def linear_dispersion(df: TfsDataFrame, qx: float = None, qy: float = None,
                      feeddown: int = 0, save_memory=False) -> TfsDataFrame:
    """ Calculate the Linear Disperion.

    Eq. 24 in [FranchiAnalyticFormulas2017]_
    Args:
        df (TfsDataFrame): Twiss Dataframe
        qx (float): Tune in X-Plane (if not given df.Q1 is assumed present)
        qy (float): Tune in Y-Plane (if not given df.Q2 is assumed present)
        feeddown (int): Levels of feed-down to include.
        save_memory (bool): Loop over elements when calculating phase-advances.
                            Might be slower for small number of elements, but
                            allows for large (e.g. sliced) optics.

    Returns:
        TfsDataFrame with dispersion columns.
    """
    # Calculate
    LOG.debug("Calculate Linear Dispersion")
    with timeit("Linear Dispersion Calculation", print_fun=LOG.debug):

        df_res = TfsDataFrame(index=df.index)
        if qx is None:
            qx = df.headers[f"{TUNE}1"]

        if qy is None:
            qy = df.headers[f"{TUNE}2"]

        if not save_memory:
            phase_advances = get_all_phase_advances(df)  # might be huge!
        else:
            raise NotImplementedError  # TODO

        if feeddown:
            raise NotImplementedError  # TODO

        # sources
        k0_mask = df['K0L'] != 0
        k0s_mask = df['K0SL'] != 0
        k1s_mask = df['K1SL'] != 0

        mx_mask = k0_mask | k1s_mask  # magnets contributing to Dx,j (-> Dy,m)
        my_mask = k0s_mask | k1s_mask  # magnets contributing to Dy,j (-> Dx,m)

        if not any(mx_mask | my_mask):
            LOG.warning("No linear dispersion contributions found. Values will be zero.")
            df_res[f'{DISPERSION}{X}'] = 0.
            df_res[f'{DISPERSION}{Y}'] = 0.
            return

        # create temporary DataFrame for magnets with coefficients already in place
        coeff = "COEFF"
        df_temp = TfsDataFrame(index=df.index).join(
            _dispersion_coeff(df.loc[:, f'{BETA}{X}'], qx)).join(
            _dispersion_coeff(df.loc[:, f'{BETA}{Y}'], qy))
        df_temp.columns = [f'{coeff}{X}', f'{coeff}{Y}']

        LOG.debug("Calculate uncoupled linear dispersion")
        df_temp.loc[my_mask, f'{DISPERSION}{X}'] = (
                df_temp.loc[my_mask, f'{coeff}{X}'] *
                _dispersion_sum(df.loc[mx_mask, 'K0L'],
                                0,
                                0,
                                df.loc[mx_mask, f'{BETA}{X}'],
                                tau(phase_advances[X].loc[mx_mask, my_mask], qx)).transpose()
        )
        df_temp.loc[mx_mask, f'{DISPERSION}{Y}'] = (
                df_temp.loc[mx_mask, f'{coeff}{Y}'] *
                _dispersion_sum(-df.loc[my_mask, 'K0SL'],  # MINUS!
                                0,
                                0,
                                df.loc[my_mask, f'{BETA}{Y}'],
                                tau(phase_advances[Y].loc[my_mask, mx_mask], qy)).transpose()
        )

        LOG.debug("  Calculate full linear dispersion values")
        df_res.loc[:, f'{DISPERSION}{X}'] = (
                df_temp.loc[:, f'{coeff}{X}'] *
                _dispersion_sum(df.loc[mx_mask, 'K0L'],
                                df.loc[mx_mask, 'K1SL'],
                                df_temp.loc[mx_mask, f'{DISPERSION}{Y}'],
                                df.loc[mx_mask, f'{BETA}{X}'],
                                tau(phase_advances[X].loc[mx_mask, :], qx)).transpose()
        )
        df_res.loc[:, f'{DISPERSION}{Y}'] = (
                df.loc[:, f'{coeff}{Y}'] *
                _dispersion_sum(-df.loc[my_mask, 'K0SL'],  # MINUS!
                                df.loc[my_mask, 'K1SL'],
                                df_temp.loc[my_mask, f'{DISPERSION}{X}'],
                                df.loc[my_mask, f'{BETA}{Y}'],
                                tau(phase_advances[Y].loc[my_mask, :], qy)).transpose()
        )

    for p in PLANES:
        LOG.debug(f"Average linear dispersion D{p}: {df_res[f'{DISPERSION}{p}'].mean():g}")


# Helper -----------------------------------------------------------------------

def _dispersion_coeff(beta, q):
    """ Helper to calculate the coefficient """
    return np.sqrt(beta) / (2 * np.sin(np.pi * q))


def _dispersion_sum(k, j, d, beta, tau):
    """ Helper to calculate the sum """
    # k, j, d , beta = columns -> convert to Series -> broadcasted
    # tau = Matrix as Frame
    calc_column = (k + j * d) * np.sqrt(beta)
    if isinstance(calc_column, pd.DataFrame):
        calc_column = calc_column.squeeze()  # convert single_column DF to Series
    return np.cos(2 * np.pi * tau).mul(calc_column, axis='index').sum(axis='index')
