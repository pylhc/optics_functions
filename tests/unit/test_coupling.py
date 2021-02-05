from pathlib import Path

import numpy as np
import pytest
import tfs

from optics_functions.constants import NAME, S, ALPHA, Y, BETA, X, GAMMA
from optics_functions.coupling import (closest_tune_approach, coupling_from_rdts,
                                       coupling_from_cmatrix, COUPLING_RDTS)
from optics_functions.utils import prepare_twiss_dataframe
from tests.unit.test_rdt import arrays_are_close_almost_everywhere

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.basic
def test_cmatrix():
    n = 5
    df = tfs.TfsDataFrame(0, index=[str(i) for i in range(n)],
    columns=[f"{ALPHA}{X}", f"{ALPHA}{Y}", f"{BETA}{X}", f"{BETA}{Y}", "R11", "R12", "R21", "R22"])
    np.random.seed(487423872)
    r = np.random.rand(n)
    df[S] = np.linspace(0, n, n)
    df.loc[:, "R11"] = np.sin(r)
    df.loc[:, "R22"] = r
    df.loc[:, "R21"] = np.cos(r)
    df.loc[:, "R12"] = -r
    df.loc[:, [f"{BETA}{X}", f"{BETA}{Y}"]] = 1

    df_res = coupling_from_cmatrix(df)
    assert all(c in df_res.columns for c in ("F1001", "F1010", "C11", "C12", "C21", "C22", GAMMA))
    assert not df_res.isna().any().any()

    detC = (df_res["C11"]*df_res["C22"] - df_res["C12"]*df_res["C21"])
    fsq_diff = np.abs(df_res["F1001"])**2 - np.abs(df_res["F1010"])**2
    f_term = 1/(1+4*fsq_diff)
    g_sq = df_res[GAMMA]**2
    assert all(np.abs(detC + g_sq - 1) < 1e-15)
    assert all(np.abs(detC / (4 * g_sq) - fsq_diff) < 1e-15)  # Eq. (13)
    assert all(np.abs(detC + f_term - 1) < 1e-15)  # Eq. (13)
    assert all(np.abs(g_sq - f_term) < 1e-15)  # Eq. (14)


@pytest.mark.basic
def test_closest_tune_approach():
    beam = 1
    df_twiss = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b{beam:d}.coupling_bump.tfs", index=NAME)
    df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss, max_order=7)
    df_cmatrix = coupling_from_cmatrix(df)
    df_twiss[list(COUPLING_RDTS)] = df_cmatrix[list(COUPLING_RDTS)]

    res = dict().fromkeys(('calaga', 'franchi', 'persson', 'persson_alt'))
    for method in res.keys():
        cta = closest_tune_approach(df_twiss, method=method)
        res[method] = np.abs(np.mean(cta))[0]
        assert not cta.isna().any().any()
        assert all(cta[df_cmatrix["F1001"] != 0] != 0)
        assert all(np.abs(cta) >= 0)
        assert err_rel(res[method], res['calaga']) < 1e-1  # ~7% for persson and persson_alt ...


@pytest.mark.extended
def test_coupling_rdt_bump_cmatrix_compare():
    beam = 1
    df_twiss = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b{beam:d}.coupling_bump.tfs", index=NAME)
    df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss, max_order=7)
    df_rdts = coupling_from_rdts(df)
    df_cmatrix = coupling_from_cmatrix(df)

    # from tests.unit.debug_helper import plot_rdts_vs
    # plot_rdts_vs(df_rdts, "analytical", df_cmatrix, "cmatrix", df_twiss, ["F1001", "F1010"])

    for rdt in ("F1010", "F1001"):
        assert arrays_are_close_almost_everywhere(df_rdts[rdt], df_cmatrix[rdt],
                                                  rtol=1e-3, atol=1e-4, percentile=0.99)


# Helper -----------------------------------------------------------------------

def err_rel(a, b):
    return np.abs((a-b)/b)
