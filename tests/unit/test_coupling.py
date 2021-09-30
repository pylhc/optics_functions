from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs

from optics_functions.constants import (
    NAME, S, ALPHA, Y, BETA, X, GAMMA,
    REAL, IMAG, TUNE, PHASE_ADV,
    F1001, F1010
)
from optics_functions.coupling import (
    closest_tune_approach, coupling_via_rdts,
    coupling_via_cmatrix, COUPLING_RDTS, rmatrix_from_coupling,
    check_resonance_relation
)
from optics_functions.utils import prepare_twiss_dataframe
from tests.unit.test_rdt import arrays_are_close_almost_everywhere

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.basic
def test_cmatrix():
    n = 5
    np.random.seed(487423872)
    df = get_df(n)

    df_res = coupling_via_cmatrix(df)
    assert all(c in df_res.columns for c in (F1001, F1010, "C11", "C12", "C21", "C22", GAMMA))
    assert not df_res.isna().any().any()

    # Checks based on CalagaBetatronCoupling2005
    detC = (df_res["C11"] * df_res["C22"] - df_res["C12"] * df_res["C21"])
    fsq_diff = np.abs(df_res[F1001])**2 - np.abs(df_res[F1010])**2
    f_term = 1/(1 + 4 * fsq_diff)
    g_sq = df_res[GAMMA]**2
    assert all(np.abs(detC + g_sq - 1) < 1e-15)
    assert all(np.abs(detC / (4 * g_sq) - fsq_diff) < 1e-15)  # Eq. (13)
    assert all(np.abs(detC + f_term - 1) < 1e-15)  # Eq. (13)
    assert all(np.abs(g_sq - f_term) < 1e-15)  # Eq. (14)


@pytest.mark.basic
@pytest.mark.parametrize('source', ['real', 'fake'])
def test_rmatrix_to_coupling_to_rmatrix(source):
    if source == "fake":
        n = 5
        np.random.seed(487423872)
        df = get_df(n)
    else:
        df = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b1.coupling_bump.tfs", index=NAME)

    df_coupling = coupling_via_cmatrix(df)
    for col in (f"{ALPHA}X", f"{BETA}X", f"{ALPHA}Y", f"{BETA}Y"):
        df_coupling[col] = df[col]

    df_res = rmatrix_from_coupling(df_coupling)

    for col in ("R11", "R12", "R21", "R22"):
        # For debugging:
        # print(col)
        # print(max(np.abs(df[col] - df_res[col])))
        assert all(np.abs(df[col] - df_res[col]) < 5e-15)


@pytest.mark.basic
def test_real_output():
    n = 7
    np.random.seed(474987942)
    df = get_df(n)
    df = prepare_twiss_dataframe(df_twiss=df)
    df.loc[:, "K1L"] = np.random.rand(n)
    df.loc[:, "K1SL"] = np.random.rand(n)

    df_cmatrix = coupling_via_cmatrix(df, complex_columns=False)
    df_rdts = coupling_via_rdts(df, qx=1.31, qy=1.32, complex_columns=False)

    assert all(np.real(df_cmatrix) == df_cmatrix)
    assert all(np.real(df_rdts) == df_rdts)
    columns = [f"{c}{r}" for c in COUPLING_RDTS for r in (REAL, IMAG)]
    assert df_rdts[columns].all().all()
    assert df_cmatrix[columns].all().all()

    assert df_cmatrix.columns.str.match(f".+{REAL}$").sum() == 2
    assert df_cmatrix.columns.str.match(f".+{IMAG}$").sum() == 2
    assert df_rdts.columns.str.match(f".+{REAL}$").sum() == 2
    assert df_rdts.columns.str.match(f".+{IMAG}$").sum() == 2


@pytest.mark.basic
def test_closest_tune_approach():
    desire = namedtuple('desire', ['err', 'isreal'])
    map = {'teapot': desire(0, True),  # results are compared to this, from a madx-match would maybe better
           'calaga': desire(0, True),  # same method as teapot
           'franchi': desire(0.001, True),
           'teapot_franchi': desire(0.0005, True),
           'persson': desire(0.25, False),  # Not sure why it is so high here
           'persson_alt': desire(0.25, False),
           'hoydalsvik': desire(0.25, False),
           'hoydalsvik_alt': desire(0.25, False),
           }

    beam = 1
    df_twiss = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b{beam:d}.coupling_bump.tfs", index=NAME)
    df = prepare_twiss_dataframe(df_twiss=df_twiss, max_order=7)
    df_cmatrix = coupling_via_cmatrix(df)
    df_twiss[F1001] = df_cmatrix[F1001]  # ignoring F1010 in this test as it is bigger than F1010

    res = dict().fromkeys(map.keys())
    err = dict().fromkeys(map.keys())
    for method, desired in map.items():
        cta = closest_tune_approach(df_twiss, method=method)
        res[method] = np.abs(np.mean(cta))[0]
        err[method] = err_rel(res[method], res['teapot'])
        print(f"{method}: {err} (relative error)")
        # assert err[method] <= desired.err

        assert not cta.isna().any().any()
        assert all(cta[df_cmatrix[F1001] != 0] != 0)

        if desired.isreal:
            assert all(np.isreal(cta))


@pytest.mark.basic
def test_check_resonance_relation_with_nan(caplog):
    df = pd.DataFrame([[1, 2, 3, 4], [2, 1, 5, 1]], index=[F1001, F1010]).T
    df_nan = check_resonance_relation(df, to_nan=True)

    assert all(df_nan.loc[0, :].isna())
    assert all(df_nan.loc[2, :].isna())
    assert all(df_nan.loc[1, :] == df.loc[1, :])
    assert all(df_nan.loc[3, :] == df.loc[3, :])

    assert "F1001 < F1010" in caplog.text


@pytest.mark.basic
def test_check_resonance_relation_without_nan(caplog):
    df = pd.DataFrame([[1, 2, 3, 4], [2, 1, 5, 1]], index=[F1001, F1010]).T
    df_out = check_resonance_relation(df, to_nan=False)

    assert (df_out == df).all().all()

    assert "F1001 < F1010" in caplog.text


@pytest.mark.basic
def test_check_resonance_relation_all_good(caplog):
    df = pd.DataFrame([[2, 3, 4, 5], [1, 3, 3, 4]], index=[F1001, F1010]).T
    df_out = check_resonance_relation(df, to_nan=True)

    assert (df_out == df).all().all()

    assert "F1001 < F1010" not in caplog.text


@pytest.mark.extended
def test_coupling_rdt_bump_cmatrix_compare():
    beam = 1
    df_twiss = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b{beam:d}.coupling_bump.tfs", index=NAME)
    df = prepare_twiss_dataframe(df_twiss=df_twiss, max_order=7)
    df_rdts = coupling_via_rdts(df)
    df_cmatrix = coupling_via_cmatrix(df)

    # from tests.unit.debug_helper import plot_rdts_vs
    # plot_rdts_vs(df_rdts, "analytical", df_cmatrix, "cmatrix", df_twiss, ["F1001", "F1010"])

    for rdt in COUPLING_RDTS:
        assert arrays_are_close_almost_everywhere(df_rdts[rdt], df_cmatrix[rdt],
                                                  rtol=1e-3, atol=1e-4, percentile=0.99)


# Helper -----------------------------------------------------------------------


def get_df(n):
    qx, qy = 1.31, 1.32
    df = tfs.TfsDataFrame(0,
                          index=[str(i) for i in range(n)],
                          columns=[
                              S,
                              f"{ALPHA}{X}", f"{ALPHA}{Y}",
                              f"{BETA}{X}", f"{BETA}{Y}",
                              f"{PHASE_ADV}{X}", f"{PHASE_ADV}{Y}",
                              "R11", "R12", "R21", "R22"],
                          headers={f"{TUNE}1": qx,
                                   f"{TUNE}2": qy
                                   }
                          )

    df[S] = np.linspace(0, n, n)
    r = np.random.rand(n)
    df[S] = np.linspace(0, n, n)
    df.loc[:, "R11"] = np.sin(r)
    df.loc[:, "R22"] = r
    df.loc[:, "R21"] = np.cos(r)
    df.loc[:, "R12"] = -r
    df[f"{PHASE_ADV}{X}"] = np.linspace(0, qx, n+1)[:n]
    df[f"{PHASE_ADV}{Y}"] = np.linspace(0, qy, n+1)[:n]
    df.loc[:, [f"{BETA}{X}", f"{BETA}{Y}"]] = 1
    return df


def err_rel(a, b):
    return np.abs((a-b)/b)
