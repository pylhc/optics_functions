from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs
from pandas.testing import assert_frame_equal
from test_rdt import arrays_are_close_almost_everywhere

from optics_functions.constants import (
    ALPHA, BETA, F1001, F1010, GAMMA, IMAG, NAME, PHASE_ADV, REAL, TUNE, S, X, Y
)
from optics_functions.coupling import (
    COUPLING_RDTS,
    check_resonance_relation,
    closest_tune_approach,
    coupling_via_cmatrix,
    coupling_via_rdts,
    rmatrix_from_coupling,
)
from optics_functions.utils import prepare_twiss_dataframe

INPUT = Path(__file__).parent.parent / "inputs"
COUPLING_BUMP_INPUTS = INPUT / "coupling_bump"
COUPLING_BUMP_TWISS_BEAM_1 = COUPLING_BUMP_INPUTS / "twiss.lhc.b1.coupling_bump.tfs"


@pytest.mark.basic
def test_cmatrix():
    n = 5
    np.random.seed(487423872)
    df = generate_fake_data(n)

    df_res = coupling_via_cmatrix(df)
    assert all(c in df_res.columns for c in (F1001, F1010, "C11", "C12", "C21", "C22", GAMMA))
    assert not df_res.isna().any().any()

    # Checks based on CalagaBetatronCoupling2005
    detC = df_res["C11"] * df_res["C22"] - df_res["C12"] * df_res["C21"]
    fsq_diff = df_res[F1001].abs() ** 2 - df_res[F1010].abs() ** 2
    f_term = 1 / (1 + 4 * fsq_diff)
    g_sq = df_res[GAMMA] ** 2

    assert all(np.abs(detC + g_sq - 1) < 1e-15)
    assert all(np.abs(detC / (4 * g_sq) - fsq_diff) < 1e-15)  # Eq. (13)
    assert all(np.abs(detC + f_term - 1) < 1e-15)  # Eq. (13)
    assert all(np.abs(g_sq - f_term) < 1e-15)  # Eq. (14)


@pytest.mark.basic
@pytest.mark.parametrize("source", ["real", "fake"])
def test_rmatrix_to_coupling_to_rmatrix(source):
    if source == "fake":
        np.random.seed(487423872)
        df = generate_fake_data(5)
    else:
        df = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b1.coupling_bump.tfs", index=NAME)

    df_coupling = coupling_via_cmatrix(df)
    for col in (f"{ALPHA}X", f"{BETA}X", f"{ALPHA}Y", f"{BETA}Y"):
        df_coupling[col] = df[col]

    df_res = rmatrix_from_coupling(df_coupling)

    for col in ("R11", "R12", "R21", "R22"):
        # print(col, "\n", max(np.abs(df[col] - df_res[col])))  # for debugging
        assert all(np.abs(df[col] - df_res[col]) < 5e-15)


@pytest.mark.basic
def test_real_output():
    n = 7
    np.random.seed(474987942)
    df = generate_fake_data(n)
    df = prepare_twiss_dataframe(df_twiss=df)
    df.loc[:, "K1L"] = np.random.rand(n)
    df.loc[:, "K1SL"] = np.random.rand(n)

    df_cmatrix = coupling_via_cmatrix(df, complex_columns=False)
    df_rdts = coupling_via_rdts(df, qx=1.31, qy=1.32, complex_columns=False)

    assert all(np.real(df_cmatrix) == df_cmatrix)
    assert all(np.real(df_rdts) == df_rdts)
    columns = [f"{c}{r}" for c in COUPLING_RDTS for r in (REAL, IMAG)]
    assert df_rdts[columns].all().all()  # check no 0-values
    assert df_cmatrix[columns].all().all()  # check no 0-values

    assert df_cmatrix.columns.str.match(f".+{REAL}$").sum() == 2
    assert df_cmatrix.columns.str.match(f".+{IMAG}$").sum() == 2
    assert df_rdts.columns.str.match(f".+{REAL}$").sum() == 2
    assert df_rdts.columns.str.match(f".+{IMAG}$").sum() == 2


@pytest.mark.basic
@pytest.mark.parametrize(
    "cta_method, max_relative_error_to_teapot, result_should_be_real",
    [
        ("calaga", 0, True),  # this is the same as teapot, hence 0 relative error
        ("franchi", 0.001, True),
        ("teapot_franchi", 0.0005, True),
        ("persson", 0.25, False),  # not sure why it is so high from here
        ("persson_alt", 0.25, False),
        ("hoydalsvik", 0.25, False),
        ("hoydalsvik_alt", 0.25, False),
    ],
)
def test_closest_tune_approach(
    cta_method, max_relative_error_to_teapot, result_should_be_real, _coupling_bump_teapot_cta
):
    df_twiss = tfs.read(COUPLING_BUMP_TWISS_BEAM_1, index=NAME)
    df = prepare_twiss_dataframe(df_twiss=df_twiss, max_order=7)
    df_cmatrix = coupling_via_cmatrix(df)
    df_twiss[F1001] = df_cmatrix[F1001]  # ignoring F1010 in this test as it is bigger than F1001

    cta_df = closest_tune_approach(df_twiss, method=cta_method)  # only one column
    cminus = cta_df.mean().abs()[0]
    relative_error = _relative_error(cminus, _coupling_bump_teapot_cta)

    assert relative_error <= max_relative_error_to_teapot
    assert not cta_df.isna().any().any()  # check no NaNs
    assert all(cta_df[df_cmatrix[F1001] != 0] != 0)

    if result_should_be_real:
        assert all(np.isreal(cta_df))


@pytest.mark.basic
def test_check_resonance_relation_with_nan(caplog):
    df = pd.DataFrame([[1, 2, 3, 4], [2, 1, -5, 1]], index=[F1001, F1010]).T
    df_nan = check_resonance_relation(df, to_nan=True)

    assert all(df_nan.loc[0, :].isna())
    assert all(df_nan.loc[2, :].isna())
    assert all(df_nan.loc[1, :] == df.loc[1, :])
    assert all(df_nan.loc[3, :] == df.loc[3, :])
    assert "|F1001| < |F1010|" in caplog.text


@pytest.mark.basic
def test_check_resonance_relation_without_nan(caplog):
    df = pd.DataFrame([[1, 2, 3, 4], [2, 1, 5, 1]], index=[F1001, F1010]).T
    df_out = check_resonance_relation(df, to_nan=False)

    assert_frame_equal(df_out, df)
    assert "|F1001| < |F1010|" in caplog.text


@pytest.mark.basic
def test_check_resonance_relation_all_good(caplog):
    df = pd.DataFrame([[2, 3, 4, 5], [1, 3, 3, 4]], index=[F1001, F1010]).T
    df_out = check_resonance_relation(df, to_nan=True)

    assert_frame_equal(df_out, df)
    assert "|F1001| < |F1010|" not in caplog.text


@pytest.mark.extended
def test_coupling_rdt_bump_cmatrix_compare():
    df_twiss = tfs.read(COUPLING_BUMP_TWISS_BEAM_1, index=NAME)
    df = prepare_twiss_dataframe(df_twiss=df_twiss, max_order=7)
    df_rdts = coupling_via_rdts(df)
    df_cmatrix = coupling_via_cmatrix(df)

    # from tests.unit.debug_helper import plot_rdts_vs
    # plot_rdts_vs(df_rdts, "analytical", df_cmatrix, "cmatrix", df_twiss, ["F1001", "F1010"])

    for rdt in COUPLING_RDTS:
        assert arrays_are_close_almost_everywhere(
            df_rdts[rdt], df_cmatrix[rdt], rtol=1e-3, atol=1e-4, percentile=0.99
        )


# ----- Helpers ----- #


def generate_fake_data(n) -> tfs.TfsDataFrame:
    qx, qy = 1.31, 1.32
    df = tfs.TfsDataFrame(0,
                          index=[str(i) for i in range(n)],
                          columns=[S, f"{ALPHA}{X}", f"{ALPHA}{Y}", f"{BETA}{X}", f"{BETA}{Y}",
                                   f"{PHASE_ADV}{X}", f"{PHASE_ADV}{Y}", "R11", "R12", "R21", "R22"],
                          headers={f"{TUNE}1": qx, f"{TUNE}2": qy}
                          )

    r = np.random.rand(n)
    df[S] = np.linspace(0, n, n)
    df.loc[:, "R11"] = np.sin(r)
    df.loc[:, "R22"] = r
    df.loc[:, "R21"] = np.cos(r)
    df.loc[:, "R12"] = -r
    df[f"{PHASE_ADV}{X}"] = np.linspace(0, qx, n + 1)[:n]
    df[f"{PHASE_ADV}{Y}"] = np.linspace(0, qy, n + 1)[:n]
    df.loc[:, [f"{BETA}{X}", f"{BETA}{Y}"]] = 1
    return df


def _relative_error(a, b):
    return np.abs((a - b) / b)


# ----- Fixtures ----- #


@pytest.fixture(scope="module")
def _coupling_bump_teapot_cta() -> float:
    """Compute and return the CTA for teapot reference method on the lhcb1 coupling bump test case."""
    df_twiss = tfs.read(COUPLING_BUMP_TWISS_BEAM_1, index=NAME)
    df = prepare_twiss_dataframe(df_twiss=df_twiss, max_order=7)
    df_cmatrix = coupling_via_cmatrix(df)
    df_twiss[F1001] = df_cmatrix[F1001]  # ignoring F1010 in this test as it is bigger than F1001

    cta_df = closest_tune_approach(df_twiss, method="teapot")  # only one column
    return cta_df.mean().abs()[0]  # this is the cminus
