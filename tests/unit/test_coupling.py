import string
from pathlib import Path
from optics_functions.constants import NAME, S, ALPHA, Y, BETA, X

import pytest
import tfs
from optics_functions.coupling import closest_tune_approach, coupling_from_rdts, coupling_from_cmatrix, COUPLING_RDTS
from optics_functions.utils import prepare_twiss_dataframe
import numpy as np

from tests.unit.test_rdt import arrays_are_close_almost_everywhere

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.basic
def test_cmatrix():
    n = 5
    df = tfs.TfsDataFrame(0, index=[str(i)  for i in range(n)],
    columns=[f"{ALPHA}{X}", f"{ALPHA}{Y}", f"{BETA}{X}", f"{BETA}{Y}", "R11", "R12", "R21", "R22"])
    r = np.rand(n)
    df[S] = np.linspace(0, n, n)
    df.loc[:, "R11"] = np.sin(r)
    df.loc[:, "R22"] = r
    df.loc[:, "R21"] = np.cos(r)
    df.loc[:, "R12"] = -r
    df.loc[:, [f"{BETA}{X}", f"{BETA}{Y}"]] = 1

    df_res = coupling_from_cmatrix(df)



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
        assert arrays_are_close_almost_everywhere(df_rdts[rdt], df_cmatrix[rdt], rtol=1e-3, atol=1e-4, percentile=0.99)


@pytest.mark.extended
def test_coupling_rdt_bump_rmatrix_compare():
    beam = 1
    df_twiss = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b{beam:d}.coupling_bump.tfs", index=NAME)
    df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss, max_order=7)
    df_cmatrix = coupling_from_cmatrix(df)
    from optics_functions.coupling import coupling_from_r_matrix
    df_rmatrix = coupling_from_r_matrix(df)

    from tests.unit.debug_helper import plot_rdts_vs
    plot_rdts_vs(df_rmatrix, "rmatrix", df_cmatrix, "cmatrix", df_twiss, ["F1001", "F1010"])


@pytest.mark.extended
def test_cmatrix_feeddown():
    beam = 1
    # id_ = "flat_orbit_no_errors"
    # id_ = "flat_orbit_with_errors"
    id_ = "xing_no_errors"
    # id_ = "xing_with_errors"
    df_twiss = tfs.read(INPUT / "twiss_optics" / f"twiss.lhc.b{beam:d}.{id_}.tfs", index=NAME)
    df_twiss[["X", "Y"]] *= 3.5e-5
    df_errors = tfs.read(INPUT / "twiss_optics" / f"errors.lhc.b{beam:d}.unsliced.tfs", index=NAME)
    df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss, df_errors=df_errors, max_order=7)
    # df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss, max_order=7)
    df_rdts = coupling_from_rdts(df, feeddown=1)
    df_cmatrix = coupling_from_cmatrix(df)
    # df_cta_rdts = closest_tune_approach(df_rdts)
    # df_cta_cmatrix = closest_tune_approach(df_rdts)

    from tests.unit.debug_helper import plot_rdts_vs
    plot_rdts_vs(df_rdts, "analytical", df_cmatrix, "cmatrix", df_twiss, ["F1001", "F1010"])


if __name__ == '__main__':
    from tests.unit.debug_helper import enable_logging
    enable_logging()
    test_coupling_rdt_bump_rmatrix_compare()