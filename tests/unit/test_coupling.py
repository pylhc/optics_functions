from pathlib import Path
from optics_functions.constants import NAME, S

import pytest
import tfs
from optics_functions.coupling import closest_tune_approach, coupling_from_rdts, coupling_from_cmatrix, COUPLING_RDTS
from optics_functions.utils import prepare_twiss_dataframe
import numpy as np

from tests.unit.test_rdt import arrays_are_close_almost_everywhere

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.extended
def test_coupling_rdt_cmatrix_compare():
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
def test_coupling_rdt_cmatrix_compare_feeddown():
    beam = 1
    df_twiss = tfs.read(INPUT / "twiss_optics" / f"twiss.lhc.b{beam:d}.unsliced.tfs", index=NAME)
    df_twiss = tfs.read(INPUT / "coupling_bump" / f"twiss.lhc.b{beam:d}.coupling_bump.tfs", index=NAME)
    # df_errors = tfs.read(INPUT / "twiss_optics" / f"errors.lhc.b{beam:d}.unsliced.tfs", index=NAME)
    # df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss, df_errors=df_errors, max_order=7)
    df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss, max_order=7)
    df_rdts = coupling_from_rdts(df, feeddown=2)
    df_cmatrix = coupling_from_cmatrix(df)
    # df_cta_rdts = closest_tune_approach(df_rdts)
    # df_cta_cmatrix = closest_tune_approach(df_rdts)

    # from tests.unit.debug_helper import plot_rdts_vs
    # plot_rdts_vs(df_rdts, "analytical", df_cmatrix, "cmatrix", df_twiss, ["F1001", "F1010"])


if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(levelname)7s | %(message)s | %(name)s"
    )
    test_coupling_rdt_cmatrix_compare()