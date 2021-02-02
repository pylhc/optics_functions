from pathlib import Path
from optics_functions.constants import NAME

import pytest
import tfs
from optics_functions.coupling import closest_tune_approach, coupling_from_rdts, coupling_from_cmatrix, COUPLING_RDTS
from optics_functions.utils import prepare_twiss_dataframe

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.extended
def test_coupling_rdt_cmatrix_compare():
    beam = 1
    df_twiss = tfs.read(INPUT / "twiss_optics" / f"twiss.lhc.b{beam:d}.unsliced.tfs", index=NAME)
    df_errors = tfs.read(INPUT / "twiss_optics" / f"errors.lhc.b{beam:d}.unsliced.tfs", index=NAME)
    df = prepare_twiss_dataframe(beam=beam, df_twiss=df_twiss)
    df_rdts = coupling_from_rdts(df, feeddown=3)
    df_cmatrix = coupling_from_cmatrix(df)
    # df_cta_rdts = closest_tune_approach(df_rdts)
    # df_cta_cmatrix = closest_tune_approach(df_rdts)

    for rdt in COUPLING_RDTS:
        assert all((df_rdts[rdt] - df_cmatrix[rdt]).abs() < 1e-15)


if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(levelname)7s | %(message)s | %(name)s"
    )
    test_coupling_rdt_cmatrix_compare()