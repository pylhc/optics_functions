from pathlib import Path
from optics_functions.constants import NAME

import pytest
import tfs
from coupling import closest_tune_approach, coupling_from_rdts, coupling_from_cmatrix, COUPLING_RDTS

from optics_functions.coupling import closest_tune_approach, coupling_from_rdts, coupling_from_cmatrix

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.extended
def test_coupling_rdt_cmatrix_compare():
    df_twiss = tfs.read(INPUT / "twiss_optics" / "twiss.lhc.b1.unsliced.tfs", index=NAME)
    df_rdts = coupling_from_rdts(df_twiss, feeddown=3)
    df_cmatrix = coupling_from_cmatrix(df_twiss)
    # df_cta_rdts = closest_tune_approach(df_rdts)
    # df_cta_cmatrix = closest_tune_approach(df_rdts)

    for rdt in COUPLING_RDTS:
        assert all(df_rdts[rdt] == df_cmatrix[rdt])


if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(levelname)7s | %(message)s | %(name)s"
    )
    test_coupling_rdt_cmatrix_compare()