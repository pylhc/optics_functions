import tfs
from optics_functions import rdts
import numpy as np

def test_rdt_calc():
    df = tfs.read_tfs("tests/data/twiss.tfs")
    f1001 = rdts.calc_fjklm(df, 1, 0, 0, 1, 0.28, 0.31)

    optics_class = tfs.read_tfs("tests/data/f1001_oc.tfs")

    assert(np.allclose(optics_class["F1001R"].values, np.real(f1001)))
    assert(np.allclose(optics_class["F1001I"].values, np.imag(f1001)))

