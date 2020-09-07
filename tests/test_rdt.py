import tfs
from optics_functions import rdts
import numpy as np


def test_rdt_calc_against_optics_class():
    _test_coupl_rdts("o")


def test_rdt_calc_against_metaclass():
    _test_coupl_rdts("m")


def _test_coupl_rdts(reference: str):
    """[summary]

    Args:
        reference (str): o for optics_class, m for metaclass
    """
    if reference not in ["m", "o"]:
        raise RuntimeError("reference must be one of ['m', 'o']")
    optics_class = tfs.read_tfs(f"tests/data/rdts_{reference}c.tfs", index="NAME")

    df = tfs.read_tfs("tests/data/twiss.tfs", index="NAME")
    index = df.index.intersection(optics_class.index)
    f1001 = rdts.calc_fjklm(df, 1, 0, 0, 1, 0.28, 0.31, index)
    f1010 = rdts.calc_fjklm(df, 1, 0, 1, 0, 0.28, 0.31, index)

    if reference == 'm':
        rtol = 2.0e-2
        atol = 1.0e-2
        swap = -1.0
    else:
        rtol = 1.0e-10
        atol = 1.0e-10
        swap = 1.0

    diff = optics_class["F1001R"]/np.real(f1001)-1
    for i in range(len(f1001)):
        if np.abs(diff[i]) > 0.02:
            print(f"{i}: diff = {diff}")

    assert(np.allclose(swap*optics_class["F1001R"], np.real(f1001), rtol=rtol, atol=atol))
    assert(np.allclose(optics_class["F1001I"], np.imag(f1001), rtol=rtol, atol=atol))
    assert(np.allclose(swap*optics_class["F1010R"], np.real(f1010), rtol=rtol, atol=atol))
    assert(np.allclose(optics_class["F1010I"], np.imag(f1010), rtol=rtol, atol=atol))
