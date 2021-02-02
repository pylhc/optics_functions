import string
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import tfs

from optics_functions.constants import PHASE_ADV, X, Y, BETA, S, TUNE, NAME
from optics_functions.rdt import rdts, generator, get_all_to_order, str2jklm, jklm2str
from optics_functions.utils import prepare_twiss_dataframe

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.basic
def test_jklm_str():
    n = 5
    rdt_list = get_all_to_order(n)
    rdt_list2 = [str2jklm(jklm2str(*r)) for r in rdt_list]
    assert rdt_list == rdt_list2
    assert (sum(str2jklm(r)) == 3 for r in ["F1002", "F1101", "F1020"])
    assert (sum(str2jklm(r)) == 4 for r in ["F2002", "F2011", "F1201"])


@pytest.mark.basic
def test_get_all_to_order():
    n = 5
    rdt_list = get_all_to_order(n)
    assert all(sum(jklm) <= n for jklm in rdt_list)
    assert len(set(rdt_list)) == len(rdt_list)


@pytest.mark.basic
def test_generator():
    n = 5
    rdt_all = generator(list(range(2, n+1)))
    rdt_skew = generator(list(range(2, n+1)), normal=False)
    rdt_normal = generator(list(range(2, n+1)), skew=False)
    rdt_cc = generator(list(range(2, n+1)), complex_conj=False)

    for o in rdt_all.keys():
        assert len(rdt_all[o]) == 2*len(rdt_skew[o])
        assert len(rdt_all[o]) == 2*len(rdt_normal[o])
        assert len(rdt_all[o]) == 2*len(rdt_cc[o])
        assert set(rdt_all[o]) == set(rdt_skew[o] + rdt_normal[o])
        assert all(is_odd(sum(rdt[2:4])) for rdt in rdt_skew[o])
        assert all(is_even(sum(rdt[2:4])) for rdt in rdt_normal[o])
        rdt_cc2 = set(rdt_all[o]) - set(rdt_cc[o])
        assert all((r[1], r[0], r[3], r[2]) in rdt_cc[o] for r in rdt_cc2)


@pytest.mark.basic
def test_generator_vs_all_to_order():
    n = 5
    rdt_dict = generator(list(range(2, n+1)))
    rdt_list = get_all_to_order(n)
    rdt_list2 = []
    for o, v in rdt_dict.items():
        rdt_list2 += v

    set_rdts = set(rdt_list)
    set_rdts2 = set(rdt_list2)
    assert not len(set_rdts2 - set_rdts)
    assert not len(set_rdts - set_rdts2)


@pytest.mark.basic
def test_rdts_normal_sextupole_bump():
    df = get_df(n=7)
    df = prepare_twiss_dataframe(beam=1, df_twiss=df)
    df.loc["C", "K2L"] = 1
    df.loc["F", "K2L"] = -1

    df_rdts = rdts(df, rdts=["F1002", "F2001"])
    df_diff, df_jump = get_absdiff_and_jumps(df_rdts)
    assert not df_jump["F2001"].any()
    assert df_jump.loc[["C", "F"], "F1002"].all()
    assert not df_jump.loc[df_jump.index.difference(["C", "F"]), "F1002"].any()


@pytest.mark.basic
def test_rdts_skew_octupole_bump():
    df = get_df(n=7)
    df = prepare_twiss_dataframe(beam=1, df_twiss=df)
    df.loc["C", "K3SL"] = 1
    df.loc["F", "K3SL"] = -1

    df_rdts = rdts(df, rdts=["F4000", "F3001"])
    df_diff, df_jump = get_absdiff_and_jumps(df_rdts)
    assert not df_jump["F4000"].any()
    assert df_jump.loc[["C", "F"], "F3001"].all()
    assert not df_jump.loc[df_jump.index.difference(["C", "F"]), "F3001"].any()


@pytest.mark.basic
def test_rdts_normal_octuple_to_sextupole_feeddown():
    df = get_df(n=7)
    df = prepare_twiss_dataframe(beam=1, df_twiss=df)
    df_comp = df.copy()

    df.loc["C", "K3L"] = 1
    df.loc["F", "K3L"] = -1

    df_comp.loc["C", ["K2L", "K2SL"]] = 1
    df_comp.loc["F", ["K2L", "K2SL"]] = -1

    df_rdts = rdts(df, rdts=["F1002", "F2001"], feeddown=1)
    df_rdts_comp = rdts(df_comp, rdts=["F1002", "F2001"])

    assert not df_rdts["F2001"].any()
    assert not df_rdts["F1002"].any()

    # Feed-down K3L -> K2L
    df["X"] = 1
    df["Y"] = 0
    df_rdts = rdts(df, rdts=["F1002", "F2001"], feeddown=1)
    assert not df_rdts["F2001"].any()
    assert df_rdts["F1002"].all()
    assert all(df_rdts["F1002"] == df_rdts_comp["F1002"])

    # Feed-down K3L -> K2SL
    df["X"] = 0
    df["Y"] = 1
    df_rdts = rdts(df, rdts=["F1002", "F2001"], feeddown=1)
    assert not df_rdts["F1002"].any()
    assert df_rdts["F2001"].all()
    assert all(df_rdts["F2001"] == df_rdts_comp["F2001"])

    # Feed-down K3L -> K2L, K2SL
    df["X"] = 1
    df["Y"] = 1
    df_rdts = rdts(df, rdts=["F1002", "F2001"], feeddown=1)
    assert all(df_rdts["F1002"] == df_rdts_comp["F1002"])
    assert all(df_rdts["F2001"] == df_rdts_comp["F2001"])


@pytest.mark.basic
def test_rdts_normal_dodecapole_to_octupole_feeddown():
    df = get_df(n=7)
    df = prepare_twiss_dataframe(beam=1, df_twiss=df)
    df_comp = df.copy()

    df.loc["C", "K5L"] = 1
    df.loc["F", "K5L"] = -1

    df_comp.loc["C", ["K3L", "K3SL"]] = 1
    df_comp.loc["F", ["K3L", "K3SL"]] = -1

    df_rdts = rdts(df, rdts=["F1003", "F0004"], feeddown=2)
    df_rdts_comp = rdts(df_comp, rdts=["F1003", "F0004"])

    assert not df_rdts["F1003"].any()
    assert not df_rdts["F0004"].any()

    # Feed-down K5L -> (x**2-y**2)/2 K3L , xy K3SL
    # Feed-down K5L -> K3L
    df["X"] = 1
    df["Y"] = 0
    df_rdts = rdts(df, rdts=["F1003", "F0004"], feeddown=2)
    assert not df_rdts["F1003"].any()
    assert df_rdts["F0004"].all()
    assert all(df_rdts["F0004"] == 0.5 * df_rdts_comp["F0004"])

    # Feed-down K5L -> K3L
    df["X"] = 0
    df["Y"] = 1
    df_rdts = rdts(df, rdts=["F1003", "F0004"], feeddown=2)
    assert not df_rdts["F1003"].any()
    assert df_rdts["F0004"].all()
    assert all(df_rdts["F0004"] == -0.5 * df_rdts_comp["F0004"])

    # Feed-down K5L -> K3SL
    df["X"] = 1
    df["Y"] = 1
    df_rdts = rdts(df, rdts=["F1003", "F0004"], feeddown=2)
    assert not df_rdts["F0004"].any()
    assert df_rdts["F1003"].all()
    assert all(df_rdts["F1003"] == df_rdts_comp["F1003"])

    # Feed-down K5L -> (x**2-y**2)/2 K3L , xy K3SL
    df["X"] = 1
    df["Y"] = 0.5
    df_rdts = rdts(df, rdts=["F1003", "F0004"], feeddown=2)
    assert all((df_rdts["F0004"] - 0.375*df_rdts_comp["F0004"]).abs() <= 1e-15)
    assert all(df_rdts["F1003"] == 0.5*df_rdts_comp["F1003"])


@pytest.mark.extended
def test_sextupole_bump():
    input_dir = INPUT / "sextupole_bump"
    df_twiss = tfs.read(input_dir / "ptc_twiss.lhc.b1.sextupole_bump.tfs", index=NAME)
    df_ptc_rdt = tfs.read(input_dir / "ptc_rdt.lhc.b1.sextupole_bump.tfs", index=NAME)
    df_rdt = rdts(df_twiss, ["F2001", "F3000", "F4000", "F3001"])
    pass

# Helper -----------------------------------------------------------------------


def get_df(n):
    """ Fake DF with nonsense values. """
    qx, qy = 1.31, 1.32
    phx, phy = f"{PHASE_ADV}{X}", f"{PHASE_ADV}{Y}"
    betax, betay = f"{BETA}{X}", f"{BETA}{Y}"
    df = tfs.TfsDataFrame(
        index=list(string.ascii_uppercase[:n]),
        columns=[S, betax, betay, phx, phy],
        headers={f"{TUNE}1": qx, f"{TUNE}2": qy}
    )
    df[S] = np.linspace(0, n, n)
    df[phx] = np.linspace(0, qx, n+1)[:n]
    df[phy] = np.linspace(0, qy, n+1)[:n]
    df[betax] = 1
    df[betay] = 1
    return df


def get_absdiff_and_jumps(df_rdts):
    df_temp = df_rdts.append(pd.DataFrame(df_rdts.iloc[[0], :].to_numpy(), index=["temp"]))
    df_diff = df_temp.abs().diff().shift(-1).iloc[:-1, :]
    df_jump = df_diff.abs() > 1e-15
    return df_diff, df_jump


def is_odd(n: int):
    return bool(n % 2)


def is_even(n: int):
    return not bool(n % 2)


if __name__ == '__main__':
    import logging, sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(levelname)7s | %(message)s | %(name)s"
    )
    test_sextupole_bump()
