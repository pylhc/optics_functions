import string
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs

from optics_functions.constants import PHASE_ADV, Y, X, REAL, IMAG, NAME, DELTA_ORBIT, PLANES, S
from optics_functions.utils import (add_missing_columns, dphi,
                                    get_all_phase_advances, tau, seq2str, i_pow,
                                    prepare_twiss_dataframe, switch_signs_for_beam4,
                                    get_format_keys, dphi_at_element, split_complex_columns)

INPUT = Path(__file__).parent.parent / "inputs"


@pytest.mark.basic
def test_add_missing_columns():
    df = pd.DataFrame([1, 2, 3], columns=["Exists"])
    df_new = add_missing_columns(df, ["Exists", "New1", "New2"])

    assert "New1" in df_new
    assert not any(df_new["New1"])

    assert "New2" in df_new
    assert not any(df_new["New2"])

    assert "Exists" in df_new
    assert all(df["Exists"])
    assert all(df["Exists"] == df_new["Exists"])


@pytest.mark.basic
def test_i_pow():
    for i in range(10):
        assert 1j**i == i_pow(i)

    assert 1j**0 == i_pow(1000)
    assert 1j**1 == i_pow(1001)
    assert 1j**2 == i_pow(1002)
    assert 1j**3 == i_pow(1003)
    assert i_pow(1) != i_pow(2)
    assert i_pow(0) == -i_pow(2)
    assert i_pow(1) == -i_pow(3)


@pytest.mark.basic
def test_seq2str():
    seq = ("a", "b", 1, 2)

    def assert_all(str_):
        assert isinstance(str_, str)
        assert all(str(item) in str_ for item in seq)

    assert_all(seq2str(seq))
    assert_all(seq2str(iter(seq)))
    assert_all(seq2str(list(seq)))


@pytest.mark.basic
def test_get_format_keys():
    keys = get_format_keys("{key:d} is a key other is {other:s}, and the {last}{}")
    assert "key" in keys
    assert "other" in keys
    assert "last" in keys
    assert "" in keys


@pytest.mark.basic
def test_get_all_phaseadvances():
    n, qx, qy = 5, 4, 5
    df = phase_df(n, qx=qx, qy=qy)
    phs_adv = get_all_phase_advances(df)
    assert len(phs_adv) == 2
    for q, adv in zip((qx, qy), phs_adv.values()):
        assert adv.shape == (n, n)
        assert not (adv + adv.T).any().any()
        assert adv.loc[0, n-1] == q


@pytest.mark.basic
def test_split_complex_columns():
    df = pd.DataFrame([1+2j, 3j + 4], columns=["Col"], index=["A", "B"])
    df_split = split_complex_columns(df, df.columns, drop=False)
    assert len(df_split.columns) == 3

    df_split = split_complex_columns(df, df.columns, drop=True)
    assert len(df_split.columns) == 2

    for col in df.columns:
        for fun, part in ((np.real, REAL), (np.imag, IMAG)):
            assert (fun(df[col]) == df_split[f"{col}{part}"]).all()


@pytest.mark.basic
def test_dphi():
    n, qx, qy = 5, 4, 5
    df = phase_df(n, qx=qx, qy=qy)
    phs_adv_dict = get_all_phase_advances(df)
    for q, adv in zip((qx, qy), phs_adv_dict.values()):
        dp = dphi(adv, q)
        diff = dp-adv
        assert (dp >= 0).all().all()
        assert (dp <= q).all().all()
        assert ((diff == 0) | (diff == q)).all().all()
        assert all(dp.loc[i, i] == q for i in range(n))


@pytest.mark.basic
def test_tau():
    n, qx, qy = 5, 4, 5
    df = phase_df(n, qx=qx, qy=qy)
    phs_adv_dict = get_all_phase_advances(df)
    for q, adv in zip((qx, qy), phs_adv_dict.values()):
        dp = tau(adv, q)
        diff = dp-adv
        assert (dp >= -q/2).all().all()
        assert (dp <= q/2).all().all()
        assert ((diff == -q/2) | (diff == q/2)).all().all()
        assert all(dp.loc[i, i] == q/2 for i in range(n))


@pytest.mark.basic
def test_dphi_at_element():
    n, qx, qy = 5, 4, 5
    df = phase_df(n, qx=qx, qy=qy)
    phs_adv_dict = get_all_phase_advances(df)
    for element in range(n):
        dphi_element_dict = dphi_at_element(df, element, qx, qy)
        for q, dp_element, adv in zip((qx, qy), dphi_element_dict.values(), phs_adv_dict.values()):
            dp = dphi(adv, q)
            assert all(dp.loc[:, element] == dp_element)


@pytest.mark.basic
def test_prepare_twiss_dataframe():
    n_index, n_kmax, n_valmax = 5, 6, 10
    n_kmax_prepare = 16
    df_twiss, df_errors = get_twiss_and_error_df(n_index, n_kmax, n_valmax)
    df = prepare_twiss_dataframe(beam=1, df_twiss=df_twiss, df_errors=df_errors, max_order=n_kmax_prepare)
    assert all(df[S] == df_twiss[S])
    k_columns = df.columns[df.columns.str.match(r"^K\d+S?L")]
    assert len(k_columns) == n_kmax_prepare * 2
    for col in k_columns:
        if col in df_twiss.columns and col in df_errors.columns:
            assert all(df[col] == (df_twiss[col] + df_errors[col]))
        elif col in df_twiss.columns:
            assert all(df[col] == df_twiss[col])
        elif col in df_errors.columns:
            assert all(df[col] == df_errors[col])
        else:
            assert not any(df[col])

    for plane in PLANES:
        assert all(df[plane] == (df_twiss[plane] + df_errors[f"{DELTA_ORBIT}{plane}"]))

    assert df.headers == df_twiss.headers
    assert not df.isna().any().any()


@pytest.mark.extended
def test_prepare_twiss_dataframe_inner():
    n_index, n_kmax, n_valmax = 5, 6, 10
    n_kmax_prepare = 5
    df_twiss, _ = get_twiss_and_error_df(n_index, n_kmax, n_valmax)
    _, df_errors_1 = get_twiss_and_error_df(n_index+3, n_kmax, n_valmax)
    df = prepare_twiss_dataframe(beam=1, df_twiss=df_twiss, df_errors=df_errors_1.iloc[3:, :],
                                 max_order=n_kmax_prepare, join="inner")

    k_columns = df.columns[df.columns.str.match(r"^K\d+S?L")]
    assert len(k_columns) == n_kmax_prepare * 2

    assert len(df.index) == n_index-3
    assert all(df[S] == df_twiss.loc[df.index, S])

    assert df.headers == df_twiss.headers
    assert not df.isna().any().any()


@pytest.mark.extended
def test_prepare_twiss_dataframe_outer():
    n_index, n_kmax, n_valmax = 8, 6, 10
    n_kmax_prepare = 16
    _, df_errors = get_twiss_and_error_df(n_index, n_kmax, n_valmax)
    df_twiss_1, _ = get_twiss_and_error_df(n_index, n_kmax+2, n_valmax)
    df = prepare_twiss_dataframe(beam=1, df_twiss=df_twiss_1.iloc[3:, :], df_errors=df_errors.iloc[:-3, :],
                                 max_order=n_kmax_prepare, join="outer")

    k_columns = df.columns[df.columns.str.match(r"^K\d+S?L")]
    assert len(k_columns) == n_kmax_prepare * 2

    assert all(df.index == df_twiss_1.index)
    assert all(df[S] == df_twiss_1.loc[df.index, S])

    assert df.headers == df_twiss_1.headers
    assert not df.isna().any().any()


@pytest.mark.extended
def test_prepare_twiss_dataframe_beams():
    n_index, n_kmax, n_valmax = 5, 6, 10
    df_twiss, df_errors = get_twiss_and_error_df(n_index, n_kmax, n_valmax)
    df1 = prepare_twiss_dataframe(beam=1, df_twiss=df_twiss, df_errors=df_errors)
    df2 = prepare_twiss_dataframe(beam=2, df_twiss=df_twiss, df_errors=df_errors)
    df4 = prepare_twiss_dataframe(beam=4, df_twiss=df_twiss, df_errors=df_errors)

    assert df1.equals(df2)
    assert not df1.equals(df4)


@pytest.mark.basic
def test_switch_signs_for_beam4():
    n_index, n_kmax, n_valmax = 5, 6, 10
    df_twiss_b4, df_errors_b4 = get_twiss_and_error_df(n_index, n_kmax, n_valmax)
    df_twiss_b2, df_errors_b2 = switch_signs_for_beam4(df_twiss_b4, df_errors_b4)

    switch_columns = [X,]  # this needs to be correct!!!
    for col in df_twiss_b4.columns:
        sign = -1 if col in switch_columns else 1
        assert df_twiss_b2[col].equals(sign*df_twiss_b4[col])

    switch_columns = [f"{DELTA_ORBIT}{X}"] + [f"K{o:d}{'' if o % 2 else 'S'}L" for o in range(n_kmax)]  # this needs to be correct!!!
    for col in df_errors_b4.columns:
        sign = -1 if col in switch_columns else 1
        assert df_errors_b2[col].equals(sign*df_errors_b4[col])


@pytest.mark.extended
def test_switch_signs_for_beam4_madx_data():
    input_dir = INPUT / "twiss_optics"
    file_twiss = "twiss.lhc.b{:d}.unsliced.tfs"
    file_errors = "errors.lhc.b{:d}.unsliced.tfs"
    df_twiss_b2 = tfs.read(input_dir / file_twiss.format(2), index=NAME)
    df_errors_b2 = tfs.read(input_dir / file_errors.format(2), index=NAME)
    df_twiss_b4 = tfs.read(input_dir / file_twiss.format(4), index=NAME)
    df_errors_b4 = tfs.read(input_dir / file_errors.format(4), index=NAME)

    # Reverse index to compare with beam 2
    df_twiss_b4 = df_twiss_b4.loc[::-1, :]
    df_errors_b4 = df_errors_b4.loc[::-1, :]

    df_twiss_b4switched, df_errors_b4switched = switch_signs_for_beam4(df_twiss_b4, df_errors_b4)
    twiss_cols, err_cols = get_twiss_and_error_columns(6)

    # as the values are not exact and not all signs perfect: check if more signs are equal than before...
    # Not the most impressive test. Other ideas welcome.
    assert ((np.sign(df_twiss_b4switched[twiss_cols]) == np.sign(df_twiss_b2[twiss_cols])).sum().sum()
            >
            (np.sign(df_twiss_b4[twiss_cols]) == np.sign(df_twiss_b2[twiss_cols])).sum().sum() + 0.9*len(df_twiss_b2))

    assert ((np.sign(df_errors_b4switched[err_cols]) == np.sign(df_errors_b2[err_cols])).sum().sum()
            >
            (np.sign(df_errors_b4[err_cols]) == np.sign(df_errors_b2[err_cols])).sum().sum() + len(df_twiss_b2))

# Helper -----------------------------------------------------------------------


def get_twiss_and_error_df(n_index, n_kmax, n_valmax):
    twiss_cols, err_cols = get_twiss_and_error_columns(n_kmax)
    data = np.random.rand(n_index, len(twiss_cols)) * n_valmax
    data[n_index // 2:, :] = -data[n_index // 2:, :]

    df_twiss = tfs.TfsDataFrame(data[:, :len(twiss_cols)],
                                index=list(string.ascii_uppercase[:n_index]),
                                columns=twiss_cols,
                                headers={"Just": "Some", "really": "nice", "header": 1111})
    df_errors = tfs.TfsDataFrame(data[:, :len(err_cols)],
                                 index=list(string.ascii_uppercase[:n_index]),
                                 columns=err_cols)
    df_twiss[S] = np.linspace(0, n_valmax, n_index)
    df_errors[S] = df_twiss[S].copy()
    return df_twiss, df_errors


def phase_df(n, qx, qy):
    phx, phy = f"{PHASE_ADV}{X}", f"{PHASE_ADV}{Y}"
    df = pd.DataFrame(columns=[phx, phy])
    df[phx] = np.linspace(0, qx, n)
    df[phy] = np.linspace(0, qy, n)
    return df


def get_twiss_and_error_columns(max_n):
    k_cols = [f"K{n}{s}L" for n in range(max_n) for s in ('S', '')]
    twiss_cols = [X, Y] + k_cols
    err_cols = [f"{DELTA_ORBIT}{X}", f"{DELTA_ORBIT}{Y}"] + k_cols
    return twiss_cols, err_cols
