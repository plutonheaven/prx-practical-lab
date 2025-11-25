import numpy as np
import pandas as pd
import pytest
from src.constants import C_LIGHT_MPS


def test_differential_carrier_corrections():
    from src.gnss import apply_differential_corrections

    df1 = pd.DataFrame(
        {
            "time_of_reception_in_receiver_time": [0, 0, 1, 1],
            "constellation": ["G"] * 4,
            "prn": [1, 2, 1, 2],
            "rnx_obs_identifier": ["1C"] * 4,
            "C_obs_m": [3, 4, 5, 6],
            "L_obs_cycles": [
                cycles * 1575.42e6 / C_LIGHT_MPS for cycles in [3, 4, 5, 6]
            ],
            "carrier_frequency_hz": [1575.42e6] * 4,
            "sat_pos_x_m": [0] * 4,
            "sat_pos_y_m": [0] * 4,
            "sat_pos_z_m": [0] * 4,
        }
    )
    df2 = pd.DataFrame(
        {
            "time_of_reception_in_receiver_time": [0, 0, 1, 1],
            "constellation": ["G"] * 4,
            "prn": [1, 2, 1, 2],
            "rnx_obs_identifier": ["1C"] * 4,
            "C_obs_m": [3, 4, 5, 6],
            "L_obs_cycles": [
                cycles * 1575.42e6 / C_LIGHT_MPS for cycles in [3, 4, 5, 6]
            ],
            "carrier_frequency_hz": [1575.42e6] * 4,
            "sat_pos_x_m": [0] * 4,
            "sat_pos_y_m": [0] * 4,
            "sat_pos_z_m": [0] * 4,
        }
    )
    pos_base = np.array([0] * 3)

    df_result = apply_differential_corrections(df1, df2, pos_base)
    assert "L_obs_corr_m" in df_result.columns, (
        "'L_obs_corr_m' not in the resulting dataframe "
    )
    assert df_result.L_obs_corr_m.to_numpy() == pytest.approx(np.array([0] * 4))

    # remove a carrier observation in the base receiver
    df_result2 = apply_differential_corrections(df1, df2.drop(index=[1]), pos_base)
    assert not (df_result2.L_obs_corr_m.isna()).any(), (
        "missing row should not result in NaN"
    )
    assert len(df_result2) == 3, (
        "Only the rows common to both dataframes should be present"
    )
    assert df_result2.L_obs_corr_m.to_numpy() == pytest.approx(np.array([0] * 3))

    # remove a carrier observation in the rover receiver
    df1.at[df1.index[0], "L_obs_cycles"] = np.nan
    df_result3 = apply_differential_corrections(df1, df2, pos_base)
    assert not (df_result3.L_obs_corr_m.isna()).any(), (
        "missing row should not result in NaN"
    )
    assert len(df_result3) == 3, (
        "Only the rows common to both dataframes should be present"
    )
    assert df_result3.L_obs_corr_m.to_numpy() == pytest.approx(np.array([0] * 3))


def test_combine_lli():
    from src.gnss import combine_lli

    df1 = pd.DataFrame(
        {
            "time_of_reception_in_receiver_time": [0, 0, 1, 1],
            "constellation": ["G"] * 4,
            "prn": [1, 2, 1, 2],
            "rnx_obs_identifier": ["1C"] * 4,
            "LLI": [0, 0, 0, 0],
        }
    )
    df2 = pd.DataFrame(
        {
            "time_of_reception_in_receiver_time": [0, 0, 1, 1],
            "constellation": ["G"] * 4,
            "prn": [1, 2, 1, 2],
            "rnx_obs_identifier": ["1C"] * 4,
            "LLI": [0, 0, 0, 0],
        }
    )
    df_result = combine_lli(df1, df2)
    for col in [
        "LLI",
        "time_of_reception_in_receiver_time",
        "constellation",
        "prn",
        "rnx_obs_identifier",
    ]:
        assert col in df_result.columns, f"{col} not in the resulting dataframe "
    assert df_result.LLI.to_numpy() == pytest.approx(np.array([0] * 4))

    # insert a cycle slip on rover at epoch 0, prn 1
    df1.at[df1.index[0], "LLI"] = 1
    df_result2 = combine_lli(df1, df2)
    assert df_result2.LLI.to_numpy() == pytest.approx(np.array([1, 0, 0, 0]))

    # insert a cycle slip on base at epoch 1, prn 2
    df2.at[df2.index[3], "LLI"] = 1
    df_result3 = combine_lli(df1, df2)
    assert df_result3.LLI.to_numpy() == pytest.approx(np.array([1, 0, 0, 1]))

    # insert a nan on rover at epoch 0, prn 2
    df2.at[df2.index[1], "LLI"] = np.nan
    df_result4 = combine_lli(df1, df2)
    assert df_result4.LLI.to_numpy() == pytest.approx(
        np.array([1, np.nan, 0, 1]), nan_ok=True
    )


def test_corrected_carrier_model():
    from src.gnss import obs_model_corrected_carrier

    # test geometric distance
    df = pd.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    amb = np.array([0, 0, 0])
    assert (
        obs_model_corrected_carrier(df, rx_pos, rx_clk, amb) == np.array([1.0] * 3)
    ).all(), "geometric distance wrongly computed"

    # test rx clock bias
    rx_clk = 1
    assert (
        obs_model_corrected_carrier(df, rx_pos, rx_clk, amb) == np.array([2.0] * 3)
    ).all(), "issue with rx_clk"
    assert (
        obs_model_corrected_carrier(df, rx_pos, amb=amb) == np.array([1.0] * 3)
    ).all(), "rx_clk default value should be 0"

    # test amb
    amb = np.array([1, 2, 3])
    assert (
        obs_model_corrected_carrier(df, rx_pos, amb=amb) == np.array([2.0, 3.0, 4.0])
    ).all(), "issue with amb"
    assert (
        obs_model_corrected_carrier(df, rx_pos, amb=None) == np.array([1.0] * 3)
    ).all(), "rx_clk default value should be 0"

    # test presence of corrections
    df = pd.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "sat_clock_offset_m": [1] * 3,
            "sagnac_effect_m": [1] * 3,
            "relativistic_clock_effect_m": [1] * 3,
            "sat_code_bias_m": [1] * 3,
            "iono_delay_m": [1] * 3,
            "tropo_delay_m": [1] * 3,
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    amb = np.array([0, 0, 0])
    assert (
        obs_model_corrected_carrier(df, rx_pos, rx_clk, amb) == np.array([1.0] * 3)
    ).all()


def test_jacobian_code_batch():
    from src.gnss import jacobian_code_batch

    df_epoch = pd.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
        }
    )
    rx_pos = [0, 0, 0]
    n_epoch = 3
    prns = [1, 2, 3]

    # jac computed for observations at epoch 0
    idx_epoch = 0
    jac = jacobian_code_batch(df_epoch, rx_pos, idx_epoch, n_epoch, prns)
    assert jac.shape == (3, 4 * 3 + 3), f"Jacobian matrix has wrong shape: {jac.shape}"
    assert jac == pytest.approx(
        np.concatenate(
            [
                -np.eye(3),  # epoch 0, pos states
                np.full((3, 1), 1),  # epoch 0, clk state
                np.zeros((3, 4)),  # epoch 1
                np.zeros((3, 4)),  # epoch 2
                np.zeros((3, 3)),  # amb states
            ],
            axis=1,
        )
    )

    # jac computed for observations at epoch 1
    idx_epoch = 1
    jac = jacobian_code_batch(df_epoch, rx_pos, idx_epoch, n_epoch, prns)
    assert jac == pytest.approx(
        np.concatenate(
            [
                np.zeros((3, 4)),  # epoch 0
                -np.eye(3),  # epoch 1, pos states
                np.full((3, 1), 1),  # epoch 1, clk state
                np.zeros((3, 4)),  # epoch 2
                np.zeros((3, 3)),  # amb states
            ],
            axis=1,
        )
    )

    # jac computed for observations at epoch 2
    idx_epoch = 2
    jac = jacobian_code_batch(df_epoch, rx_pos, idx_epoch, n_epoch, prns)
    assert jac == pytest.approx(
        np.concatenate(
            [
                np.zeros((3, 4)),  # epoch 0
                np.zeros((3, 4)),  # epoch 1
                -np.eye(3),  # epoch 2, pos states
                np.full((3, 1), 1),  # epoch 2, clk state
                np.zeros((3, 3)),  # amb states
            ],
            axis=1,
        )
    )


def test_jacobian_carrier_batch():
    from src.gnss import jacobian_carrier_batch

    df_epoch = pd.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "prn": [1, 2, 3],
        }
    )
    rx_pos = [0, 0, 0]
    n_epoch = 3
    prns = [1, 2, 3]

    # jac computed for observations at epoch 0
    idx_epoch = 0
    jac = jacobian_carrier_batch(df_epoch, rx_pos, idx_epoch, n_epoch, prns)
    assert jac.shape == (3, 4 * 3 + 3), f"Jacobian matrix has wrong shape: {jac.shape}"
    assert jac == pytest.approx(
        np.concatenate(
            [
                -np.eye(3),  # epoch 0, pos states
                np.full((3, 1), 1),  # epoch 0, clk state
                np.zeros((3, 4)),  # epoch 1
                np.zeros((3, 4)),  # epoch 2
                np.eye(3),  # amb states
            ],
            axis=1,
        )
    )

    # jac computed for observations at epoch 1
    idx_epoch = 1
    jac = jacobian_carrier_batch(df_epoch, rx_pos, idx_epoch, n_epoch, prns)
    assert jac == pytest.approx(
        np.concatenate(
            [
                np.zeros((3, 4)),  # epoch 0
                -np.eye(3),  # epoch 1, pos states
                np.full((3, 1), 1),  # epoch 1, clk state
                np.zeros((3, 4)),  # epoch 2
                np.eye(3),  # amb states
            ],
            axis=1,
        )
    )

    # jac computed for observations at epoch 2
    idx_epoch = 2
    jac = jacobian_carrier_batch(df_epoch, rx_pos, idx_epoch, n_epoch, prns)
    assert jac == pytest.approx(
        np.concatenate(
            [
                np.zeros((3, 4)),  # epoch 0
                np.zeros((3, 4)),  # epoch 1
                -np.eye(3),  # epoch 2, pos states
                np.full((3, 1), 1),  # epoch 2, clk state
                np.eye(3),  # amb states
            ],
            axis=1,
        )
    )
