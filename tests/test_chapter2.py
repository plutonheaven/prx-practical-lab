import numpy as np
import pandas as pd
import pytest


def test_differential_code_corrections():
    from src.gnss import apply_differential_corrections

    df1 = pd.DataFrame(
        {
            "time_of_reception_in_receiver_time": [0, 0, 1, 1],
            "constellation": ["G"] * 4,
            "prn": [1, 2, 1, 2],
            "rnx_obs_identifier": ["1C"] * 4,
            "C_obs_m": [3, 4, 5, 6],
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
            "sat_pos_x_m": [0] * 4,
            "sat_pos_y_m": [0] * 4,
            "sat_pos_z_m": [0] * 4,
        }
    )
    pos_base = np.array([0] * 3)

    df_result = apply_differential_corrections(df1, df2, pos_base)
    assert "C_obs_corr_m" in df_result.columns
    assert df_result.C_obs_corr_m.to_numpy() == pytest.approx(np.array([0] * 4))

    # remove one row from df2
    df_result2 = apply_differential_corrections(df1, df2.drop(index=[1]), pos_base)
    assert not (df_result2.C_obs_corr_m.isna()).any(), (
        "missing row should not result in NaN"
    )
    assert len(df_result2) == 3, (
        "Only the rows common to both dataframes should be present"
    )
    assert df_result2.C_obs_corr_m.to_numpy() == pytest.approx(np.array([0] * 3))


def test_corrected_code_model():
    """
    Test different df_prx with corrections present.
    They should not be taken into account in the corrected code observation model.
    """
    from src.gnss import obs_model_corrected_code
    # test geometric distance
    df = pd.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "sat_clock_offset_m": [0] * 3,
            "sagnac_effect_m": [0] * 3,
            "relativistic_clock_effect_m": [0] * 3,
            "sat_code_bias_m": [0] * 3,
            "iono_delay_m": [0] * 3,
            "tropo_delay_m": [0] * 3,
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    assert (obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([1.0] * 3)).all(), (
        "geometric distance wrongly computed"
    )

    # test rx clock bias
    rx_clk = 1
    assert (obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([2.0] * 3)).all(), (
        "issue with rx_clk"
    )
    assert (obs_model_corrected_code(df, rx_pos) == np.array([1.0] * 3)).all(), (
        "rx_clk default value should be 0"
    )

    # test corrections with plus sign
    df = pd.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "sat_clock_offset_m": [0] * 3,
            "sagnac_effect_m": [1] * 3,
            "relativistic_clock_effect_m": [0] * 3,
            "sat_code_bias_m": [1] * 3,
            "iono_delay_m": [1] * 3,
            "tropo_delay_m": [1] * 3,
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    assert (obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([1.0] * 3)).all()

    # test corrections with minus sign
    df = pd.DataFrame(
        {
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
            "sat_clock_offset_m": [1] * 3,
            "sagnac_effect_m": [0] * 3,
            "relativistic_clock_effect_m": [1] * 3,
            "sat_code_bias_m": [0] * 3,
            "iono_delay_m": [0] * 3,
            "tropo_delay_m": [0] * 3,
        }
    )
    rx_pos = [0, 0, 0]
    rx_clk = 0
    assert (obs_model_corrected_code(df, rx_pos, rx_clk) == np.array([1.0] * 3)).all()


def test_corrected_code_residual():
    from src.gnss import residuals_corrected_code

    # test with without noise and corrections
    df = pd.DataFrame(
        {
            "time_of_reception_in_receiver_time": [0, 0, 0],
            "C_obs_corr_m": [1, 1, 1],
            "sat_pos_x_m": [1, 0, 0],
            "sat_pos_y_m": [0, 1, 0],
            "sat_pos_z_m": [0, 0, 1],
        }
    )
    results = pd.DataFrame(
        np.array([0] * 5).reshape(1, 5),
        columns=["epoch", "pos_x", "pos_y", "pos_z", "clk_b"],
    )
    residuals_corrected_code(df, results)
    assert "residual_code" in df.columns
    assert df.residual_code.to_numpy() == pytest.approx(np.zeros(3))
