from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.offline as pyo
from plotly.subplots import make_subplots

EARTH_FLATTENINGFACTOR = 1 / 298.257223563
EARTH_SEMIMAJORAXIS_M = 6378137.0
EARTH_ECCENTRICITY = np.sqrt(2 * EARTH_FLATTENINGFACTOR - EARTH_FLATTENINGFACTOR**2)


def ecef_to_geodetic(pos_ecef_x: np.array, pos_ecef_y: np.array, pos_ecef_z: np.array):
    """
    pos_ecef_{x,y,z}: np.array of shape (n,)
    Reference:
    GNSS data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.4),(B.5),(B.6)
    """
    p = np.linalg.norm(np.array([pos_ecef_x, pos_ecef_y]), axis=0)
    longitude_rad = np.arctan2(pos_ecef_y, pos_ecef_x)
    precision_rad = np.full(
        pos_ecef_x.shape, 1.6e-10
    )  # desired precision in radians, corresponds to 1 mm
    delta_phi_rad = np.full(
        pos_ecef_x.shape, 1.0
    )  # initialization to a value larger than precision
    altitude_m = np.zeros(pos_ecef_x.shape)
    latitude_rad = np.arctan2(pos_ecef_z, p * (1 - EARTH_ECCENTRICITY**2))
    while (delta_phi_rad > precision_rad).any():
        n = EARTH_SEMIMAJORAXIS_M / np.sqrt(
            1 - EARTH_ECCENTRICITY**2 * np.sin(latitude_rad) ** 2
        )
        # altitude_previous = altitude_m
        altitude_m = p / np.cos(latitude_rad) - n
        # delta_h_m = np.abs(altitude_m - altitude_previous)
        latitude_prev = latitude_rad
        latitude_rad = np.arctan2(
            pos_ecef_z,
            p * (1 - n * EARTH_ECCENTRICITY**2 / (n + altitude_m)),
        )
        delta_phi_rad = np.abs(latitude_rad - latitude_prev)
    return latitude_rad, longitude_rad, altitude_m


def ecef_to_enu_matrix(lat_rad, lon_rad):
    # ESA Book - eq (B.11)
    R = np.array(
        [
            [-np.sin(lon_rad), np.cos(lon_rad), 0],
            [
                -np.cos(lon_rad) * np.sin(lat_rad),
                -np.sin(lon_rad) * np.sin(lat_rad),
                np.cos(lat_rad),
            ],
            [
                np.cos(lon_rad) * np.cos(lat_rad),
                np.sin(lon_rad) * np.cos(lat_rad),
                np.sin(lat_rad),
            ],
        ]
    )
    return R


def ecef_to_enu(xyz, xyz_ref):
    (lat_rad, lon_rad, _) = ecef_to_geodetic(
        xyz_ref[0],
        xyz_ref[1],
        xyz_ref[2],
    )
    R_ecef_to_enu = ecef_to_enu_matrix(lat_rad, lon_rad)
    return R_ecef_to_enu @ (xyz - xyz_ref)


def compute_enu_pos_error(results_df, ref_pos):
    """
    Compute the position error in the ENU frame, add the ENU position error to the results_df as new columns.

    Inputs:
    - results_df: pd.DataFrame, dataframe containing the columns "epochs", "pos_x", "pos_y", "pos_z"
    - ref_pos: list, true position of the receiver in ECEF frame

    Outputs:
    - results_df: same pd.Dataframe with additional columns "pos_e", "pos_n", "pos_u"
    """
    n_epoch = len(results_df)
    enu_est = np.empty((n_epoch, 3))
    for idx_epoch in range(n_epoch):
        enu_est[idx_epoch, :] = ecef_to_enu(
            results_df.iloc[idx_epoch][["pos_x", "pos_y", "pos_z"]], ref_pos
        )
    results_df = pd.concat(
        [results_df, pd.DataFrame(enu_est, columns=["pos_e", "pos_n", "pos_u"])],
        axis=1,
    )
    return results_df


def plot_enu_error(
    filepath_save: str,
    results_df: pd.DataFrame,
    title: str,
):
    """
    Plot ENU error using Plotly Express.
    One subplot for horizontal error scatter plot, another for vertical error time series.

    Inputs:
    - filepath_save: str, path to save the HTML file
    - results_df: pd.DataFrame with columns "epoch", "pos_e", "pos_n", "pos_u"
    - title: str, plot title
    """
    # Create horizontal error scatter plot
    fig_hor = px.scatter(
        results_df,
        x="pos_e",
        y="pos_n",
        title="Horizontal Position Error",
        labels={"pos_e": "East error [m]", "pos_n": "North error [m]"},
    )
    fig_hor.update_traces(marker=dict(size=4), name=title)

    # Create vertical error time series plot
    fig_vert = px.scatter(
        results_df,
        x="epoch",
        y="pos_u",
        title="Vertical Position Error",
        labels={"epoch": "Epoch", "pos_u": "Up error [m]"},
    )
    fig_vert.update_traces(marker=dict(size=4))

    # Combine subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Horizontal Error", "Vertical Error")
    )
    fig.add_trace(fig_hor.data[0], row=1, col=1)
    fig.add_trace(fig_vert.data[0], row=1, col=2)

    fig.update_xaxes(title_text="East error [m]", row=1, col=1)
    fig.update_yaxes(title_text="North error [m]", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Up error [m]", row=1, col=2)

    fig.update_layout(title_text=title, xaxis=dict(scaleanchor="y", scaleratio=1))
    pyo.plot(fig, auto_open=False, filename=filepath_save + ".html")


def plot_enu_error_cdf(filepath_save: str, results_df: pd.DataFrame, title: str):
    """
    Save a figure of the ENU error cumulative density function. One subplot for the horizontal error, another for the vertical error.

    Inputs:
    - filepath_save: str, location for saving the figure. Ex: "figures/enu.png"
    - results_df: pd.DataFrame, dataframe containing the columns "epochs", "pos_e", "pos_n", "pos_u"
    - title: str, plot title
    """
    # Compute horizontal error
    results_df["hor_error"] = np.sqrt(
        results_df["pos_e"] ** 2 + results_df["pos_n"] ** 2
    )

    # Create ECDF plots
    fig_hor = px.ecdf(
        results_df,
        x="hor_error",
        title="Horizontal Position Error CDF",
        labels={"hor_error": "Horizontal error [m]"},
    )
    fig_vert = px.ecdf(
        results_df,
        x="pos_u",
        title="Vertical Position Error CDF",
        labels={"pos_u": "Up error [m]"},
    )

    # Customize layout
    fig_hor.update_layout(showlegend=True)
    fig_vert.update_layout(showlegend=True)

    # Combine subplots
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(fig_hor.data[0], row=1, col=1)
    fig.add_trace(fig_vert.data[0], row=1, col=2)

    fig.update_xaxes(title_text="Horizontal error [m]", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative distribution", row=1, col=1)
    fig.update_xaxes(title_text="Up error [m]", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative distribution", row=1, col=2)

    fig.update_layout(title_text=title)
    pyo.plot(fig, auto_open=False, filename=filepath_save + ".html")


def plot_residuals_code(filepath_save: str, df: pd.DataFrame, title: str):
    df["prn"] = df["prn"].astype(str).str.zfill(2)
    fig = px.scatter(
        df.sort_values(by="prn"),
        x="time_of_reception_in_receiver_time",
        y="residual_code",
        color="prn",  # Each PRN gets a unique color
        hover_data=["constellation", "prn"],  # Show constellation and PRN on hover
        labels={
            "time_of_reception_in_receiver_time": "Reception Time",
            "residual_code": "Residual Code",
            "prn": "PRN",
        },
        title="Residual Code per PRN",
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(title_text=title, showlegend=True, legend_title="PRN")
    pyo.plot(fig, auto_open=False, filename=filepath_save + ".html")


def analyze_results_feather(file):
    results = pd.read_feather(file)
    analyze_results(results)


def analyze_results(results):
    results["pos_h"] = results[["pos_e", "pos_n"]].apply(np.linalg.norm, axis=1)
    results["pos_3d"] = results[["pos_e", "pos_n", "pos_u"]].apply(
        np.linalg.norm, axis=1
    )
    results_desc = results[["pos_e", "pos_n", "pos_u", "pos_h", "pos_3d"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.95]
    )
    score = results_desc.loc["50%", "pos_3d"] + results_desc.loc["95%", "pos_3d"]
    print(results_desc)
    print("------------------------")
    print(
        f"Score = 0.5 * ({results_desc.loc['50%', 'pos_3d']:.3f} + {results_desc.loc['95%', 'pos_3d']:.3f})"
    )
    print(f"      = {score:.3f}")
    return score


def find_clean_intervals(df, n_intervals: int = 5):
    # compute nb of cycle slips per epochs
    n_cycle_slips = df.groupby("time_of_reception_in_receiver_time")[["LLI"]].apply(
        np.sum
    )
    # compute cumulative number of cycle slips
    n_cycle_slips["cumulative_lli"] = n_cycle_slips["LLI"].cumsum()
    # compute start and end date of periods without LLI
    continuous_periods = (
        n_cycle_slips.reset_index()
        .groupby("cumulative_lli")
        .agg(
            start=("time_of_reception_in_receiver_time", "first"),
            end=("time_of_reception_in_receiver_time", "last"),
        )
    )
    continuous_periods["length_s"] = (
        continuous_periods["end"] - continuous_periods["start"]
    ).dt.total_seconds()
    # return 5 longest periods
    return continuous_periods.sort_values(by="length_s", ascending=False).head(
        n_intervals
    )


def repo_root():
    return Path(__file__).parents[1]


def make_workspace_dirs(workspace_folder: Path):
    for subfolder in ["figures", "results"]:
        workspace_folder.joinpath(subfolder).mkdir(parents=True, exist_ok=True)
