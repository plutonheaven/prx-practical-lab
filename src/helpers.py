import pandas as pd
import numpy as np
import pyproj
import json


# def check_answer(answer, correct_value):
#     print("Sucess!!") if (answer == correct_value).all() else print("Try again...")


# def prx_csv_to_pandas(filepath: str):
#     """
#     Read a PRX_CSV file and convert it to pandas DataFrame
#     """
#     data_prx = pd.read_csv(
#         filepath,
#         comment="#",
#         parse_dates=["time_of_reception_in_receiver_time"],
#     )
#     return data_prx


def parse_prx_csv_file_metadata(prx_file):
    with open(prx_file, "r") as f:
        metadata = json.loads(f.readline().replace("# ", ""))
    return metadata


def compute_jacobian_line(
    sat_position_ecef_m : pd.Series,
    rx_pos: np.array,
):
    vec_sat2rx = rx_pos - sat_position_ecef_m [["sat_pos_x_m","sat_pos_y_m","sat_pos_z_m"]].values
    H_pos = vec_sat2rx / np.linalg.norm(vec_sat2rx)
    H_clk = 1.
    return pd.Series({"jacobian_pos_x": H_pos[0], "jacobian_pos_y": H_pos[1], "jacobian_pos_z": H_pos[2], "jacobian_clk": H_clk})


def ecef_to_geodetic(x, y, z):
    transformer = pyproj.Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    longitude, latitude, altitude = transformer.transform(x, y, z)
    return latitude, longitude, altitude


def df_add_geodetic_coord(df: pd.DataFrame):
    geodetic = ecef_to_geodetic(
        df["estimated_position_ecef_x_m"],
        df["estimated_position_ecef_y_m"],
        df["estimated_position_ecef_z_m"],
    )
    coords = pd.Series([geodetic[0], geodetic[1], geodetic[2]])
    return coords


# from https://stackoverflow.com/a/65048500
import scipy.spatial.transform
def geodetic_to_enu(lat, lon, alt, lat_origin, lon_origin, alt_origin):
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    )
    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    x_org, y_org, z_org = transformer.transform(
        lon_origin, lat_origin, alt_origin, radians=False
    )
    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T

    rot1 = scipy.spatial.transform.Rotation.from_euler(
        "x", -(90 - lat_origin), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1
    rot3 = scipy.spatial.transform.Rotation.from_euler(
        "z", -(90 + lon_origin), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1

    rotMatrix = rot1.dot(rot3)

    enu = rotMatrix.dot(vec).T.ravel()
    return enu.T


def df_add_enu_coord(df: pd.DataFrame, lat_origin, lon_origin, alt_origin):
    enu = geodetic_to_enu(
        df["estimated_lat_deg"],
        df["estimated_lon_deg"],
        df["estimated_alt_m"],
        lat_origin,
        lon_origin,
        alt_origin,
    )
    coords = pd.Series([enu[0], enu[1], enu[2]])
    return coords


def enu_to_geodetic(x, y, z, lat_origin, lon_origin, alt_origin):
    transformer1 = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    )
    transformer2 = pyproj.Transformer.from_crs(
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
    )

    x_org, y_org, z_org = transformer1.transform(
        lon_origin, lat_origin, alt_origin, radians=False
    )
    ecef_org = np.array([[x_org, y_org, z_org]]).T

    rot1 = scipy.spatial.transform.Rotation.from_euler(
        "x", -(90 - lat_origin), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1
    rot3 = scipy.spatial.transform.Rotation.from_euler(
        "z", -(90 + lon_origin), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1

    rotMatrix = rot1.dot(rot3)

    ecefDelta = rotMatrix.T.dot(np.array([[x, y, z]]).T)
    ecef = ecefDelta + ecef_org
    lon, lat, alt = transformer2.transform(
        ecef[0, 0], ecef[1, 0], ecef[2, 0], radians=False
    )

    return [lat, lon, alt]


def ecef_to_enu_matrix(lat, lon, alt):
    # ESA Book - eq (B.11)
    R = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0],
            [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)],
            [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)],
        ]
    )
    return R
