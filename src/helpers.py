import pandas as pd
import numpy as np

# import pyproj
import json

EARTH_FLATTENINGFACTOR = 1 / 298.257223563
EARTH_SEMIMAJORAXIS_M = 6378137.0
EARTH_ECCENTRICITY = np.sqrt(2 * EARTH_FLATTENINGFACTOR - EARTH_FLATTENINGFACTOR**2)

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


def parse_prx_metadata(prx_file):
    with open(prx_file, "r") as f:
        metadata = json.loads(f.readline().replace("# ", ""))
    return metadata


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


# def df_add_geodetic_coord(df: pd.DataFrame):
#     geodetic = ecef_to_geodetic(
#         df["estimated_position_ecef_x_m"],
#         df["estimated_position_ecef_y_m"],
#         df["estimated_position_ecef_z_m"],
#     )
#     coords = pd.Series([geodetic[0], geodetic[1], geodetic[2]])
#     return coords


# # from https://stackoverflow.com/a/65048500
# import scipy.spatial.transform
# def geodetic_to_enu(lat, lon, alt, lat_origin, lon_origin, alt_origin):
#     transformer = pyproj.Transformer.from_crs(
#         {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
#         {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
#     )
#     x, y, z = transformer.transform(lon, lat, alt, radians=False)
#     x_org, y_org, z_org = transformer.transform(
#         lon_origin, lat_origin, alt_origin, radians=False
#     )
#     vec = np.array([[x - x_org, y - y_org, z - z_org]]).T

#     rot1 = scipy.spatial.transform.Rotation.from_euler(
#         "x", -(90 - lat_origin), degrees=True
#     ).as_matrix()  # angle*-1 : left handed *-1
#     rot3 = scipy.spatial.transform.Rotation.from_euler(
#         "z", -(90 + lon_origin), degrees=True
#     ).as_matrix()  # angle*-1 : left handed *-1

#     rotMatrix = rot1.dot(rot3)

#     enu = rotMatrix.dot(vec).T.ravel()
#     return enu.T


# def df_add_enu_coord(df: pd.DataFrame, lat_origin, lon_origin, alt_origin):
#     enu = geodetic_to_enu(
#         df["estimated_lat_deg"],
#         df["estimated_lon_deg"],
#         df["estimated_alt_m"],
#         lat_origin,
#         lon_origin,
#         alt_origin,
#     )
#     coords = pd.Series([enu[0], enu[1], enu[2]])
#     return coords


# def enu_to_geodetic(x, y, z, lat_origin, lon_origin, alt_origin):
#     transformer1 = pyproj.Transformer.from_crs(
#         {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
#         {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
#     )
#     transformer2 = pyproj.Transformer.from_crs(
#         {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
#         {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
#     )

#     x_org, y_org, z_org = transformer1.transform(
#         lon_origin, lat_origin, alt_origin, radians=False
#     )
#     ecef_org = np.array([[x_org, y_org, z_org]]).T

#     rot1 = scipy.spatial.transform.Rotation.from_euler(
#         "x", -(90 - lat_origin), degrees=True
#     ).as_matrix()  # angle*-1 : left handed *-1
#     rot3 = scipy.spatial.transform.Rotation.from_euler(
#         "z", -(90 + lon_origin), degrees=True
#     ).as_matrix()  # angle*-1 : left handed *-1

#     rotMatrix = rot1.dot(rot3)

#     ecefDelta = rotMatrix.T.dot(np.array([[x, y, z]]).T)
#     ecef = ecefDelta + ecef_org
#     lon, lat, alt = transformer2.transform(
#         ecef[0, 0], ecef[1, 0], ecef[2, 0], radians=False
#     )

#     return [lat, lon, alt]


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
