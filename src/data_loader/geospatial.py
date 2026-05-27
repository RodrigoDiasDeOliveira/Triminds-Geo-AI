import numpy as np


class GeoSpatial:

    def __init__(self, crs="EPSG:4326"):
        self.crs = crs

    def normalize_coordinates(self, coords):
        """
        Normaliza coordenadas (lat, lon) para range 0-1
        """
        lat, lon = coords

        norm_lat = (lat + 90) / 180
        norm_lon = (lon + 180) / 360

        return np.array([norm_lat, norm_lon])

    def calculate_patch_center(self, bbox):
        """
        Calcula centro de uma bounding box geoespacial
        bbox = (min_lat, min_lon, max_lat, max_lon)
        """
        min_lat, min_lon, max_lat, max_lon = bbox

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        return (center_lat, center_lon)

    def compute_distance(self, point1, point2):
        """
        Distância simples euclidiana (baseline)
        """
        return np.linalg.norm(
            np.array(point1) - np.array(point2)
        )