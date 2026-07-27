import logging

import numpy as np
import rasterio
from pyproj import Geod

logger = logging.getLogger(__name__)


class GeoSpatial:
    """
    Geospatial utilities for satellite imagery.
    """

    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs
        self.geod = Geod(ellps="WGS84")

    def normalize_coordinates(self, lat: float, lon: float) -> np.ndarray:
        """Normalize geographic coordinates to the range [0, 1]."""
        norm_lat = (lat + 90) / 180
        norm_lon = (lon + 180) / 360
        return np.array([norm_lat, norm_lon])

    def calculate_patch_center(
        self,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """
        Calculate the center of a bounding box.

        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon)

        Returns:
            (latitude, longitude)
        """
        min_lat, min_lon, max_lat, max_lon = bbox

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        return center_lat, center_lon

    def compute_distance(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> float:
        """
        Compute geodesic distance between two points in meters.
        """
        lat1, lon1 = point1
        lat2, lon2 = point2

        _, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)

        return distance

    def get_raster_metadata(self, image_path: str) -> dict:
        """
        Extract metadata from a GeoTIFF image.
        """
        try:
            with rasterio.open(image_path) as src:
                return {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                    "count": src.count,
                    "driver": src.driver,
                }

        except rasterio.errors.RasterioIOError as err:
            raise RuntimeError(f"Failed to read raster metadata from '{image_path}'.") from err

    def extract_patch_coordinates(self, image_path: str) -> dict | None:
        """
        Extract center coordinates and bounds from a raster image.
        """
        try:
            metadata = self.get_raster_metadata(image_path)

            bounds = metadata["bounds"]

            center = self.calculate_patch_center(
                (
                    bounds.bottom,
                    bounds.left,
                    bounds.top,
                    bounds.right,
                )
            )

            return {
                "center": center,
                "bounds": (
                    bounds.bottom,
                    bounds.left,
                    bounds.top,
                    bounds.right,
                ),
                "crs": metadata["crs"],
            }

        except RuntimeError as err:
            logger.warning(
                "Unable to extract coordinates from '%s': %s",
                image_path,
                err,
            )
            return None


def create_geospatial_handler(crs: str = "EPSG:4326") -> GeoSpatial:
    """Factory function for GeoSpatial."""
    return GeoSpatial(crs=crs)
