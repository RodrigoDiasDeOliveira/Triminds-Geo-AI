import numpy as np
import rasterio
from pyproj import Geod


class GeoSpatial:
    """
    Utilitários geoespaciais para imagens de satélite.
    """

    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs
        self.geod = Geod(ellps="WGS84")  # Para cálculos mais precisos

    def normalize_coordinates(self, lat: float, lon: float) -> np.ndarray:
        """Normaliza coordenadas geográficas para range [0, 1]"""
        norm_lat = (lat + 90) / 180
        norm_lon = (lon + 180) / 360
        return np.array([norm_lat, norm_lon])

    def calculate_patch_center(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Calcula o centro de uma bounding box (min_lat, min_lon, max_lat, max_lon)"""
        min_lat, min_lon, max_lat, max_lon = bbox
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        return (center_lat, center_lon)

    def compute_distance(self, point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """Calcula distância geodésica precisa entre dois pontos (lat, lon) em metros"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        _, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)
        return distance

    def get_raster_metadata(self, image_path: str) -> dict:
        """Extrai metadados geoespaciais de uma imagem raster (GeoTIFF)"""
        try:
            with rasterio.open(image_path) as src:
                return {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                    "count": src.count,  # número de bandas
                    "driver": src.driver
                }
        except Exception as e:
            raise RuntimeError(f"Failed to read raster metadata from {image_path}: {e}")
        

    def extract_patch_coordinates(self, image_path: str) -> dict | None:
        """Extrai coordenadas do centro e bounds de uma imagem raster"""
        try:
            metadata = self.get_raster_metadata(image_path)
            bounds = metadata["bounds"]
            center = self.calculate_patch_center(
                (bounds.bottom, bounds.left, bounds.top, bounds.right)
            )
            return {
                "center": center,
                "bounds": (bounds.bottom, bounds.left, bounds.top, bounds.right),
                "crs": metadata["crs"]
            }
        except Exception:
            return None


# Função helper
def create_geospatial_handler(crs: str = "EPSG:4326") -> GeoSpatial:
    return GeoSpatial(crs=crs)