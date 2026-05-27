from src.data_loader.geospatial_utils import GeoSpatialUtils


def test_coordinate_normalization():

    geo = GeoSpatialUtils()

    norm = geo.normalize_coordinates((45, 90))

    assert 0 <= norm[0] <= 1
    assert 0 <= norm[1] <= 1