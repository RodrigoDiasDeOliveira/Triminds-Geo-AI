from src.data_loader.dataset import SatelliteDataset


def test_dataset_length():
    dataset = SatelliteDataset(data_path="data/raw")
    assert len(dataset) >= 0