from src.data_loader.dataset import SatelliteDataset, default_transforms


def test_dataset_length_empty(tmp_path):
    dataset = SatelliteDataset(data_path=str(tmp_path))
    assert len(dataset) == 0


def test_dataset_from_lists():
    dataset = SatelliteDataset(image_paths=[], labels=[], transform=default_transforms())
    assert len(dataset) == 0
