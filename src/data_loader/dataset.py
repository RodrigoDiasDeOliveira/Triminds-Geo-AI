from collections.abc import Callable
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def default_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SatelliteDataset(Dataset):
    """Dataset de imagens de satélite (RGB, extensível a multiespectral)."""

    def __init__(
        self,
        image_paths: list[str] | None = None,
        labels: list[int] | None = None,
        transform: Callable | None = None,
        image_size: int = 224,
        data_path: str | None = None,
    ):
        # Suporta construção via listas OU via diretório organizado em subpastas por classe
        if data_path is not None and image_paths is None:
            image_paths, labels = self._scan_directory(data_path)
        self.image_paths = image_paths or []
        self.labels = labels or []
        self.image_size = image_size
        self.transform = transform or default_transforms(image_size)

    @staticmethod
    def _scan_directory(root: str) -> tuple[list[str], list[int]]:
        root_p = Path(root)
        if not root_p.exists():
            return [], []
        classes = sorted([d.name for d in root_p.iterdir() if d.is_dir()])
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        paths, labels = [], []
        for c in classes:
            for f in (root_p / c).glob("*"):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    paths.append(str(f))
                    labels.append(cls_to_idx[c])
        return paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloader(
    image_paths, labels, batch_size=32, shuffle=True, num_workers=4, transform=None
) -> DataLoader:
    dataset = SatelliteDataset(image_paths, labels, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
