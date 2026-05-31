from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config.config import Config  # Vamos importar a config depois


class SatelliteDataset(Dataset):
    """
    Dataset personalizado para imagens de satélite.
    Suporta RGB e futuro suporte multiespectral.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        
        # Usa transform passado ou cria o default
        self.transform = transform or self._get_default_transforms()

    def _get_default_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            raise RuntimeError(f"Error loading image {self.image_paths[idx]}: {e}")


def get_dataloader(
    image_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional = None
):
    """Função helper para criar DataLoader rapidamente."""
    dataset = SatelliteDataset(image_paths, labels, transform=transform)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )