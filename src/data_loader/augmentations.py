
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(image_size: int = 224) -> A.Compose:
    """Augmentations para treinamento (com forte augmentação)"""
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.4),
        
        # Color / Lighting
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        
        # Weather / Noise
        A.RandomFog(p=0.1),
        A.GaussNoise(p=0.2),
        
        # Normalization + Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_augmentations(image_size: int = 224) -> A.Compose:
    """Augmentations leves para validação/teste"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_augmentations(
    mode: str = "train", 
    image_size: int = 224
) -> A.Compose:
    """
    Retorna as augmentations conforme o modo.
    mode: 'train', 'val', 'test'
    """
    if mode == "train":
        return get_train_augmentations(image_size)
    else:
        return get_val_augmentations(image_size)
