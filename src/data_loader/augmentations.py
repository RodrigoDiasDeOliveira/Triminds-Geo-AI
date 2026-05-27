import albumentations as A
from albumentations.pytorch import ToTensorV2


def satellite_augmentations(image_size=224):

    return A.Compose([

        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.RandomBrightnessContrast(p=0.3),

        A.GaussianBlur(p=0.2),

        A.Normalize(),

        ToTensorV2()
    ])