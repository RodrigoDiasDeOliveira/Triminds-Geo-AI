import timm


def build_vit(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
):
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    ) 