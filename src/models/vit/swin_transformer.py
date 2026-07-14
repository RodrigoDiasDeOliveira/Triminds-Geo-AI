import timm


def build_swin_transformer(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
):
    return timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )