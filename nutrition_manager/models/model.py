"""Model factory: an ImageNet-pretrained backbone with a 5-way regression head."""
import timm


def build_model(backbone: str, num_outputs: int = 5, dropout: float = 0.2, pretrained: bool = True):
    """Return a timm backbone whose classifier head is a `num_outputs` regressor.

    timm's `num_classes` swaps the classification head for a linear layer of the
    requested width, which we use directly as the regression head.
    """
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_outputs,
        drop_rate=dropout,
    )
