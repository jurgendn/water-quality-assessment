from dataclasses import dataclass

from torchvision.models import mobilenet_v3_large, resnet18, resnet34, resnet50


@dataclass
class MetaData:
    FULL_FEATURES_LIST = [
        "W.T.",
        "pH",
        "DO",
        "EC",
        "BOD5",
        "CODMn",
        "SS",
        "TN",
        "TP",
        "TOC",
        "DOC",
        "Chl-a",
        "TN,",
        "NH3-N",
        "NO3-N",
        "DTP",
        "PO4-P",
    ]
    AVAILABLE_BACKBONE = ["r34"]
    ATTENTION_TYPE = ["sagan", "multihead-attention"]
    BACKBONE = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "mobilenet_v3_large": mobilenet_v3_large,
    }
