import os
from dataclasses import dataclass
from torchvision.models.resnet import resnet34, resnet18, resnet50
from typing import Literal


@dataclass
class TypeHint:
    MODEL_NAME = Literal["r18", "r34", "r50"]


@dataclass
class MetaData:
    MODEL_LIST = {"r18": resnet18, "r34": resnet34, "r50": resnet50}
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


@dataclass
class DataPath:
    BASE_PATH = "./dataset"
    INPUT_DATA = os.path.join(BASE_PATH, "heat_map_EEM.pkl")
    TARGET_DATA = os.path.join(BASE_PATH, "ref_EEM.pkl")
    WATER_DEPTH = os.path.join(BASE_PATH, "water-depth.pkl")

@dataclass
class DataLoader:
    TARGET_FEATURES = [
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
        # "TN,",
        "NH3-N",
        # "NO3-N",
        "DTP",
        "PO4-P",
    ]
    TARGET_FEATURES_INDEX = [
        MetaData.FULL_FEATURES_LIST.index(feature) for feature in TARGET_FEATURES
    ]
    TRAIN_VAL_RATIO = 0.2
    BATCH_SIZE = 32


@dataclass
class ModelConfigs:
    R18 = {
        "model_name": "r18",
        "kwargs": {
            "lr": 1e-3,
            "model_name": "r18",
            "num_classes": len(DataLoader.TARGET_FEATURES),
            "return_layers": {"layer1": "output"},
        },
    }
    R34 = {
        "model_name": "r34",
        "kwargs": {
            "lr": 4e-3,
            "model_name": "r34",
            "num_classes": len(DataLoader.TARGET_FEATURES),
            "return_layers": {"layer2": "output"},
        },
    }


@dataclass
class TrainerConfigs:
    ACCELERATOR = "cpu"
    LOG_EVERY_N_STEPS = 1
    MAX_EPOCHS = 100
