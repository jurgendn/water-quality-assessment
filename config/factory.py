import os
from typing import Dict, List

from pydantic import BaseModel, Field

from constants import MetaData


class DataLoader(BaseModel):
    input_data: str
    target_data: str
    water_depth: str
    target_features: List[str]
    train_val_ratio: float
    batch_size: int

    @property
    def target_features_index(self):
        return [
            MetaData.FULL_FEATURES_LIST.index(feature)
            for feature in self.target_features
        ]

    @property
    def num_classes(self):
        return len(self.target_features)


class ModelConfig(BaseModel):
    model_name: str
    backbone_name: str
    attention_type: str
    num_classes: int
    return_layers: Dict[str, str]


class OptimizerConfig(BaseModel):
    optimizer: str
    lr: float
    lr_scheduler_factor: float
    lr_scheduler_step_size: int
    lr_scheduler_monitor: str


class TrainerConfig(BaseModel):
    accelerator: str
    log_every_n_steps: int = Field(default=1)
    max_epochs: int = Field(default=100)
