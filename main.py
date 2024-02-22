import os
import pickle

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

from components.callbacks import early_stopping
from components.data_module import WaterQualityDataModule
from config import DataLoader, DataPath, ModelConfigs, TrainerConfigs
from models.model_lit import LitModule


def train():
    with open(file=DataPath.INPUT_DATA, mode="rb") as f:
        data = pickle.load(f)
    with open(file=DataPath.TARGET_DATA, mode="rb") as f:
        target = pickle.load(f)
    with open(file=DataPath.WATER_DEPTH, mode="rb") as f:
        depth_data = pickle.load(f)
    datamodule = WaterQualityDataModule(
        x=data,
        y=target,
        target_feature_names=DataLoader.TARGET_FEATURES,
        target_feature_indices=DataLoader.TARGET_FEATURES_INDEX,
        val_ratio=DataLoader.TRAIN_VAL_RATIO,
        batch_size=DataLoader.BATCH_SIZE,
    )
    logger = NeptuneLogger(
        project=os.environ["PROJECT_NAME"],
        api_token=os.environ["API_KEY"],
    )
    model = LitModule(**ModelConfigs.R34["kwargs"])
    trainer = Trainer(
        accelerator=TrainerConfigs.ACCELERATOR,
        log_every_n_steps=TrainerConfigs.LOG_EVERY_N_STEPS,
        logger=logger,
        max_epochs=TrainerConfigs.MAX_EPOCHS,
        callbacks=[early_stopping],
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
