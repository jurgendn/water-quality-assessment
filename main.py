import pickle
from argparse import ArgumentParser
import random
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from components.callbacks import early_stopping
from components.data_module import WaterQualityDataModule
from config.factory import DataLoader, ModelConfig, OptimizerConfig, TrainerConfig
from models.model_lit import LitModule


def train():
    parsers = ArgumentParser()
    parsers.add_argument("--input-data", dest="input_data")
    parsers.add_argument("--target-data", dest="target_data")
    parsers.add_argument("--water-depth", dest="water_depth", default=None)
    parsers.add_argument("--config-path", dest="config_path", default=None)
    parsers.add_argument("--epochs", dest="epochs", default=500)
    args = parsers.parse_args()

    with open(file=args.config_path, mode="r") as f:
        config = yaml.full_load(stream=f)

    data_config = DataLoader(
        input_data=args.input_data,
        target_data=args.target_data,
        water_depth=args.water_depth,
        target_features=config["data_config"]["target_features"],
        train_val_ratio=config["data_config"]["train_val_ratio"],
        batch_size=config["data_config"]["batch_size"],
    )
    model_config = ModelConfig(
        model_name=config["model_config"]["model_name"],
        backbone_name=config["model_config"]["backbone_name"],
        attention_type=config["model_config"]["attention_type"],
        return_layers=config["model_config"]["return_layers"],
        num_classes=data_config.num_classes,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=config["optimizer_config"]["lr"],
        lr_scheduler_step_size=config["optimizer_config"]["lr_scheduler_step_size"],
        lr_scheduler_factor=config["optimizer_config"]["lr_scheduler_factor"],
        lr_scheduler_monitor=config["optimizer_config"]["lr_scheduler_monitor"],
    )
    trainer_config = TrainerConfig(
        accelerator=config["trainer_config"]["accelerator"],
        log_every_n_steps=config["trainer_config"]["log_every_n_steps"],
        max_epochs=args.epochs,
    )

    with open(file=data_config.input_data, mode="rb") as f:
        data = pickle.load(f)
    with open(file=data_config.target_data, mode="rb") as f:
        target = pickle.load(f)
    with open(file=data_config.water_depth, mode="rb") as f:
        depth_data = pickle.load(f)

    datamodule = WaterQualityDataModule(
        x=data,
        y=target,
        target_feature_names=data_config.target_features,
        target_feature_indices=data_config.target_features_index,
        val_ratio=data_config.train_val_ratio,
        batch_size=data_config.batch_size,
    )

    logger = WandbLogger(project="Water Quality Assessment")
    model = LitModule(model_config=model_config, optimizer=optimizer_config)

    trainer = Trainer(
        accelerator=trainer_config.accelerator,
        log_every_n_steps=trainer_config.log_every_n_steps,
        logger=logger,
        max_epochs=trainer_config.max_epochs,
        callbacks=[early_stopping],
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    random.seed(a=42)
    train()
