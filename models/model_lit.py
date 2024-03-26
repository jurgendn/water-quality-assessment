import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler

from config.factory import ModelConfig, OptimizerConfig
from models.model import BlahBlahModel

from .base_model.regression import LightningRegression
from .metrics.regression import regression_metrics


class LitModule(LightningRegression):
    def __init__(self, model_config: ModelConfig, optimizer: OptimizerConfig) -> None:
        super(LitModule, self).__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optimizer = optimizer
        self.model = BlahBlahModel(model_config=model_config)

    def forward(self, x: Tensor) -> Tensor:
        y = self.model(x)
        return y

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input=input, target=target)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.optimizer.lr)
        scheduler = {
            "scheduler": optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self.optimizer.lr_scheduler_step_size,
                gamma=self.optimizer.lr_scheduler_factor,
            ),
            "interval": "epoch",
            "monitor": self.optimizer.lr_scheduler_monitor,
        }
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler: LRScheduler, metric):
        scheduler.step()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(input=logits, target=y)
        metrics = regression_metrics(preds=logits, target=y)

        with torch.no_grad():
            self.train_step_output.append({"loss": loss, **metrics})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.loss(input=logits, target=y)
        metrics = regression_metrics(preds=logits, target=y)
        with torch.no_grad():
            self.validation_step_output.append({"loss": loss, **metrics})
        return loss
