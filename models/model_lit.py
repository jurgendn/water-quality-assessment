from typing import Dict
import torch
from torch import Tensor, optim
from torch.nn import functional as F

from config import TypeHint
from models.modules.resnet_no_attention import ResNetBackboneWithoutAttention

from .base_model.regression import LightningRegression
from .metrics.regression import regression_metrics
from models import MODEL_DICT


class LitModule(LightningRegression):
    def __init__(
        self,
        lr: float,
        backbone_name: TypeHint.BACKBONE_NAME,
        model_name: str,
        num_classes: int,
        return_layers: Dict[str, str],
    ) -> None:
        super(LitModule, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        model_blueprint = MODEL_DICT.get(model_name)
        self.model = model_blueprint(
            backbone_name=backbone_name,
            num_classes=num_classes,
            return_layers=return_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.model(x)
        return y

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input=input, target=target)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer

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
