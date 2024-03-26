from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from config.factory import ModelConfig
from constants import MetaData
from models.commons import SelfAttention, IdentityAttention


class BlahBlahModel(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:
        super(BlahBlahModel, self).__init__()
        self.model_config = model_config
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=(1, 1), padding=1
        )
        model_constructor = MetaData.BACKBONE.get(model_config.backbone_name, None)
        if model_constructor is None:
            raise Exception("No model constructor")
        self.resnet_backbone = IntermediateLayerGetter(
            model=model_constructor(),
            return_layers=model_config.return_layers,
        )
        if model_config.attention_type not in MetaData.ATTENTION_TYPE:
            raise Exception("Invalid attention type")
        self.num_channels = self.__get_num_channels()
        if model_config.attention_type is None:
            self.attention = IdentityAttention()
        elif model_config.attention_type == "multihead-attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.num_channels, num_heads=2, dropout=0.05
            )
        elif model_config.attention_type == "sagan":
            self.attention = SelfAttention(
                in_dim=self.num_channels, activation=nn.LeakyReLU
            )
        self.fc = nn.Linear(
            in_features=self.num_channels, out_features=model_config.num_classes
        )

    def __get_num_channels(self) -> int:
        x = torch.randn(size=(1, 1, 64, 64))
        x = self.conv_1(x)
        x = self.resnet_backbone(x)["output"]
        return x.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.resnet_backbone(x)["output"]
        x = F.adaptive_avg_pool2d(input=x, output_size=(1, 1))
        if self.model_config.attention_type in [None, "sagan"]:
            x, _ = self.attention(query=x, key=x, value=x)
            x = x.view(-1, self.num_channels)
        elif self.model_config.attention_type in ["multihead-attention"]:
            x = x.view(-1, self.num_channels)
            x, _ = self.attention(query=x, key=x, value=x)
        x = self.fc(input=x)
        y = F.leaky_relu(input=x)
        return y
