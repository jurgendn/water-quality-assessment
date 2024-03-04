from typing import Dict

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from config import TypeHint, MetaData


class ResNetBackboneWithAttention(nn.Module):
    def __init__(
        self,
        backbone_name: TypeHint.BACKBONE_NAME,
        num_classes: int,
        return_layers: Dict[str, str],
    ) -> None:
        super(ResNetBackboneWithAttention, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=(1, 1), padding=1
        )
        model_constructor = MetaData.MODEL_LIST.get(backbone_name, None)
        if model_constructor is None:
            raise Exception("No model constructor")
        self.resnet_backbone = IntermediateLayerGetter(
            model=model_constructor(),
            return_layers=return_layers,
        )
        self.num_features = [
            module
            for module in self.resnet_backbone.modules()
            if not isinstance(module, nn.Sequential)
        ][-1].num_features
        self.attention = nn.MultiheadAttention(
            embed_dim=self.num_features, num_heads=2, dropout=0.05
        )
        self.fc = nn.Linear(in_features=self.num_features, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.resnet_backbone(x)["output"]
        x = F.adaptive_avg_pool2d(input=x, output_size=(1, 1))
        x = x.view(-1, self.num_features)
        x, _ = self.attention(query=x, key=x, value=x)
        x = self.fc(input=x)
        y = F.leaky_relu(input=x)
        return y
