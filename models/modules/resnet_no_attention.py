from typing import Dict

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from config import TypeHint, MetaData


class ResNetBackboneWithoutAttention(nn.Module):
    def __init__(
        self,
        model_name: TypeHint.MODEL_NAME,
        num_classes: int,
        return_layers: Dict[str, str],
    ) -> None:
        super(ResNetBackboneWithoutAttention, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=(1, 1), padding=1
        )
        model_constructor = MetaData.MODEL_LIST.get(model_name, None)
        if model_constructor is None:
            raise "Error"
        self.resnet_backbone = IntermediateLayerGetter(
            model=model_constructor(),
            return_layers=return_layers,
        )
        self.num_features = [
            module
            for module in self.resnet_backbone.modules()
            if not isinstance(module, nn.Sequential)
        ][-1].num_features
        self.fc = nn.Linear(in_features=self.num_features, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.resnet_backbone(x)["output"]
        x = F.adaptive_avg_pool2d(input=x, output_size=(1, 1))
        x = x.view(-1, self.num_features)
        x = self.fc(input=x)
        y = F.leaky_relu(input=x)
        return y
