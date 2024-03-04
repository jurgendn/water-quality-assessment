from config import ModelName

from models.modules.resnet_attention import ResNetBackboneWithAttention
from models.modules.resnet_no_attention import ResNetBackboneWithoutAttention

MODEL_DICT = {
    ModelName.RESNET_ATTENTION: ResNetBackboneWithAttention,
    ModelName.RESNET_NO_ATTENTION: ResNetBackboneWithoutAttention,
}
