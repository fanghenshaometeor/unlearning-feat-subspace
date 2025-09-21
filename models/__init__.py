from .resnet import *
from .swin import *

model_dict = {
    "resnet50": resnet50,
    "swin_t": swin_t,
}
