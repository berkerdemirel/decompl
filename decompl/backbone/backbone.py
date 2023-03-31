import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from thop import profile, clever_format
from fvcore.nn import activation_count, flop_count, parameter_count
import numpy as np
from typing import Dict, Tuple, List, Union


class MyVGG16(nn.Module):
    def __init__(self, pretrained: bool=False):
        super(MyVGG16,self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        self.features = vgg.features
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("vgg", params)
        '''
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.features(x)
        return [x]
    

if __name__ == '__main__':
    model = MyVGG16(pretrained=True)
    img = torch.randn((1, 3, 1280, 720))
    out = model(img)
    print(out[0].shape)
