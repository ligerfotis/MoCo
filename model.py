
######################
### from MoCo repo ###
######################
from functools import partial

import torch
from torchvision.models import resnet

from dataset_utils import SplitBatchNorm


class ModelBase(torch.nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, feature_dim=128, arch="resnet18", bn_splits=8):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = (partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else torch.nn.BatchNorm2d)
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == "conv1":
                module = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if isinstance(module, torch.nn.MaxPool2d):
                continue
            if isinstance(module, torch.nn.Linear):
                self.net.append(torch.nn.Flatten(1))
            self.net.append(module)

        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x


######################
### from MoCo repo ###
######################
def copy_params(encQ, encK, m=None):
    # direct copy of weights
    if m is None:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    # copy a moving average of weights
    else:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)


def create_encoder(emb_dim, device):
    model = ModelBase(emb_dim)
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model