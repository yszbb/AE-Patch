import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from .snn_neu import LIFSpike
from functions import seed_all
cfg = {
    'vgg5': [[64, 'A'],
             [128, 128, 'A'],
             [],
             [],
             []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vggdvs': [
        [16, 32, 'A'],
        [32, 32, 'A'],
        [64, 64, 'A'],
        [64, 64, 'A'],
        []
    ],
    'vgggesture': [
        [16, 32, 'A'],
        [32, 32, 'A'],
        [],
        [],
        []
    ]
}


# 'vggdvs': [
#         [16, 32, 'A'],
#         [32, 32, 'A'],
#         [64, 64, 'A'],
#         [128, 64, 'A'],
#         []
#     ],

class VGGDvs(nn.Module):
    def __init__(self, vgg_name, tau=0.9, alph=1., v_reset=0., mode='bptt', num_class=10, init_c=2,
                 init_s=34, v_threshold: float = 1.):
        super(VGGDvs, self).__init__()
        self.tau = tau
        self.alph = alph
        self.v_reset = v_reset
        self.mode = mode
        self.v_threshold = v_threshold
        self.init_channels = init_c

        cnt = 0
        for l in cfg[vgg_name]:
            if len(l) > 0:
                cnt += 1

        self.W = int(init_s / (1 << cnt)) ** 2

        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.classifier = self._make_classifier(num_class, cfg[vgg_name][cnt - 1][1])

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(layer.AvgPool2d(2))
            else:
                layers.append(layer.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(layer.BatchNorm2d(x))
                layers.append(LIFSpike(tau=self.tau, v_reset=self.v_reset, alph=self.alph, v_threshold=self.v_threshold,
                                       mode=self.mode))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class, channels):
        layers = [layer.Flatten(), layer.Linear(channels * self.W, num_class)]
        return nn.Sequential(*layers)

    def set_bp_single(self, mode='bptt'):
        for module in self.modules():
            if isinstance(module, LIFSpike):
                module.mode = mode

    def set_bp_mixed(self, mode="bptr_old"):
        self.set_bp_single(mode='bptt')
        seed_all(42)
        for module in self.modules():
            index = torch.randn(1)
            if isinstance(module, LIFSpike) and index.item() >= 0.:
                module.mode = mode

    def set_bp_same(self, mode='bptr_old'):
        for module in self.modules():
            if isinstance(module, LIFSpike):
                module.mode = mode

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classifier(out)
        # out 的 shape T*batchsize*num_class
        return out
