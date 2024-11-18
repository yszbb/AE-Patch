import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from .snn_neu import LIFSpike
from functions import seed_all
cfg = {
    'lenet5_simple': [8, 8, 8, 64],
    'lenet5': [16, 32, 64, 256]
}


# 使用反向传播进行训练的LeNet5
class LeNet5Dvs(nn.Module):
    def __init__(self, lenet_name, tau=0.9, alph=0.1, v_reset=0., mode='bptt', num_class=10, init_c=2,
                 init_s=34, v_threshold: float = 1.):
        super(LeNet5Dvs, self).__init__()
        self.tau = tau
        self.alph = alph
        self.v_reset = v_reset
        self.mode = mode
        self.v_threshold = v_threshold
        self.init_channels = init_c
        self.W = int(((init_s - 2) / 2 - 2) / 4)
        self.layer_info = cfg[lenet_name]
        self.model = nn.Sequential(
            layer.Conv2d(2, self.layer_info[0], 5, 1, padding=1, bias=False),
            layer.BatchNorm2d(self.layer_info[0]),
            LIFSpike(tau=self.tau, v_reset=self.v_reset, alph=self.alph, v_threshold=self.v_threshold, mode=self.mode),
            layer.AvgPool2d(2),

            layer.Conv2d(self.layer_info[0], self.layer_info[1], 5, 1, padding=1, bias=False),
            layer.BatchNorm2d(self.layer_info[1]),
            LIFSpike(tau=self.tau, v_reset=self.v_reset, alph=self.alph, v_threshold=self.v_threshold, mode=self.mode),
            layer.AvgPool2d(2),

            layer.Conv2d(self.layer_info[1], self.layer_info[2], 3, 1, padding=1, bias=False),
            layer.BatchNorm2d(self.layer_info[2]),
            LIFSpike(tau=self.tau, v_reset=self.v_reset, alph=self.alph, v_threshold=self.v_threshold, mode=self.mode),
            layer.AvgPool2d(2),

            layer.Flatten(),
            layer.Linear(self.layer_info[2] * self.W * self.W, self.layer_info[3], bias=False),
            #  layer.BatchNorm1d(self.layer_info[3]),
            LIFSpike(tau=self.tau, v_reset=self.v_reset, alph=self.alph, v_threshold=self.v_threshold, mode=self.mode),
            layer.Linear(self.layer_info[3], out_features=num_class, bias=False)

        )

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
        out = self.model(input)
        # out 的 shape T*batchsize*num_class
        return out
