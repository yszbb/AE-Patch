#  自定义求导方式  替代导数
import torch
from torch import nn


def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
    v = v - spike * v_threshold
    return v


def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset):
    v = (1. - spike) * v + spike * v_reset
    return v


def neuronal_reset(v: torch.Tensor, spike, v_reset: float = 0., v_threshold: float = 1.):
    if v_reset == 0.:
        v = jit_hard_reset(v, spike, v_reset)
    else:
        v = jit_soft_reset(v, spike, v_threshold)
    return v


class Surrogate_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alph):
        alph = torch.tensor([alph])
        spike = (x >= 0).float()
        ctx.save_for_backward(x, spike, alph)

        return spike

    @staticmethod
    def backward(ctx, grad_output):
        (x, spike, alph) = ctx.saved_tensors
        alph = alph[0].item()
        grad = (1 / alph) * (1 / alph) * ((alph - x.abs()).clamp(min=0))
        grad_input = grad_output * grad
        return grad_input, None


class NormBp(nn.Module):
    def __init__(self, tau: float = 0.9, alph=0.1, v_reset: float = 0., v_threshold: float = 1.):
        super(NormBp, self).__init__()
        self.tau = tau

        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.alph = alph
        self.Surrogate_fun = Surrogate_fun.apply

    def v_float_to_tensor(self, v: torch.Tensor, x: torch.Tensor):
        if isinstance(v, float):
            v_init = v
            v = torch.full_like(x.data, v_init)
        return v

    def neuronal_charge(self, v: torch.Tensor, x: torch.Tensor):
        v = v * self.tau + x
        return v

    def neuronal_fire(self, v: torch.Tensor):
        return self.Surrogate_fun(v - self.v_threshold, self.alph)

    def single_step_forward(self, v: torch.Tensor, x: torch.Tensor):

        v = self.v_float_to_tensor(v, x)

        v = self.neuronal_charge(v, x)

        spike = self.neuronal_fire(v)

        v = neuronal_reset(v, spike, self.v_reset, self.v_threshold)

        return v, spike

    def forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        v = 0.
        for t in range(T):
            v, y = self.single_step_forward(v=v, x=x_seq[t])

            y_seq.append(y)

        return torch.stack(y_seq)


#   自定义求导方式   LIF神经元的前向传播过程，以及反向传播过程。 以及反向传播时候使用的速率梯度相似的使用  默认是多步模式
class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau, flag, v_reset: float = 0., v_threshold: float = 1.):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * tau + x[t, ...]  # 积分
            spike = ((mem - 1.) > 0).float()  # 点火, 此操作反向传播不可导

            mem = neuronal_reset(mem, spike, v_reset, v_threshold)  # 重置

            spike_pot.append(spike)

        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau), torch.tensor(flag))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out, tau, flag = ctx.saved_tensors
        x = x.mean(0, keepdim=True)  # 取平均值，保持原来的维度, T=1

        gamma = 0.2  # 第一段直线的导数
        ext = 1  # 1 + ext  相当于论文中的 βθ = 2
        # tau 就是论文中的 λ
        des = 1  # 就是一个常数

        # --------------------
        k_1 = 0.4
        u_1 = 2.0

        a = 0.5
        mid = (1. - u_1 * k_1) / a
        mid = torch.tensor(mid)

        b = - torch.log(mid) / u_1

        if flag == 0.:
            grad = (x >= 1 - tau).float() * (x <= 1 + ext).float() * (des - gamma + gamma * tau) / (tau + ext) + (
                    x <= 1 - tau).float() * (x >= 0).float() * gamma
        else:
            grad = (x <= u_1).float() * (x >= 0).float() * k_1 + (x >= u_1).float() * (a * b * torch.exp(-b * x))

        grad_input = grad_output * grad

        return grad_input, None, None, None, None


class LIFSpike(nn.Module):
    def __init__(self, tau=0.9, v_reset: float = 0., alph=0.1, v_threshold: float = 1., mode='bptt'):
        super(LIFSpike, self).__init__()

        self.tau = tau  # 泄露因子 膜电位常数
        self.v_reset = v_reset  # 进行硬重置还是软重置
        self.alph = alph
        self.v_threshold = v_threshold

        self.normbp = NormBp(tau, alph, v_reset, v_threshold)

        self.ratebp = RateBp.apply
        #  bptt反向传播通过时间  bptr反向传播通过速率
        self.mode = mode

    def forward(self, x):

        if self.mode == 'bptr_old':
            x = self.ratebp(x, self.tau, 0., self.v_reset, self.v_threshold)
        elif self.mode == 'bptr_new':
            x = self.ratebp(x, self.tau, 1., self.v_reset, self.v_threshold)
        else:
            x = self.normbp(x)

        return x
