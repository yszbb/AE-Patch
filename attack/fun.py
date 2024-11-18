import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def draw_loss(epochs, loss):
    # Generate the success line
    plt.figure(num=0)
    plt.plot(epochs, loss, label='Loss-fun-curve', linewidth=2, color='r', marker='o', markersize=10,
             markerfacecolor='red', markeredgecolor='red')
    plt.xlabel("epoch")
    plt.ylabel("Loss-average-value")
    plt.xlim(-1, max(epochs) + 1)
    # plt.ylim(0, 1.0)
    plt.title("patch attack loss value")
    plt.legend()
    plt.savefig("D:\\experiment\\Snn_LearnedShapePatch_Attack-master\\attack_log\\loss.pdf")
    plt.close(0)


def save_grey_img(url, img):
    x_y_option = img.cpu().detach().numpy()
    x_y_option = (x_y_option * 255).astype(np.uint8)
    x_y_option = Image.fromarray(x_y_option)
    x_y_option.save(url, quality=99)


class thredOne(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        alpha_1 = 0.2
        alpha_2 = 0.8
        # one = torch.ones_like(input)
        # 可能更合适一点
        ctx.save_for_backward(input)
        input[input < alpha_1] = 0.
        input[input > alpha_2] = 1.
        # input = torch.where((input < alpha_1 or input > alpha_2), one, input)   # 小于0.2的地方替换成1

        return input

    @staticmethod
    def backward(ctx, grad_output):
        alpha_1 = 0.2
        alpha_2 = 0.8
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < alpha_1] = 0
        grad_input[input > alpha_2] = 0

        return grad_input
