import torch
import torch.nn as nn
from torchattacks.attack import Attack


class PGD(Attack):
    r"""
    altered from torchattack
    """

    def __init__(self, model, forward_function=None, eps=0.3, alpha=2 / 255, steps=40, alph=None, v_reset=None,
                 random_start=True, **kwargs):
        super().__init__("PGD", model)
        self._targeted = None  # 是否为定向攻击
        self.eps = eps  # 初始化图像需要使用 以及 扰动大小 2.55
        self.alpha = alpha  # 迭代的步幅 2
        self.steps = steps  # 迭代次数
        self.random_start = random_start  # 迭代是从随机点开始 还是 从原始样本点开始
        self._supported_mode = ['default', 'targeted1']
        self.forward_function = forward_function
        self.v_reset = v_reset

        self.alph = alph
        self.max_hamming_distance = 2000

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        n_queries = 0
        loops = 0
        images = images.transpose(0, 1)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        fool_label = labels[0]
        # 求出真实标签
        n_queries += 1
        pred_label = self.model(images).mean(0)
        _, pred_label = pred_label.cpu().max(1)

        # if self.random_start:
        #     # Starting at a uniformly random point
        #     adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        if labels[0] == pred_label[0]:
            while labels[0] == fool_label and loops < self.steps:
                adv_images.requires_grad = True
                n_queries += 1
                outputs = self.model(adv_images).mean(0)
                cost = loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                # adv_images = adv_images.detach() + grad.sign()
                # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

                n_queries += 1
                _, fool_label = self.model(adv_images).mean(0).cpu().max(1)

                fool_label = fool_label[0]
                loops += 1

            # 修改的数量  扰动大小
            # L0 = int(torch.sum(torch.abs(adv_images - images)))

            return_dict = {"success": 1 if not (labels[0] == fool_label) else 0,
                           "X_adv": adv_images.transpose(0, 1),  # "L0": L0,
                           "n_queries": n_queries,
                           "loops": loops,
                           "predicted": labels[0],
                           "predicted_attacked": fool_label}

            if return_dict["success"]:
                print("--------++++Success attack loops+++++-------", loops)
            else:
                print("No success loops", loops)

            return return_dict
        else:
            return None
