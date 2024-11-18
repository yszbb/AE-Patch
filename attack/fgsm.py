import torch
import torch.nn as nn
from torchattacks.attack import Attack


class FGSM(Attack):
    r"""
    altered from torchattack
    """

    def __init__(self, model, forward_function=None, eps=0.007, v_reset=None, surrogate_fun=None, alph=None, **kwargs):
        super().__init__("FGSM", model)
        self._targeted = None
        self.eps = eps
        self._supported_mode = ['default', 'targeted1']
        self.forward_function = forward_function
        self.v_reset = v_reset
        self.surrogate_fun = surrogate_fun
        self.alph = alph

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        # images.unsqueeze_(1)
        # images = images.repeat(8, 1, 1, 1, 1)
        images = images.transpose(0, 1)
        images = images.clone().detach().to(self.device)

        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        fool_label = labels[0]

        pred_label = self.model(images).mean(0)
        _, pred_label = pred_label.cpu().max(1)
        adv_images.requires_grad = True

        if labels[0] == pred_label[0]:
            outputs = self.model(adv_images).mean(0)
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = images + self.alph * grad.sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            _, fool_label = self.model(adv_images).mean(0).cpu().max(1)
            fool_label = fool_label[0]

            return_dict = {"success": 1 if not (labels[0] == fool_label) else 0,
                           "X_adv": adv_images.transpose(0, 1),  # "L0": L0,
                           "n_queries": 1,
                           "predicted": labels[0],
                           "predicted_attacked": fool_label}

            if return_dict["success"]:
                print("--------++++Success attack +++++-------")
            else:
                print("No success loops")

            return return_dict

        else:
            return None

