import os
import pickle
import time

import torch.nn as nn
from spikingjelly.datasets import play_frame

from .fun import *
from .patch_utils import *


# 非目标攻击
def patch_attack(image, applied_patch, mask, patch, x, y, t, probability_threshold, model, target_probability, label,
                 target, device, lr=1., max_iteration=15):
    model.eval()
    model.to(device)
    count = 0
    # 被攻击对象对应概率
    image = image.clone().detach().to(device)
    mask = mask.clone().detach().to(device)
    applied_patch = applied_patch.to(device)

    perturbed_img_stro = (torch.mul(mask.type(torch.cuda.FloatTensor), applied_patch.type(torch.cuda.FloatTensor)) +
                          torch.mul((1 - mask.type(torch.cuda.FloatTensor)), image.type(torch.cuda.FloatTensor)))

    applied_patch_stro = applied_patch.data

    total_loss_mean_fin = 0.
    grad_momentum = 0.
    pro_min = target_probability

    # probability_threshold 默认是0.1
    while target_probability > probability_threshold and count < max_iteration:
        count += 1
        applied_patch.requires_grad = True
        # tensor 对应元素相乘
        perturbated_image = torch.mul(mask.type(torch.cuda.FloatTensor),
                                      applied_patch) + torch.mul(
            (1 - mask.type(torch.cuda.FloatTensor)), image.type(torch.cuda.FloatTensor))

        output = model(perturbated_image).mean(0)

        untarget_log_softmax = torch.nn.functional.log_softmax(output, dim=1)

        label_score = untarget_log_softmax[:, target]

        # 模型预测正确的样本的得分平均值
        final_total_score = (label_score * label).mean()
        lambda_loss = 1000

        total_loss = - final_total_score * lambda_loss
        print("总损失total_loss:", total_loss.item())

        total_loss_mean = total_loss.item()

        patch_grad = torch.autograd.grad(total_loss, applied_patch,
                                         retain_graph=False, create_graph=False)[0]

        patch_grad = torch.mul(mask.type(torch.cuda.FloatTensor), patch_grad.type(torch.cuda.FloatTensor))

        grad_new = patch_grad.clone()

        # grad_new = grad_new / torch.mean(torch.abs(grad_new)) + 0.85 * grad_momentum
        # grad_momentum = grad_new
        patch_grad = grad_new

        if patch_grad.norm() == 0:
            print("--------------------Gradient vanish----------------------")
            break

        applied_patch = lr * patch_grad.sign() + applied_patch.detach()
        applied_patch = torch.clamp(applied_patch.data, min=0., max=1.)

        # Test the patch
        perturbed_img = (torch.mul(mask.type(torch.cuda.FloatTensor), applied_patch.type(torch.cuda.FloatTensor)) +
                         torch.mul((1 - mask.type(torch.cuda.FloatTensor)), image.type(torch.cuda.FloatTensor)))

        # 重新测试
        output = model(perturbated_image).mean(0)
        target_probability = torch.nn.functional.softmax(output, dim=1)
        score = target_probability[:, target]
        # 模型预测正确的样本的得分平均值
        target_probability = (score * label).sum() / label.sum()

        if pro_min > target_probability:
            pro_min = target_probability.data
            perturbed_img_stro = perturbed_img.data
            applied_patch_stro = applied_patch.data
            total_loss_mean_fin = total_loss_mean
    # 返回用一个样本更新的patch  以及对抗样本
    return perturbed_img_stro, applied_patch_stro, total_loss_mean_fin


# 损失函数 更新步幅(已经解决) 以及梯度问题 速率梯度重新改写
class Event_patch_attack_target(nn.Module):
    def __init__(self, model, test_fun, patch_type, image_size, noise_percentage, logger, epochs,
                 probability_threshold, max_iteration, lr, target, location_global, device):
        super(Event_patch_attack_target, self).__init__()

        self.test = test_fun  # 测试干净训练集以及测试集的准确率
        self.patch_type = patch_type  # patch 的形状
        self.image_size = image_size  # 输入图像的大小 (T, 2, 34, 34)  输入模型的图像大小(T, 1, 2, 34, 34)
        self.noise_percentage = noise_percentage
        self.logger = logger
        self.epochs = epochs
        self.probability_threshold = probability_threshold
        self.max_iteration = max_iteration
        self.lr = lr
        self.device = device
        self.model = model
        self.target = target
        self.location_global = location_global
        # batchsize = 1

    def forward(self, train_loader, test_loader):

        # Load the model
        model = self.model
        model.to(self.device)
        model.eval()

        # Load the datasets 但是batchsize的大小就是1
        train_loader = train_loader
        test_loader = test_loader

        logger = self.logger

        # Initialize the patch  all_zero
        patch = patch_initialization(self.patch_type, image_size=self.image_size,
                                     noise_percentage=self.noise_percentage)

        logger.info(f'The shape of the patch is{patch.shape}')
        # +++++++++++++++++++++++++++++++
        best_patch, best_patch_success_rate = 0, 0

        # # Test the accuracy of model on trainset and testset    clean example
        # trainset_acc, test_acc = self.test(model, train_loader, self.device, clamp_=True), self.test(model, test_loader,
        #                                                                                             self.device,
        #                                                                                             clamp_=True)
        #
        # logger.info('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(trainset_acc,
        #                                                                                                 test_acc))
        path_best = "D:\\SNN-attack\\ECAI-SNN-attack\\patchs\\untargeted\\" + str(
            self.target) + "\\"
        if not os.path.exists(path_best):
            os.makedirs(path_best)

        best_patch_url = path_best + "best_patch.pkl"
        url_tif = path_best + "best_patch.tif"
        for epoch in range(self.epochs):
            train_total, train_actual_total, train_success, numerate_index = 0, 0, 0, 0
            mean_loss = 0.

            # 不同图像的不同位置
            for (image, label) in train_loader:
                numerate_index += 1
                # if numerate_index > 50:
                #     break

                begin_time = time.time()
                # batchsize*T*2*len*len   模型的输入是 T*batchsize*2*len*len
                train_total += label.shape[0]

                image = image.transpose(0, 1)  # T*batchszie*2*w*h
                # 进行预处理
                image = torch.clamp(image.data, min=0., max=1.)

                image = image.cuda()
                label = label.cuda()
                output = model(image).mean(0)  # batchsize * labels

                # 预测正确的置信度
                target_probability = torch.nn.functional.softmax(output, dim=1)

                score, predicted_old = torch.max(target_probability.data, 1)  # batchsize * labels
                # 把正确的且属于该类的样本挑选出来
                predicted = predicted_old.cuda()
                predicted = predicted.eq(label).float()
                pred = (predicted_old == self.target).float()
                predicted = predicted * pred
                if predicted.sum() == 0:
                    print("没有符合条件的样本,跳过这个batchsize样本")
                    continue

                # 选择模型预测准确的样本进行攻击
                train_actual_total += predicted.sum()

                # 已经符合标准形式  applied_patch  mask shape [T, 1, 2, len, len]
                applied_patch, mask, x_location, y_location, t_location = mask_generation(self.patch_type, patch,
                                                                                          image_size=self.image_size,
                                                                                          location_global=self.location_global,
                                                                                          target=self.target)

                perturbated_image = (
                        torch.mul(mask.type(torch.cuda.FloatTensor), applied_patch.type(torch.cuda.FloatTensor)) +
                        torch.mul((1 - mask.type(torch.cuda.FloatTensor)), image.type(torch.cuda.FloatTensor)))

                output = model(perturbated_image).mean(0)
                # 只求目标的概率
                target_probability_adv = torch.nn.functional.softmax(output, dim=1)
                score = target_probability_adv[:, self.target]

                # 模型预测正确的样本的得分平均值
                mean_score = (score * predicted).sum() / predicted.sum()

                # 返回用一个样本更新的patch  以及对抗样本
                perturbated_image, applied_patch, loss_ = patch_attack(image, applied_patch, mask, patch, x_location,
                                                                       y_location, t_location,
                                                                       self.probability_threshold,
                                                                       model, mean_score, predicted, self.target,
                                                                       self.device,
                                                                       self.lr, self.max_iteration)

                output = model(perturbated_image).mean(0)
                _, predicted_new = torch.max(output.data, 1)
                label_target = torch.full((1, label.shape[0]), self.target)
                label_target = label_target.cuda()

                train_success += ((1. - predicted_new.eq(label_target).float()) * predicted).sum()

                applied_patch = applied_patch.squeeze(1)
                # 抠图抠出来的patch
                patch = applied_patch[t_location:t_location + patch.shape[0], :,
                        x_location:x_location + patch.shape[2], y_location:y_location + patch.shape[3]]

                end_time = time.time()

                mean_loss = (mean_loss * (numerate_index - 1) + loss_) / numerate_index

                # print(f"第{numerate_index}次,花费时间{end_time - begin_time}")
                print(f"干净样本{predicted.sum()}个",
                      f"攻击成功{((1. - predicted_new.eq(label_target).float()) * predicted).sum()}个",
                      f"累计攻击成功{train_success}个")

                if train_actual_total % 30 == 0:
                    patch = torch.round(patch)
                    test_success_rate = test_patch_untargeted(self.patch_type, patch, test_loader, model,
                                                              self.image_size, self.location_global,
                                                              self.target)
                    logger.info(
                        "Epoch:{} numerate_index: {} Patch attack success rate on testset: {:.3f}%".format(epoch,
                                                                                                           train_actual_total,
                                                                                                           100 * test_success_rate))

                    if test_success_rate > best_patch_success_rate:
                        best_patch_success_rate = test_success_rate
                        best_patch = patch

            # 每一个epoch 对 patch进行四舍五入
            patch = torch.round(patch)

            # 每一个干净样本更新patch后，看是否这个更新后的patch能够引起错误分类
            logger.info("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch,
                                                                                         100 * train_success / train_actual_total))

            test_success_rate = test_patch_untargeted(self.patch_type, patch, test_loader, model, self.image_size,
                                                      self.location_global, self.target)

            logger.info("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

            if test_success_rate > best_patch_success_rate:
                best_patch_success_rate = test_success_rate
                best_patch = patch
        # 保存每一个类对应的对抗补丁
        with open(best_patch_url, "wb") as file:
            list_best = [{
                "best_patch": best_patch
            }]
            pickle.dump(list_best, file, True)

        play_frame(best_patch, url_tif)
        logger.info("The best patch is found  with success rate {}% on testset".format(100 * best_patch_success_rate))
        return None
