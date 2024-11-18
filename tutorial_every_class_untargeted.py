import argparse
import os
import sys
from spikingjelly.activation_based import functional
import torch
from torch.utils.data import DataLoader
import time
import attack
import data_loaders
from functions import (get_logger, seed_all, set_single_bp_way, set_mixed_bp_way, set_same_bp_way)
from attack import adversarial_patchs_every_class_untargeted
from model import VGG, LeNet5

from model.VGG import *

from model.LeNet5 import *
from utils import val, val_success_rate
from torchvision import transforms

#  为每一个类单独训练一个patch，全局位置任意，非目标攻击
dataset = str(sys.argv[1])
model_ = str(sys.argv[2])
T = int(sys.argv[3])
target = int(sys.argv[4])
noise_percentage = float(sys.argv[5])
lr = float(sys.argv[6])
location_global = int(sys.argv[7])

# dvscifar10    5       dvsgesture  10
max_iteration = int(sys.argv[8])
batch_size = int(sys.argv[9])
workers = 0

seed = 42
suffix = ""
tau = 0.9
alph = 1.
v_reset = 0.
device = "0"
mask_type = 'rectangle'
epochs = 8
probability_threshold = 0.2

os.environ["CUDA_VISIBLE_DEVICES"] = device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # global args

    if dataset.lower() == 'dvscifar':
        num_labels = 10
        init_s = 64
    elif dataset.lower() == 'dvsgesture':
        num_labels = 11
        init_s = 64
    elif dataset.lower() == 'nmnist':
        num_labels = 10
        init_s = 34

    # identifier 就是模型的名字
    identifier = model_
    identifier += '_T[%d]' % T
    identifier += suffix

    # 需要攻击的模型所在的文件夹
    model_dir = './model-checkpoints/%s-checkpoints' % dataset

    # 存放日志的文件

    log_dir = f'./patch_attack_log/{dataset}-results/untargeted/{target}/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 攻击模型的 identifier
    log_dir = os.path.join(log_dir, '%s.log' % identifier)
    logger = get_logger(log_dir)
    logger.info('start testing!')

    # 生成随机种子
    # seed_all(seed)

    if 'dvsgesture' in dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvsgesture(root='E:\\data_set\\data_dvs'
                                                                               '\\DVSGesture', frames_number=T)
    elif 'dvscifar' in dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvscifar_patch(root='E:\\data_set\\data_dvs'
                                                                                   '\\CIFAR10DVS', frames_number=T)
    elif 'nmnist' in dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_nmnist(
            root='E:\\data_set\\data_dvs\\NMNIST', frame_number=T)
    else:
        raise AssertionError("data_dvs not supported")

    # 创建数据集加载器
    print("训练集长度:::::::::", train_dataset.__len__())  # 1176
    print("测试集长度:::::::::", val_dataset.__len__())  # 288

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False)

    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=False)

    if 'vgg' in model_.lower():
        model = VGG.VGGDvs(model_.lower(), tau=tau, alph=alph, v_reset=v_reset, init_s=init_s,
                           num_class=num_labels)
    elif 'lenet' in model_.lower():
        model = LeNet5.LeNet5Dvs(lenet_name=model_.lower(), tau=tau, alph=alph, v_reset=v_reset,
                                 init_s=init_s, num_class=num_labels)
    else:
        raise AssertionError("model not supported")

    # # 查看每一类的分布情况
    # class_nums = [0] * 10
    # for _, class_labels in train_loader:
    #     for class_label in class_labels:
    #         class_nums[class_label.item()] += 1
    #
    # class_nums_ = [0] * 10
    # for _, class_labels in test_loader:
    #     for class_label in class_labels:
    #         class_nums_[class_label.item()] += 1
    #
    # print(class_nums, "+++++++++++++++++++++")
    # print(class_nums_, "_____________________")

    functional.set_step_mode(model, step_mode='m')

    state_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    # the core of method ######
    # set_single_bp_way(model, "bptt")

    # set_mixed_bp_way(model, "bptr_old")  # SRGA
    # set_same_bp_way(model, "bptr_old")   # RGA

    # set_mixed_bp_way(model, "bptr_new")  # SERGA
    set_same_bp_way(model, "bptr_new")  # ERGA

    image_size = (T, 2, init_s, init_s)

    fun = val

    # for (image, label) in train_loader:
    #     image = image.transpose(0, 1)  # T*batchszie*2*w*h
    #     # 进行预处理
    #     image = torch.clamp(image.data, min=0., max=1.)
    #
    #     image = image.cuda()
    #     label = label.cuda()
    #     output = model(image).mean(0)  # batchsize * labels
    #
    #     # 预测正确的置信度
    #     target_probability = torch.nn.functional.softmax(output, dim=1)
    #
    #     score, predicted_old = torch.max(target_probability.data, 1)  # batchsize * labels
    #     # 把正确的且属于该类的样本挑选出来
    #     predicted = predicted_old.cuda()
    #     predicted = predicted.eq(label).float()
    #     pred = (predicted_old == target).float()
    #     predicted = predicted * pred
    #     if predicted.sum() == 0:
    #         print("没有符合条件的样本,跳过这个batchsize样本")
    #         continue
    #
    #     x = image.squeeze(1)
    #
    #     to_img = transforms.ToPILImage()
    #     img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    #     img_tensor[:, 1] = x[:, 0]
    #     img_tensor[:, 2] = x[:, 1]
    #     path_total = './pictures/%d-target/' % target
    #     if not os.path.exists(path_total):
    #         os.makedirs(path_total)
    #     img_list = []
    #     for t in range(img_tensor.shape[0]):
    #         path_total_ = path_total + '%d' % t + '.png'
    #         to_img(img_tensor[t]).save(path_total_)
    #
    #     # img_list.append(to_img(img_tensor[t]))
    #     # img_list[0].save(save_gif_to, save_all=True, append_images=img_list[1:], loop=0)
    #     # print(f'Save frames to [{save_gif_to}].')
    #     break
    begin_time = time.clock()
    attack_dvs = adversarial_patchs_every_class_untargeted.Event_patch_attack_target(model, fun, mask_type, image_size,
                                                                                     noise_percentage,
                                                                                     logger, epochs,
                                                                                     probability_threshold,
                                                                                     max_iteration, lr, target,
                                                                                     location_global, device)

    attack_dvs(train_loader, test_loader)
    end_time = time.clock()
    logger.info("target:%d" % target)
    logger.info("time consume: %f" % (end_time - begin_time))


if __name__ == "__main__":
    main()
