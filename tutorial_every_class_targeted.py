import argparse
import os
import sys
from spikingjelly.activation_based import functional
import torch
from torch.utils.data import DataLoader

import attack
import data_loaders
from functions import (get_logger, seed_all, set_single_bp_way, set_mixed_bp_way, set_same_bp_way)
from attack import adversarial_patchs_targeted
from model import VGG, LeNet5

from model.VGG import *

from model.LeNet5 import *
from utils import val, val_success_rate

dataset = str(sys.argv[1])
model_ = str(sys.argv[2])
time = int(sys.argv[3])
target = int(sys.argv[4])
noise_percentage = float(sys.argv[5])
lr = float(sys.argv[6])
location_global = int(sys.argv[7])

workers = 0
seed = 42
suffix = ""
tau = 0.9
alph = 1.
v_reset = 0.
device = "0"
mask_type = 'rectangle'
epochs = 8
probability_threshold = 0.75
max_iteration = int(sys.argv[8])
batch_size = int(sys.argv[9])

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
    identifier += '_T[%d]' % time
    identifier += suffix

    # 需要攻击的模型所在的文件夹
    model_dir = './model-checkpoints/%s-checkpoints' % dataset

    # 存放日志的文件
    log_dir = f'./patch_attack_log/{dataset}-results/targeted/{target}/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 攻击模型的 identifier
    log_dir = os.path.join(log_dir, '%s.log' % identifier)
    logger = get_logger(log_dir)
    logger.info('start testing!')

    # 生成随机种子
    seed_all(seed)

    if 'dvsgesture' in dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvsgesture(root='E:\\data_set\\data_dvs'
                                                                               '\\DVSGesture', frames_number=time)
    elif 'dvscifar' in dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvscifar_patch(root='E:\\data_set\\data_dvs'
                                                                                   '\\CIFAR10DVS', frames_number=time)
    elif 'nmnist' in dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_nmnist(
            root='E:\\data_set\\data_dvs\\NMNIST', frame_number=time)
    else:
        raise AssertionError("data_dvs not supported")

    # 创建数据集加载器
    print("训练集长度:::::::::", train_dataset.__len__())  # 1176
    print("测试集长度:::::::::", val_dataset.__len__())  # 288

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False)

    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=False)

    test_loader_new = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                 num_workers=workers, pin_memory=False)

    if 'vgg' in model_.lower():
        model = VGG.VGGDvs(model_.lower(), tau=tau, alph=alph, v_reset=v_reset, init_s=init_s,
                           num_class=num_labels)
    elif 'lenet' in model_.lower():
        model = LeNet5.LeNet5Dvs(lenet_name=model_.lower(), tau=tau, alph=alph, v_reset=v_reset,
                                 init_s=init_s, num_class=num_labels)
    else:
        raise AssertionError("model not supported")

    functional.set_step_mode(model, step_mode='m')

    state_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    set_single_bp_way(model, "bptt")
    # set_same_bp_way(model, "bptr_new")  # ERGA

    # set_mixed_bp_way(model, "bptr_new")  # SERGA

    image_size = (20, 2, init_s, init_s)

    fun = val

    attack_dvs_targeted = adversarial_patchs_targeted.Event_patch_attack_targeted(model, fun, mask_type, image_size,
                                                                                  noise_percentage,
                                                                                  logger, epochs,
                                                                                  probability_threshold, max_iteration,
                                                                                  lr, target, location_global,
                                                                                  device)

    attack_dvs_targeted(train_loader, test_loader)


if __name__ == "__main__":
    main()
