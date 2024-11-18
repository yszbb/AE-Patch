import argparse
import copy
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

import attack
import data_loaders
from functions import (get_logger, seed_all, set_single_bp_way, set_mixed_bp_way, set_same_bp_way)
from attack import (fgsm, pgd)
from model import VGG, LeNet5
from spikingjelly.activation_based import functional
from model.VGG import *

from model.LeNet5 import *
from utils import val, val_success_rate

parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data_dvs loading workers')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size')
# 随机种子
parser.add_argument('-sd', '--seed', default=42, type=int, help='seed for initializing training.')
parser.add_argument('-suffix', '--suffix', default='', type=str, help='suffix')

# data_dvs = "dvsgesture"
# arch = "lenet5_simple"
# data_dvs = "dvscifar"
# arch = "vggdvs"
# model configuration
parser.add_argument('-data_dvs', '--dataset', default='dvsgesture', type=str, help='dataset')
#  时间窗口大小
parser.add_argument('-T', '--time', default=20, type=int, metavar='N', help='snn simulation time, set 0 as ANN')
parser.add_argument('-arch', '--model', default='lenet5_simple', type=str, help='model')

parser.add_argument('-tau', '--tau', default=0.9, type=float, metavar='N', help='leaky constant')

# 阶跃函数反向求导
parser.add_argument('-alph', '--alph', default=1., type=float, metavar='N', help='leaky constant')
parser.add_argument('-v_reset', '--v_reset', default=0., type=float, metavar='N', help='enhance attack')

# 攻击配置设置
parser.add_argument('-config', '--config', default='config', type=str, help='test configuration file')

# training configuration
parser.add_argument('-dev', '--device', default='0', type=str, help='device')

# {"attack": "fgsm", "eps": 2.55, "bb": true},
# {"attack": "pgd", "eps": 2.55, "bb": true, "alpha": 2, "steps":5},
# adv atk configuration
parser.add_argument('-attack_mode', '--attack_mode', default="bptt", type=str, help='attack mode')
parser.add_argument('-atk', '--attack', default='', type=str, help='attack')
parser.add_argument('-eps', '--eps', default=8, type=float, metavar='N', help='attack eps')
# 决定是进行白盒攻击还是进行黑盒攻击
parser.add_argument('-bb', '--bbmodel', default='', type=str, help='black box model')
# only pgd
parser.add_argument('-alpha', '--alpha', default=2.55 / 1, type=float, metavar='N', help='pgd attack alpha')
parser.add_argument('-steps', '--steps', default=7, type=int, metavar='N', help='pgd attack steps')

parser.add_argument('-stdout', '--stdout', default='', type=str, help='log file')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# onebyone attack  model是在没有clamp数数据集上训练的。攻击是在clamp数据集上进行的  T=10
def main():
    global args

    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
    elif args.dataset.lower() == 'dvscifar':
        num_labels = 10
        init_s = 64
    elif args.dataset.lower() == 'dvsgesture':
        num_labels = 11

        init_s = 64
    elif args.dataset.lower() == 'nmnist':
        num_labels = 10
        init_s = 34

    # 需要攻击的模型所在的文件夹
    model_dir = './model-checkpoints/%s-checkpoints' % (args.dataset)

    # 存放日志的文件
    log_dir = './log/%s-attack-log' % (args.dataset)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    identifier = args.model
    identifier += '_T[%d]' % args.time
    identifier += args.suffix
    # 攻击模型的 identifier
    logger = get_logger(os.path.join(log_dir, '%s.log' % identifier))
    logger.info('start testing!')

    # 生成随机种子
    seed_all(args.seed)

    if 'dvsgesture' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvsgesture(root='E:\\data_set\\data_dvs'
                                                                               '\\DVSGesture', frames_number=args.time)
    elif 'dvscifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvscifar_patch(root='E:\\data_set\\data_dvs'
                                                                                   '\\CIFAR10DVS',
                                                                              frames_number=args.time)
    elif 'nmnist' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_nmnist(
            root='E:\\data_set\\data_dvs\\NMNIST', frame_number=args.time)
    else:
        raise AssertionError("data_dvs not supported")

    # 创建测试数据集加载器
    test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=False)

    # 创建模型

    if 'vgg' in args.model.lower():
        model = VGG.VGGDvs(vgg_name=args.model.lower(), tau=args.tau, alph=args.alph, v_reset=args.v_reset,
                           init_s=init_s,
                           num_class=num_labels)  # cifar10dvs
    elif 'lenet' in args.model.lower():
        model = LeNet5.LeNet5Dvs(lenet_name=args.model.lower(), tau=args.tau, alph=args.alph, v_reset=args.v_reset,
                                 init_s=init_s,
                                 num_class=num_labels)
    else:
        raise AssertionError("model not supported")

    model.to(device)

    functional.set_step_mode(model, step_mode='m')
    #  白盒攻击
    bbmodel = copy.deepcopy(model)
    bbstate_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
    bbmodel.load_state_dict(bbstate_dict, strict=False)

    state_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # 加载json文件中的配置  [{}] 数组里面都是对象
    if len(args.config) > 0:
        with open(args.config + '.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]

    for atk_config in config:
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])

        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = bbmodel  # 白盒攻击
        else:
            atkmodel = model  # 黑盒攻击 没有进行训练的模型
        # 默认用bptt进行梯度回传
        set_single_bp_way(atkmodel, "bptt")

        if atk_config['attack_mode'] == 'bptt':
            set_single_bp_way(atkmodel, "bptt")
        elif atk_config['attack_mode'] == 'bptr_old':
            set_same_bp_way(atkmodel, "bptr_old")
        elif atk_config['attack_mode'] == 'bptr_new':
            set_same_bp_way(atkmodel, "bptr_new")
        elif atk_config['attack_mode'] == 'mbptr_new':
            set_mixed_bp_way(atkmodel, "bptr_new")
        else:
            set_mixed_bp_way(atkmodel, "bptr_old")
        if args.attack.lower() == 'fgsm':
            atk = fgsm.FGSM(atkmodel, eps=args.eps / 255, alph=args.alpha / 255, v_reset=args.v_reset)
        elif args.attack.lower() == 'pgd':
            atk = pgd.PGD(atkmodel, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, alph=args.alph,
                          v_reset=args.v_reset)
        else:
            atk = None

        if atk is not None:
            acc, mean_queries = val_success_rate(test_loader, device, atk)
            logger.info("attack mode is :::::::  %s" % atk_config['attack_mode'])
            logger.info(acc)
            logger.info(mean_queries)


if __name__ == "__main__":
    main()
