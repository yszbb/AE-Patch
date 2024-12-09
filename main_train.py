import warnings
from functions import *
import argparse
import os
import warnings
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from spikingjelly.activation_based import functional
from attack import *
import data_loaders
# from model import ResNet
from model import VGG, LeNet5
from utils import train, val

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data_dvs loading workers')
parser.add_argument('-b', '--batch_size', default=80, type=int, metavar='N', help='mini-batch size')
#  优化函数选择
parser.add_argument('--optim', default='sgd', type=str, help='model')
# model configuration
#  使用那个数据集
parser.add_argument('-data_dvs', '--dataset', default='dvsgesture', type=str, help='dataset')
#  使用那个模型
parser.add_argument('-arch', '--model', default='lenet5_simple', type=str, help='model')
#  时间窗口大小
parser.add_argument('-T', '--time', default=20, type=int, metavar='N', help='snn simulation time, set 0 as ANN')
#  膜时间常数大小
parser.add_argument('-tau', '--tau', default=0.9, type=float, metavar='N', help='leaky constant')
# 阶跃函数反向求导 # 加快求导速度
parser.add_argument('-alph', '--alph', default=1., type=float, metavar='N', help='leaky constant')
# 选择重置方式
parser.add_argument('-v_reset', '--v_reset', default=0., type=float, metavar='N', help='leaky constant')
# training configuration
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-lr', '--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-dev', '--device', default='0', type=str, help='device')
#  权重衰减率
parser.add_argument('-wd', '--wd', default=5e-4, type=float, help='weight decay')

# 初始化训练
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('-suffix', '--suffix', default='', type=str, help='suffix')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args

    if args.dataset.lower() == 'dvscifar':
        num_labels = 10
        init_s = 48
    elif args.dataset.lower() == 'dvsgesture':
        num_labels = 11
        init_s = 64
    elif args.dataset.lower() == 'nmnist':
        num_labels = 10
        init_s = 34

    # >>>>>>>IMPORTANT<<<<<<<< Edit log_dir
    #  创建保存模型的文件夹
    model_save_dir = './model-checkpoints/%s-checkpoints' % args.dataset
    log_dir = './log/%s-log' % args.dataset

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 需要进一步探究其作用
    seed_all(args.seed)

    if 'dvsgesture' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvsgesture(root='data_dvs/DVSGesture', frames_number=args.time)
    elif 'dvscifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvscifar(root='data_dvs/CIFAR10DVS', frames_number=args.time)
    elif 'nmnist' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_nmnist(
            root='data_dvs/NMNIST', frame_number=args.time)
    else:
        raise AssertionError("data_dvs not supported")

    # 创建数据集加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=False)

    if 'vgg' in args.model.lower():
        model = VGG.VGGDvs(vgg_name=args.model.lower(), tau=args.tau, alph=args.alph, v_reset=args.v_reset, init_s=init_s,
                           num_class=num_labels)  # cifar10dvs
    elif 'lenet' in args.model.lower():
        model = LeNet5.LeNet5Dvs(lenet_name=args.model.lower(), tau=args.tau, alph=args.alph, v_reset=args.v_reset, init_s=init_s,
                                 num_class=num_labels)
    else:
        raise AssertionError("model not supported")

    # 创建log文件
    # identifier 就是模型的名字
    identifier = args.model
    identifier += '_T[%d]' % args.time
    identifier += args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log' % identifier))

    # 先用T = 10 进行训练，然后再用T=20进行微调,训练出T=20的模型
    model.to(device)
    functional.set_step_mode(model, step_mode='m')
    model.set_bp_single("bptt")

    if os.path.exists(os.path.join(model_save_dir, '%s.pth' % identifier)):
        state_dict = torch.load(os.path.join(model_save_dir, '%s.pth' % identifier), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 定义最好的精确率
    best_acc = 0

    # IMPORTANT<<<<<<<<<<<<< modifed

    logger.info('start training!')
    for epoch in range(args.epochs):
        # 训练过程
        loss, acc = train(model, device, train_loader, criterion, optimizer)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        scheduler.step()
        # 测试过程
        tmp = val(model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch, args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), os.path.join(model_save_dir, '%s.pth' % identifier))

        logger.info('Best Test acc={:.3f}'.format(best_acc))


if __name__ == "__main__":
    main()
