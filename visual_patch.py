import pickle

import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader

import data_loaders
from attack.patch_utils import mask_generation
from model import VGG, LeNet5
from model.LeNet5 import *
from spikingjelly.activation_based import functional

# class_labels = ["hand clapping", "right hand wave", "left hand wave", "right arm clockwise",
#                 "right arm counter clockwise", "left arm clockwise", "left arm counter clockwise", "arm roll",
#                 "air drums",
#                 "air guitar", "other gestures"
#                 ]
class_labels = ["hand clapping", "RHW", "LHW", "right arm clockwise",
                "right arm counter clockwise", "LAC", "left arm counter clockwise", "arm roll",
                "air drums",
                "air guitar", "other gestures"
                ]
colors = [(1., 0., 0., 0.0), (1., 0., 0., 1.0)]
nodes = [0, 1]
cmap = mpl.colors.LinearSegmentedColormap.from_list("transparent_red", list(zip(nodes, colors)))


# obj = {
#     "patch_source": patch_source,
#     "image_source": image_source,
#     "source_label": source_label,
#     "attack_label": predicted_adv[0]
# }
def plot(args):
    axes, sample, idx, class_labels, T, is_last = args

    dt = T / len(axes)

    time_labels = ["TS: %d" % (dt * i) for i in range(len(axes))]

    X0 = sample["image_source"].squeeze().sum(dim=1)  # shape T * W * H
    X_diff = sample["patch_source"].squeeze().sum(dim=1)

    num_frames_available = len(axes)
    num_frames = X0.shape[0]
    t = int(num_frames / num_frames_available)
    frames_X0 = [X0[i * t].cpu().numpy()[::-1] for i in range(len(axes))]
    frames_X_diff = [X_diff[i * t].cpu().numpy()[::-1] for i in range(len(axes))]

    for ax_idx, (frame, frame_diff) in enumerate(zip(frames_X0, frames_X_diff)):
        axes[ax_idx].pcolormesh(frame, vmin=0, vmax=2, cmap=plt.cm.gray_r)
        axes[ax_idx].pcolormesh(frame_diff, vmin=0, vmax=1, cmap=cmap)
        if is_last:
            axes[ax_idx].text(3, 4, time_labels[ax_idx], color="blue", fontsize=15, backgroundcolor="white")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel(
                class_labels[sample["source_label"]] + r"$\rightarrow$" + class_labels[sample["attack_label"]],
                color="blue", fontsize=15)


def plot_figure(samples, T, index):
    # len(samples) == 3
    # 画图开始的地方
    N_rows = 3
    N_cols = 5
    num_per_sample = int(N_rows * N_cols / len(samples))

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))

    spec = gridspec.GridSpec(ncols=N_cols, nrows=N_rows, figure=fig)
    axes = [fig.add_subplot(spec[i, j]) for i in range(N_rows) for j in range(N_cols)]

    for ax in axes:
        ax.set_aspect("equal")
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       top=False,
                       labelbottom=False,
                       right=False,
                       left=False,
                       labelleft=False)

    sub_axes_samples = [(axes[i * num_per_sample:(i + 1) * num_per_sample], samples[i], i, class_labels, T, True) for
                        i in range(len(samples))]

    list(map(plot, sub_axes_samples))
    plt.savefig("D:\\SNN-attack\\ECAI-SNN-attack\\patchs\\adversarial_patch_%d.pdf" % index)


def select_sample(patch_type, patch, test_loader, model, image_size, location_global, target):
    samples = []
    model.eval()
    index = 0
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        source_label = target
        attack_label = None

        test_total += label.shape[0]
        image = image.transpose(0, 1)  # T*batchszie*2*w*h
        image = torch.clamp(image, 0, 1)

        image = image.cuda()
        label = label.cuda()
        output = model(image).mean(0)
        _, predicted_old = torch.max(output.data, 1)
        predicted = predicted_old.eq(label).float()  # 把正确的样本挑出来
        pred = (predicted_old == target).float()
        predicted = predicted * pred

        if predicted.sum() == 0:
            continue

        applied_patch, mask, x_location, y_location, t_location = mask_generation(patch_type, patch,
                                                                                  image_size=image_size,
                                                                                  location_global=location_global,
                                                                                  target=target)
        patch_source = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor))
        image_source = torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = patch_source + image_source
        perturbated_image = perturbated_image.cuda()

        output = model(perturbated_image).mean(0)
        _, predicted_adv = torch.max(output.data, 1)

        # 只有预测正确  且除去原始为target的样本
        label_target = torch.full((1, label.shape[0]), target)
        label_target = label_target.cuda()

        if ((1. - predicted_adv.eq(label_target).float()) * predicted).sum() > 0:
            index += 1
            obj = {
                "patch_source": patch_source.squeeze(),
                "image_source": image_source.squeeze(),
                "source_label": source_label,
                "attack_label": predicted_adv[0].item()
            }
            samples.append(obj)
        if index > 2:
            break

    return samples


if __name__ == "__main__":
    batch_size = 1
    time = 20
    workers = 0
    dataset = "dvsgesture"
    model_ = "lenet5_simple"
    tau = 0.9
    alph = 1.
    v_reset = 0.
    init_s = 64
    num_labels = 11
    device = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
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
    # print("训练集长度:::::::::", train_dataset.__len__())  # 1176
    # print("测试集长度:::::::::", val_dataset.__len__())  # 288

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

    suffix = ""
    model_dir = './model-checkpoints/%s-checkpoints' % dataset
    identifier = model_
    identifier += '_T[%d]' % time
    identifier += suffix

    functional.set_step_mode(model, step_mode='m')
    state_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    # the core of method ######
    # set_single_bp_way(model, "bptt")
    # set_same_bp_way(model, "bptr_new")  # ERGA

    image_size = (time, 2, init_s, init_s)
    for i in range(11):
        path_patch = "./patchs/untargeted/%d/best_patch.pkl" % i
        with open(path_patch, "rb") as file:
            list1 = pickle.load(file)
            patch = list1[0]["best_patch"]

            samples = select_sample('rectangle', patch, test_loader, model, image_size, 1., i)

            plot_figure(samples, time, i)
