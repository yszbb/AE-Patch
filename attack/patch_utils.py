import math
import numpy as np
import torch

MIDDLE = ((35, 40), (25, 30))
TOP_LEFT = ((5, 25), (10, 20))
TOP_RIGHT = ((5, 25), (30, 45))
GEN = ((5, 45), (5, 45))
receptive_field = {
    0: MIDDLE,  # y,x
    1: TOP_LEFT,
    2: TOP_RIGHT,
    3: TOP_LEFT,
    4: TOP_LEFT,
    5: TOP_RIGHT,
    6: TOP_RIGHT,
    7: MIDDLE,
    8: MIDDLE,
    9: MIDDLE,
    10: TOP_LEFT
}


# Initialize the patch  0.035 0.04 0.05 0.06
# TODO: Add circle type
def patch_initialization(patch_type='rectangle', image_size=(20, 2, 128, 128), noise_percentage=0.03):
    if patch_type == 'rectangle':
        patch_length = int((noise_percentage * image_size[2] * image_size[3]) ** 0.5)
        patch = torch.zeros(image_size[0], image_size[1], patch_length, patch_length)
        return patch


# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(20, 2, 64, 64), location_global=1, target=0):
    applied_patch = torch.zeros(image_size)

    mask = torch.zeros(image_size)
    mask_value = torch.ones(patch.shape)

    if mask_type == 'rectangle':
        if location_global != 0:
            # patch location  随机生成一个位置
            t_location = np.random.randint(low=0, high=1)
            x_location = np.random.randint(low=0, high=image_size[2] - patch.shape[2])
            y_location = np.random.randint(low=0, high=image_size[3] - patch.shape[3])
        else:
            x, y = receptive_field[target]
            t_location = np.random.randint(low=0, high=1)
            x_location = np.random.randint(low=x[0], high=x[1])
            y_location = np.random.randint(low=y[0], high=y[1])

        applied_patch[t_location:t_location + patch.shape[0], :, x_location:x_location + patch.shape[2],
        y_location:y_location + patch.shape[3]] = patch

        mask[t_location:t_location + patch.shape[0], :, x_location:x_location + patch.shape[2],
        y_location:y_location + patch.shape[3]] = mask_value

    applied_patch = applied_patch.unsqueeze(1)  # 将 applied_patch  mask shape [T, 1, 2, len, len]
    mask = mask.unsqueeze(1)

    return applied_patch, mask, x_location, y_location, t_location


# Test the patch on dataset
# Batchsize 不一样了
def test_patch_untargeted(patch_type, patch, test_loader, model, image_size, location_global, target):
    model.eval()
    index = 0
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        index += 1

        # if index > 50:
        #     break

        test_total += label.shape[0]

        image = image.transpose(0, 1)  # T*batchszie*2*w*h
        image = torch.clamp(image, 0, 1)

        image = image.cuda()
        label = label.cuda()
        output = model(image).mean(0)
        _, predicted_old = torch.max(output.data, 1)
        predicted = predicted_old.eq(label).float()  # 把正确的样本挑出来
        pred = (predicted_old == target).float()

        if predicted.sum() == 0:
            continue

        if target != -1:
            if (predicted * pred).sum() == 0:
                continue

        applied_patch, mask, x_location, y_location, t_location = mask_generation(patch_type, patch,
                                                                                  image_size=image_size,
                                                                                  location_global=location_global,
                                                                                  target=target)

        perturbated_image = (torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) +
                             torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor)))

        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image).mean(0)
        _, predicted_adv = torch.max(output.data, 1)

        if target == -1:
            # 只有正确分类的样本才会进行测试
            test_actual_total += predicted.sum()
            predicted_adv = predicted_adv.eq(label).float()

            test_success = test_success + ((1. - predicted_adv) * predicted).sum()
        else:
            predicted = predicted * pred
            test_actual_total += predicted.sum()
            # 只有预测正确  且除去原始为target的样本
            label_target = torch.full((1, label.shape[0]), target)
            label_target = label_target.cuda()
            test_success = test_success + ((1. - predicted_adv.eq(label_target).float()) * predicted).sum()

    return test_success / test_actual_total


def test_patch_targeted(patch_type, patch, test_loader, model, image_size, location_global, target):
    model.eval()
    index = 0
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        index += 1

        # if index > 50:
        #     break

        test_total += label.shape[0]

        image = image.transpose(0, 1)  # T*batchszie*2*w*h
        image = torch.clamp(image, 0, 1)
        image = image.cuda()
        label = label.cuda()
        output = model(image).mean(0)
        _, predicted_old = torch.max(output.data, 1)
        predicted = predicted_old.eq(label).float()  # 把正确的样本挑出来
        pred = (predicted_old != target).float()

        predicted = predicted * pred
        if predicted.sum() == 0:
            continue

        applied_patch, mask, x_location, y_location, t_location = mask_generation(patch_type, patch,
                                                                                  image_size=image_size,
                                                                                  location_global=location_global,
                                                                                  target=target)

        perturbated_image = (torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) +
                             torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor)))

        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image).mean(0)
        _, predicted_adv = torch.max(output.data, 1)

        test_actual_total += predicted.sum()
        # 只有预测正确  且除去原始为target的样本
        label_target = torch.full((1, label.shape[0]), target)
        label_target = label_target.cuda()
        test_success = test_success + ((predicted_adv.eq(label_target).float()) * predicted).sum()

    return test_success / test_actual_total
