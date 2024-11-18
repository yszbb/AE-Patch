import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
from spikingjelly.datasets import play_frame


def train(model, device, train_loader, criterion, optimizer):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        # T * batchsize * c * w * h
        images = images.transpose(0, 1)
        outputs = model(images).mean(0)

        loss = criterion(outputs, labels)

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        # 求样本总数
        total += float(labels.size(0))

        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


def val(model, test_loader, device, clamp_=False):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs.transpose(0, 1)
        ###### ++++++++++++++++  ######
        if clamp_:
            inputs = torch.clamp(inputs, 0, 1)

        with torch.no_grad():
            outputs = model(inputs).mean(0)

        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets.cpu()).sum().item())
    final_acc = 100 * correct / total
    return final_acc


def val_success_rate(test_loader, device, atk=None):
    total_queries = 0
    total_L0 = 0
    total_test = 0
    success_num = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        X0 = inputs.to(device)
        X0 = torch.clamp(X0, 0.0, 1.0)

        if atk is not None:
            return_dict = atk(X0, targets.to(device))

        if return_dict is None:
            continue

        X_adv_attack = return_dict["X_adv"]
        original_prediction = return_dict["predicted"]
        predicted_attacked = return_dict["predicted_attacked"]

        return_dict["X_ori"] = X0[0]
        return_dict["X_adv"] = X_adv_attack[0]

        print("+++++++++++Original prediction %d ++++++++++ predicted_attacked %d ++++++++++" % (
            original_prediction, predicted_attacked))

        total_test += 1
        if return_dict["success"]:
            success_num += 1
            total_queries += return_dict["n_queries"]

        if batch_idx >= 1000:
            break

    acc = 100 * success_num / total_test

    # 攻击成功的样本平均查询次数
    mean_queries = total_queries / success_num

    return acc, mean_queries
