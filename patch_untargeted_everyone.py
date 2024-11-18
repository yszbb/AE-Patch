import os
# patch 攻击 模型在clamp的数据集上训练的，patch也是在clamp数据集上训练的  T=20
# 为每一个目标类设置的定向攻击
if __name__ == "__main__":
    data_dvs = "dvscifar"
    arch = "vggdvs"
    # data_dvs = "dvsgesture"
    # arch = "lenet5_simple"
    if data_dvs == "dvscifar":
        class_num = 10
        max_iteration = 8
        batch_size = 4
    else:
        class_num = 11
        max_iteration = 10
        batch_size = 1

    T = 20
    noise_percentage = 0.08
    lr = 0.1
    location_global = 1  # 代表全局位置任意

    for target in range(0, class_num):
        os.system(f"python tutorial_every_class_untargeted.py {data_dvs} {arch} {T} {target} {noise_percentage} {lr} {location_global} {max_iteration} {batch_size}")