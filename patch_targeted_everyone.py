import os

# 为每一个目标类设置的定向攻击
if __name__ == "__main__":
    data_dvs = "dvsgesture"
    arch = "lenet5_simple"
    # data_dvs = "dvscifar"
    # arch = "vggdvs"
    if data_dvs == "dvscifar":
        class_num = 10
        max_iteration = 8
        batch_size = 4
    else:
        class_num = 11
        max_iteration = 10
        # batch_size = 10
        batch_size = 1
    T = 20
    noise_percentage = 0.06
    lr = 0.1
    location_global = 1  # 0代表局部任意  1 代表全局任意

    for target in range(0, class_num):
        os.system(f"python tutorial_every_class_targeted.py {data_dvs} {arch} {T} {target} {noise_percentage} {lr} {location_global} {max_iteration} {batch_size}")

