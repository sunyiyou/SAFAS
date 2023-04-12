import numpy as np

setting = {
    'ptoto': {'arg': '--protocol', 'value': ['O_C_I_to_M'], 'inname': True},  #O_C_I_to_M, O_M_I_to_C, O_C_M_to_I, I_C_M_to_O
    'bsz': {'arg': '--batch_size', 'value': [96, 128], 'inname': True},
    'aepoch': {'arg': '--align_epoch', 'value': [0, 10, 20], 'inname': True},
    'rot': {'arg': '--train_rotation', 'value': [False], 'inname': True},
    'lr': {'arg': '--base_lr', 'value': [0.015, 0.01, 0.02], 'inname': True},
    'alpha': {'arg': '--alpha', 'value': [0.99, 0.999], 'inname': True},
    'scale': {'arg': '--scale', 'value': [0.5, 1], 'inname': True},
    'floss': {'arg': '--feat_loss', 'value': ['supcon'], 'inname': True},
    'flossw': {'arg': '--feat_loss_weight', 'value': [0.4, 0.5, 0.6], 'inname': True},
    'temp': {'arg': '--temperature', 'value': [0.1, 0.07], 'inname': True},
    'seed': {'arg': '--seed', 'value': list(range(1000)), 'inname': True},
    'pre': {'arg': '--pretrain', 'value': ['imagenet'], 'inname': True}
}

name_used = set()
command_list = []

def command_generator():
    args_msgs = []
    for token, item in setting.items():
        arg = setting[token]['arg']
        val = np.random.choice(setting[token]['value'])
        args_msgs.append(f"{arg} {val}")
    args_msg = " ".join(args_msgs)

    command = "python train.py " + args_msg
    return command, args_msg


CMD_PER_GPU = 25

gpuids = [0, 1, 2, 3, 4, 5, 6, 7]

for gpu in gpuids:
    for i in range(CMD_PER_GPU):
        command, name = command_generator()
        command_gpu = f"CUDA_VISIBLE_DEVICES={gpu} " + command
        if name not in name_used:
            command_list.append(command_gpu)
            name_used.add(name)
        print(command_gpu)
    print()
    print()


