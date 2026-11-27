import os
import json
from collections import OrderedDict


def parse(opt_path, is_train=True):
    json_str = ''
    with open(opt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase.split('_')[0]
        dataset['n_channels'] = opt['n_channels']

    for key, path in opt['path'].items():
        if path:
            opt['path'][key] = os.path.expanduser(path)

    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task
    if is_train:
        opt['path']['models'] = os.path.join(path_task, 'models')
        opt['path']['images'] = os.path.join(path_task, 'images')
    else:
        opt['path']['images'] = os.path.join(path_task, 'test_images')

    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    opt['num_gpu'] = len(opt['gpu_ids'])
    print('number of GPUs is: ' + str(opt['num_gpu']))

    return opt


def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg
