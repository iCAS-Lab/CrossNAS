import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots"):
        os.makedirs("./snapshots")
    filename = os.path.join("./snapshots/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)


def random_choice(num_choice, layers):
    ran_ch = list(np.random.randint(num_choice-1, size=np.random.randint(1,layers+1)))
    while ran_ch.count(0)>4:
        ran_ch = list(np.random.randint(num_choice-1, size=np.random.randint(1,layers+1)))
    for num in range(layers - len(ran_ch)):
        ran_ch.append(3)
    return ran_ch


def plot_hist(acc_list, min=0, max=101, interval=5, name='search'):
    plt.hist(acc_list, bins=max - min, range=(min, max), histtype='bar')
    plt.xticks(np.arange(min, max, interval))
    img_path = name + '.png'
    plt.savefig(img_path)
    plt.show()


def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))
