import torch
import torchvision
import os
import utils

import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.modelq import Quantize

from imcfunc import *
#from imcfunc import add_model
#from imcfunc import run_mnsim
#from imcfunc import update_conf

assert torch.cuda.is_available()

train_dataprovider, val_dataprovider = None, None
test_dataprovider = None
class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


def get_cand_err(cand, args):
    max_train_iters = args.max_train_iters
    max_test_iters = args.max_test_iters
    top1 = 0
    top5 = 0
    total = 0
    #try:
    if True:
        accuracy, latency, energy, area = get_results(cand)
        #accuracy, latency, energy, area = run_mnsim(str([6, 8, 8]))
        accuracy = accuracy * 100
        latency = latency * 1e-6
        energy = energy * 1e-6
        latency_norm = 100*(latency-0.164)/(1.414-0.164)
        energy_norm = 100*(energy-0.033)/(63-0.033)
        print('latency: ' + str(latency) + ' Energy: ' + str(energy))
        #return top1**2/(latency*energy), top5**2/(latency*energy)
        return 0.99*accuracy-0.01*latency_norm*energy_norm, None#0.8*top5-0.2*latency_norm*energy_norm
    #except:
    #    return False
    #return top1, top5


def main():
    pass
