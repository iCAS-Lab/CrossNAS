import torch
import torchvision
import os
import utils

#from imagenet_dataset import get_train_dataprovider, get_val_dataprovider
import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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


@no_grad_wrapper
def get_cand_err(model, cand, args):
    global train_dataprovider, val_dataprovider, test_provider
    if train_dataprovider is None:
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #generator1 = torch.Generator().manual_seed(42)
        trainset = datasets.CIFAR10(
            root=os.path.join(args.data_root, args.dataset), train=True, download=True, transform=transform_train)

        trainset, valset = torch.utils.data.random_split(trainset, [42500, 7500])
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.train-batch-size, shuffle=True, num_workers=2)

        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.test-batch-size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(
            root=os.path.join(args.data_root, args.dataset), train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        train_dataprovider = DataIterator(train_loader)
        val_dataprovider = DataIterator(val_loader)

        #args.max_train_iters = len(train_loader)
        #args.max_test_iters = len(val_loader)

    max_train_iters = args.max_train_iters
    max_test_iters = args.max_test_iters
    
    
    cand = list(cand)
    for num in range(len(cand)):
        if cand[num] == 9:
            cand.pop(num)
            cand.append(9)
    cand = tuple(cand)
    print("cand = ", cand)
    choice = [x//3 for x in cand]
    kchoice = [x%3 for x in cand]
    print("choice = ", choice)
    print("kchoice = ", kchoice)

    print('clear bn statics....')
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
    
    print('train bn with training set (BN sanitize) ....')
    model.train()
    
    for step in tqdm.tqdm(range(max_train_iters)):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        data, target = train_dataprovider.next()
        # print('get data',data.shape)
        target = target.type(torch.LongTensor)
        data, target = data.to(args.device), target.to(args.device)
        output = model(data, choice, kchoice)
        del data, target, output
    
    top1 = 0
    top5 = 0
    total = 0
    print('starting test....')
    model.eval()
    for step in tqdm.tqdm(range(max_test_iters)):
        # print('test step: {} total: {}'.format(step,max_test_iters))
        data, target = val_dataprovider.next()
        batchsize = data.shape[0]
        # print('get data',data.shape)
        target = target.type(torch.LongTensor)
        data, target = data.to(args.device), target.to(args.device)
        logits = model(data, choice, kchoice)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        # print(prec1.item(),prec5.item())
        top1 += prec1.item() * batchsize
        top5 += prec5.item() * batchsize
        total += batchsize
        del data, target, logits, prec1, prec5
    top1, top5 = top1 / total, top5 / total
    #top1, top5 = top1 / 100, top5 / 100
    #top1, top5 = 1 - top1 / 100, 1 - top5 / 100
    #print('top1: {:.2f} top5: {:.2f}'.format(top1 * 100, top5 * 100))
    print('top1: {:.2f} top5: {:.2f}'.format(top1, top5))
    return top1, top5
    # uncomment for latency and energy
    '''
    add_model(choice, kchoice, cand)
    try:
        latency,energy = run_mnsim(str(cand))
        latency = latency * 1e-6
        energy = energy * 1e-6
        latency_norm = 100*(latency-0.164)/(1.414-0.164)
        energy_norm = 100*(energy-0.033)/(63-0.033)
        print('latency: ' + str(latency) + ' Energy: ' + str(energy))
        #return top1**2/(latency*energy), top5**2/(latency*energy)
        return 0.8*top1-0.2*latency_norm*energy_norm, 0.8*top5-0.2*latency_norm*energy_norm
    except:
        return False
    '''
    


def main():
    pass
