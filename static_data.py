from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from models.MPIS_SNNs import MPIS_static
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 SNN Training')

parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--path', default='D:/datasets/', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-t', '--time_step', default=1, type=int)
parser.add_argument('--vth', default=2., type=float)
parser.add_argument('--leaky', default=0.999, type=float)
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', default=500, type=int)
parser.add_argument('--drop', '--dropout', default=0.2, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[15,40,80,100,150,200],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='SGD', type=str, help='which optimizer')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

best_acc = 0  
current_iter = 0
def main():
    global best_acc
    start_epoch = args.start_epoch  

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print('==> Preparing dataset %s' % args.dataset)
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
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root=args.path, train=False, download=False, transform=transform_train)
    testset = dataloader(root=args.path, train=False, download=False, transform=transform_test)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print("==> creating model SNN_Conv")
    in_planes = 3
    config = {}
    config['MODEL'] = {}
    config['MODEL']['c_in'] = in_planes
    config['MODEL']['c_hidden'] = 128
    config['MODEL']['num_classes'] = num_classes
    config['MODEL']['kernel_size_x'] = 3
    config['MODEL']['stride_x'] = 2
    config['MODEL']['padding_x'] = 1
    config['MODEL']['pooling_x'] = False
    config['MODEL']['c_s1'] = 256
    config['MODEL']['c_s2'] = 512
    config['MODEL']['c_s3'] = 1024
    config['MODEL']['kernel_size_s'] = 3
    config['MODEL']['h_hidden'] = 4
    config['MODEL']['w_hidden'] = 4
    config['MODEL']['dropout'] = args.drop
    config['MODEL']['threshold'] = 30
    config['MODEL']['time_step'] = args.time_step
    config['MODEL']['vth'] = args.vth
    config['MODEL']['leaky'] = args.leaky
    config['OPTIM'] = {}
    config['OPTIM']['solver'] = 'broy'
    model = MPIS_static(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    
    title = 'cifar-' + 'SNN_Conv'
    if args.resume:
        
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        args.warmup = 0
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Train Top5', 'Valid Acc.', 'Valid Top5' ])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda, args.time_step, num_classes)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, train_top5 = train(trainloader, model, criterion, optimizer, epoch, use_cuda, args.time_step, num_classes, warmup=args.warmup)
        print("train:")
        print((train_loss, train_acc, train_top5))
        test_loss, test_acc, test_top5 = test(testloader, model, criterion, epoch, use_cuda, args.time_step, num_classes)
        print("test:")
        print((test_loss, test_acc, test_top5))
        logger.append([state['lr'], train_loss, test_loss, train_acc, train_top5, test_acc, test_top5])

        
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()

    print('Best acc:')
    print(best_acc)

    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, time_step, num_classes, warmup=0):
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    global current_iter

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

        if warmup != 0 and current_iter < warmup:
            adjust_warmup_lr(optimizer, current_iter, warmup)
            current_iter += 1
        
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        batch_time.update(time.time() - end)
        end = time.time()

        
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)


def test(testloader, model, criterion, epoch, use_cuda, time_step, num_classes):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        
        batch_time.update(time.time() - end)
        end = time.time()

        
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def adjust_warmup_lr(optimizer, citer, warmup):
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr'] * (citer + 1.) / warmup

if __name__ == '__main__':
    main()
