import argparse
import shutil

import collections

from torch import nn
from torch.backends import cudnn
import torch.optim
import torch.utils.data
from pathlib import Path
from torch.utils.data import DataLoader

from training.fp16util import (model_grads_to_master_grads,
                               network_to_half,
                               prep_param_lists,
                               master_params_to_model_params)

from training import resnet
import copy
import tqdm
import dataloader

from training import experimental_utils
from training.meter import AverageMeter
import albumentations as albu

lr = 1.0
batch_size = [1024, 224, 128]  # largest batch size that fits in memory for each image size

batch_size_scale = [x / batch_size[0] for x in batch_size]

phases = [
    {'epoch': 0,
     'size': 128,
     'batch_size': batch_size[0]},
    {'epoch': (0, 10),
     'size': 128,
     'lr': (lr, lr * 2),
     'batch_size': batch_size[0]},
    {'epoch': (10, 20),
     'size': 128,
     'lr': (lr * 2, lr / 4),
     'batch_size': batch_size[0]},
    {'epoch': 20,
     'size': 224,
     'batch_size': batch_size[1],
     'min_scale': 0.087},
    {'epoch': (20, 30),
     'size': 224,
     'lr': (lr * batch_size_scale[1], lr / 10 * batch_size_scale[1]),
     'batch_size': batch_size[1],
     'min_scale': 0.087},
    {'epoch': (30, 40),
     'size': 224,
     'lr': (lr / 10 * batch_size_scale[1], lr / 100 * batch_size_scale[1]),
     'batch_size': batch_size[1],
     'min_scale': 0.087},
    {'epoch': 40,
     'size': 288,
     'batch_size': batch_size[2],
     'min_scale': 0.5},
    {'epoch': (40, 50),
     'size': 288,
     'lr': (lr / 100 * batch_size_scale[2], lr / 1000 * batch_size_scale[2]),
     'batch_size': batch_size[2],
     'min_scale': 0.5}
]


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset', type=Path)

    parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')

    parser.add_argument('-j', '--num_workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--init-bn0', action='store_true', help='Intialize running batch norm mean to 0')

    parser.add_argument('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode. Default True')
    parser.add_argument('--loss-scale', type=float, default=1024,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--logdir', default='runs/debug', type=str, help='where logs go')
    parser.add_argument('--device-ids', type=str, default='0,1', help='For example 0,1 to run on two GPUs')
    parser.add_argument('--best_top5', type=float, default=93, help='min top5 to save checkpoint')
    return parser


cudnn.benchmark = True
args = get_parser().parse_args()

Path(args.logdir).mkdir(exist_ok=True, parents=True)


def main():
    model = resnet.resnet50(bn0=args.init_bn0).cuda()

    if args.device_ids:
        device_ids = list(map(int, args.device_ids.split(',')))
    else:
        device_ids = None

    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    if args.fp16:
        model = network_to_half(model)

    best_top5 = args.best_top5  # only save models over 93%. Otherwise it stops to save every time

    global model_params, master_params

    if args.fp16:
        model_params, master_params = prep_param_lists(model)
    else:
        model_params = master_params = model.parameters()

    if args.no_bn_wd:
        optim_params = experimental_utils.bnwd_optim_params(model, model_params, master_params)
    else:
        optim_params = master_params

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # start with 0 lr. Scheduler will change this later
    optimizer = torch.optim.SGD(optim_params, 0, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top5 = checkpoint['best_top5']
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = Scheduler(optimizer, [copy.deepcopy(p) for p in phases if 'lr' in p])

    for epoch in range(args.start_epoch, scheduler.tot_epochs):

        target_size = scheduler.get_current_size(epoch)
        min_scale = scheduler.get_current_min_scale(epoch)
        batch_size = scheduler.get_current_batch_size(epoch)

        train_transform = albu.Compose([albu.HorizontalFlip(p=0.5),
                                        albu.Resize(height=target_size, width=target_size, p=1),
                                        albu.Normalize(p=1)])

        val_transform = albu.Compose([albu.Resize(height=int(target_size * 1.14), width=int(target_size * 1.14), p=1),
                                      albu.CenterCrop(height=target_size, width=target_size, p=1),
                                      albu.Normalize(p=1)])

        train_data_loader = DataLoader(dataset=dataloader.DatasetGenerator(args.data / 'train',
                                                                           transform=train_transform,
                                                                           mode='train',
                                                                           min_scale=min_scale),
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       batch_size=batch_size,
                                       pin_memory=torch.cuda.is_available())

        val_data_loader = DataLoader(dataset=dataloader.DatasetGenerator(args.data / 'validation',
                                                                         transform=val_transform,
                                                                         mode='val'),
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     batch_size=batch_size,
                                     pin_memory=torch.cuda.is_available())

        train(train_data_loader, model, criterion, optimizer, scheduler, epoch)

        top1, top5 = validate(val_data_loader, model, criterion)

        print(f'top1 = {top1}, top5 = {top5}')

        is_best = top5 > best_top5
        best_top5 = max(top5, best_top5)

        if is_best:
            save_checkpoint(epoch, model, best_top5, optimizer, is_best=True, filename='model_best.pth.tar')
        phase = scheduler.get_current_phase(epoch)
        if phase:
            save_checkpoint(epoch, model, best_top5, optimizer, filename=f'sz{phase["size"]}_checkpoint.path.tar')


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    batch_size = scheduler.get_current_batch_size(epoch)

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description(f'Epoch {epoch}')

    for i, (inputs, targets) in enumerate(train_loader):
        scheduler.update_lr(epoch, i + 1, len(train_loader))

        inputs = inputs.cuda()

        with torch.no_grad():
            targets = targets.cuda()

        output = model(inputs)
        loss = criterion(output, targets)

        # compute gradient and do SGD step
        if args.fp16:
            loss = loss * args.loss_scale
            model.zero_grad()
            loss.backward()
            model_grads_to_master_grads(model_params, master_params)
            for param in master_params:
                param.grad.data = param.grad.data / args.loss_scale

            optimizer.step()
            master_params_to_model_params(model_params, master_params)
            loss = loss / args.loss_scale
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Train batch done. Logging results
        corr1, corr5 = correct(output.data, targets, topk=(1, 5))

        reduced_loss = to_python_float(loss.data)
        batch_total = to_python_float(batch_size)

        top1acc = to_python_float(corr1) * (100.0 / batch_total)
        top5acc = to_python_float(corr5) * (100.0 / batch_total)

        losses.update(reduced_loss, batch_total)
        top1.update(top1acc, batch_total)
        top5.update(top5acc, batch_total)

        tq.update(batch_size)

        lr = scheduler.get_lr(epoch, i + 1, len(train_loader))
        output_string = f'{losses.average:.4f} Acc@1 = {top1.average:.3f} Acc@5 = {top5.average:.3f} lr = {lr:.3f}'

        tq.set_postfix(loss=output_string)

    tq.close()


def validate(val_loader, model, criterion):
    with torch.no_grad():
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()

        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets).data

            batch_size = inputs.size(0)
            top1acc, top5acc = accuracy(output.data, targets, topk=(1, 5))

            # Eval batch done. Logging results
            losses.update(to_python_float(loss), to_python_float(batch_size))
            top1.update(to_python_float(top1acc), to_python_float(batch_size))
            top5.update(to_python_float(top5acc), to_python_float(batch_size))

    return top1.average, top5.average


# ### Learning rate scheduler
class Scheduler:
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer
        self.current_lr = None
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['epoch']) for p in self.phases])

    def format_phase(self, phase):
        phase['epoch'] = listify(phase['epoch'])
        phase['lr'] = listify(phase['lr'])
        if len(phase['lr']) == 2:
            assert (len(phase['epoch']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def linear_phase_lr(self, phase, epoch, batch_curr, batch_tot):
        lr_start, lr_end = phase['lr']
        ep_start, ep_end = phase['epoch']
        if 'epoch_step' in phase:
            batch_curr = 0  # Optionally change learning rate through epoch step
        ep_relative = epoch - ep_start
        ep_tot = ep_end - ep_start
        return self.calc_linear_lr(lr_start, lr_end, ep_relative, batch_curr, ep_tot, batch_tot)

    def calc_linear_lr(self, lr_start, lr_end, epoch_curr, batch_curr, epoch_tot, batch_tot):
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + batch_curr
        step_size = (lr_end - lr_start) / step_tot
        return lr_start + step_curr * step_size

    def get_current_phase(self, epoch):
        for phase in reversed(self.phases):
            if epoch >= phase['epoch'][0]:
                return phase
        raise Exception('Epoch out of range')

    def get_current_size(self, epoch):
        current_phase = self.get_current_phase(epoch)
        return current_phase['size']

    def get_current_batch_size(self, epoch):
        current_phase = self.get_current_phase(epoch)
        return current_phase['batch_size']

    def get_current_min_scale(self, epoch):
        current_phase = self.get_current_phase(epoch)
        return current_phase.get('min_scale', 0.08)

    def get_lr(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase['lr']) == 1:
            return phase['lr'][0]  # constant learning rate
        return self.linear_phase_lr(phase, epoch, batch_curr, batch_tot)

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if self.current_lr == lr:
            return

        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    return t.item()


def save_checkpoint(epoch, model, best_top5, optimizer, is_best=False, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_top5': best_top5,
             'optimizer': optimizer.state_dict(), }
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{args.logdir}/{filename}')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k."""
    corrrect_ks = correct(output, target, topk)
    batch_size = target.size(0)
    return [correct_k.float().mul_(100.0 / batch_size) for correct_k in corrrect_ks]


def correct(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k."""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        res.append(correct_k)
    return res


def listify(p=None, q=None):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    n = q if type(q) == int else 1 if q is None else len(q)
    if len(p) == 1:
        p = p * n
    return p


if __name__ == '__main__':
    main()
