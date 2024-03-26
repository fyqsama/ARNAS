import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import genotypes

from torch.autograd import Variable
from model import NetworkCIFAR_new as Network
from trades import trades_loss, madry_loss


parser = argparse.ArgumentParser("tinyimagenet")
parser.add_argument('--gpus', type=list, default=[0, 1, 2], help='gpu device id')
parser.add_argument('--batch_size', type=int, default=26, help='batch size')
parser.add_argument('--epochs', type=int, default=90, help='num of training epochs')
parser.add_argument('--adv_loss', type=str, default='pgd', help='experiment name')
parser.add_argument('--data', type=str, default='/home/ouyuwei/tiny-imagenet-200/', help='location of the data corpus')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation')
parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
parser.add_argument('--step_size', type=float, default=0.01, help='perturb step size')
parser.add_argument('--beta', type=float, default=6.0, help='regularization in TRADES')
parser.add_argument('--init_channels', type=int, default=84, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


CLASSES = 200


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('args: %s' % args)

    genotype = genotypes.ARNAS
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model = nn.DataParallel(model, device_ids=args.gpus).cuda()

    print("param size: ", utils.count_parameters_in_MB(model), 'MB')

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_imagenet()
    train_data = dset.ImageFolder(root=args.data + 'train', transform=train_transform)
    valid_data = dset.ImageFolder(root=args.data + 'val', transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    best_acc = 0.0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('epoch: %d' % epoch)

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        print('train_acc: ', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)

        if valid_acc > best_acc:
            best_acc = valid_acc
        save_location =  '/home/ouyuwei/fyq_advrush_tiny_imagenet/advrush/exp/model_adv_' + str(epoch) + '.pt'
        utils.save(model, save_location)
        print('save model! ')

        print('valid acc: ', valid_acc, 'best_acc: ', best_acc)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda(non_blocking=True)
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        if args.adv_loss == 'pgd':
            loss = madry_loss(
                model,
                input,
                target,
                optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps)
        elif args.adv_loss == 'trades':
            loss = trades_loss(model,
                input,
                target,
                optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                distance='l_inf')

        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, requires_grad=False).cuda(non_blocking=True)
            target = Variable(target, requires_grad=False).cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate
    if epoch >= 30:
        lr = args.learning_rate * 0.1
    if epoch >= 60:
        lr = args.learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()