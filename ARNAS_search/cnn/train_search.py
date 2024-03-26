import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from torch.autograd import Variable
from model_search import Network
from architect_2 import Architect

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=24, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--regularization_coefficient', type=float, default=0.1, help='regularization_coefficient of adversarial loss')
args = parser.parse_args()




def main():
  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    print(architect.optimizer.param_groups[0]['lr'])
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_accurate, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))
    print(F.softmax(model.alphas_accurate, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    #valid_acc, valid_acc_pgd = infer(valid_queue, model, criterion)
    #logging.info('valid_acc_natural: %f, valid_acc_pgd: %f', valid_acc, valid_acc_pgd)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    '''
    p = input
    y = input[0]
    yy = y.data.cpu().numpy()
    yy=(yy+2.0)/4
    yy = yy.swapaxes(0, 1)
    yy = yy.swapaxes(1, 2)
    plt.imshow(yy, vmin=-1.0, vmax=1.0)
    plt.savefig('text.png')
    plt.show()
    '''
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)
    input_pgd = projected_gradient_descent(model, input, 8/255, 2/255, 7, np.inf, y=target)
    input_pgd = Variable(input_pgd.data, requires_grad=False)

    '''
    t = input_pgd[0]
    tt = t.data.cpu().numpy()
    tt = (tt + 2.0) / 4
    tt = tt.swapaxes(0, 1)
    tt = tt.swapaxes(1, 2)
    plt.imshow(tt, vmin=-1.0, vmax=1.0)
    plt.savefig('text.png')
    plt.show()
    '''
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)
    input_search = Variable(input_search, requires_grad=False).cuda()
    input_search_pgd = projected_gradient_descent(model, input_search, 8/255, 2/255, 7, np.inf, y=target_search)
    input_search_pgd = Variable(input_search_pgd.data, requires_grad=False)

    

    architect.step(input_pgd, target, input_search_pgd, target_search, lr, optimizer, args.unrolled, input_search, epoch, args.regularization_coefficient)

    optimizer.zero_grad()
    logits = model(input_pgd)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  objs_pgd = utils.AvgrageMeter()
  top1_pgd = utils.AvgrageMeter()
  top5_pgd = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    input_pgd = projected_gradient_descent(model, input, 8/255, 2/510, 7, np.inf, y=target)
    target = Variable(target, volatile=True).cuda(non_blocking=True)

    logits = model(input)
    logits_pgd = model(input_pgd)
    loss = criterion(logits, target)
    loss_pgd = criterion(logits_pgd, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    prec1_pgd, prec5_pgd = utils.accuracy(logits_pgd, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)
    objs_pgd.update(loss_pgd.data.item(), n)
    top1_pgd.update(prec1_pgd.data.item(), n)
    top5_pgd.update(prec5_pgd.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d natural:%e %f %f, pgd:%e %f %f', step, objs.avg, top1.avg, top5.avg, objs_pgd.avg, top1_pgd.avg, top5_pgd.avg)

  return top1.avg, top1_pgd.avg


if __name__ == '__main__':
  main() 

