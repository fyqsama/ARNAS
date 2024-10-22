from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from model import NetworkCIFAR_new as Network
from auto_attack.autoattack import AutoAttack
import genotypes
os.environ['CUDA_VISIBLE_DEVICES']='1'

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=1,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=8/255,
                    help='perturb step size')
parser.add_argument('--random',
                    default=False,
                    help='random initialization for PGD')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--arch', type=str, default='ARNAS', help='which architecture to use')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--target_arch', type=str, default='ADVRUSH', help='which architecture to use')
parser.add_argument('--source_arch', type=str, default='ADVRUSH', help='which architecture to use')
parser.add_argument('--target_checkpoint', type=str, default='./', help='which architecture to use')
parser.add_argument('--source_checkpoint', type=str, default='./', help='which architecture to use')
parser.add_argument('--log_path', type=str, default='./', help='path to store log file')
parser.add_argument('--checkpoint', type=str, default='EXP/ARNAS.pth.tar', help='which architecture to use')
parser.add_argument('--data_type', type=str, default='cifar10', help='which dataset to use')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# set up data loader
if args.data_type == 'cifar10':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.data_type == 'cifar100':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
elif args.data_type == 'svhn':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out, _ = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd)[0], y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    err_pgd = (model(X_pgd)[0].data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for step, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        if step % 50 == 0:
            print('natural_err: ', natural_err_total)
            print('robust_er: ', robust_err_total)
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def eval_adv_test_whitebox_AA(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    for param in model.parameters():
        param.requires_grad = False
    x_test = [x for (x, y) in test_loader]
    x_test = torch.cat(x_test, dim=0)
    y_test = [y for (x, y) in test_loader]
    y_test = torch.cat(y_test, dim=0)
    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, verbose=True)
    adversary.plus = False

    print('*************** AA Attack Eval *************')
    x_adv, robust_accuracy = adversary.run_standard_evaluation(x_test, y_test, bs=args.test_batch_size)
    robust_accuracy = robust_accuracy * 100

def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)

def main():
    

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        if args.data_type == 'cifar100':
            CIFAR_CLASSES = 100
        else:
            CIFAR_CLASSES = 10
        genotype = eval("genotypes.%s" % args.arch)
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        model.drop_path_prob = args.drop_path_prob
        model.cuda()
        eval_adv_test_whitebox_AA(model, device, test_loader)

    else:
        # black-box attack
        CIFAR_CLASSES = 10
        print('pgd black-box attack')
        target_genotype = eval("genotypes.%s" % args.target_arch)
        source_genotype = eval("genotypes.%s" % args.source_arch)
        
        model_source = Network(args.init_channels,CIFAR_CLASSES, args.layers, args.auxiliary, source_genotype)
        source_checkpoint = torch.load(args.source_checkpoint)
        model_source.load_state_dict(source_checkpoint['state_dict'])
        model_source.drop_path_prob = args.drop_path_prob
        model_source.cuda()

        model_target = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, target_genotype)
        target_checkpoint = torch.load(args.target_checkpoint)
        model_target.load_state_dict(target_checkpoint['state_dict'])
        model_target.drop_path_prob = args.drop_path_prob
        model_target.cuda()

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
