import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import logging


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(moment+dtheta,  alpha=eta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, input_search, epoch, regularization_coefficient):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer, input_search, epoch, regularization_coefficient)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, input_search, epoch, regularization_coefficient):
    # epoch < 20 不进行多目标优化
    if False:
    #if epoch <= 20:
      unrolled_model_pgd = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
      unrolled_loss_pgd = unrolled_model_pgd._loss(input_valid, target_valid)
      unrolled_loss_pgd.backward()
      # 得到对架构参数的梯度
      dalpha_pgd = [v.grad for v in unrolled_model_pgd.arch_parameters()]
      # 得到对模型参数的梯度
      vector_pgd = [v.grad.data for v in unrolled_model_pgd.parameters()]
      # 计算二阶近似
      implicit_grads_pgd = self._hessian_vector_product(vector_pgd, input_train, target_train)

      # 计算最终梯度
      for g, ig in zip(dalpha_pgd, implicit_grads_pgd):
        g.data.sub_(ig.data, alpha=eta)

      # 把计算出的梯度更新到原模型中
      for v, g in zip(self.model.arch_parameters(), dalpha_pgd):
        if v.grad is None:
          v.grad = Variable(g.data)
        else:
          v.grad.data.copy_(g.data)
    # 否则进行多目标优化
    else:
      # 先计算natural的梯度
      # 复制一个模型
      unrolled_model_natural = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
      # 分别计算每个目标对架构参数的损失
      unrolled_loss_natural = unrolled_model_natural._loss(input_search, target_valid)
      unrolled_loss_natural.backward()
      # 得到对架构参数的梯度
      dalpha_natural = [v.grad for v in unrolled_model_natural.arch_parameters()]
      # 得到对模型参数的梯度
      vector_natural = [v.grad.data for v in unrolled_model_natural.parameters()]
      # 计算二阶近似
      implicit_grads_natural = self._hessian_vector_product(vector_natural, input_train, target_train)
      # 计算最终梯度
      for g, ig in zip(dalpha_natural, implicit_grads_natural):
        g.data.sub_(ig.data, alpha=eta)

      # 再计算pgd的梯度
      # 复制一个模型
      unrolled_model_pgd = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
      # 分别计算每个目标对架构参数的损失
      unrolled_loss_pgd = regularization_coefficient * unrolled_model_pgd._loss(input_valid, target_valid)
      unrolled_loss_pgd.backward()
      # 得到对架构参数的梯度
      dalpha_pgd = [v.grad for v in unrolled_model_pgd.arch_parameters()]
      # 得到对模型参数的梯度
      vector_pgd = [v.grad.data for v in unrolled_model_pgd.parameters()]
      # 计算二阶近似
      implicit_grads_pgd = self._hessian_vector_product(vector_pgd, input_train, target_train)
      # 计算最终梯度
      for g, ig in zip(dalpha_pgd, implicit_grads_pgd):
        g.data.sub_(ig.data, alpha=eta)

      u1 = _concat(dalpha_natural)
      u2 = _concat(dalpha_pgd)
      #计算gamma_star
      gamma_star = self._compute_gamma_star(u1, u2)
      gamma_star.requires_grad = False
      # 把计算出的梯度更新到原模型中
      # multi-objective adversarial training
      
      for v, gn, gp in zip(self.model.arch_parameters(), dalpha_natural, dalpha_pgd):
        if v.grad is None:
          v.grad = Variable((gamma_star * gn + (1 - gamma_star) * gp).data)
        else:
          v.grad.data.copy_((gamma_star * gn + (1 - gamma_star) * gp).data)
      
      # 把计算出的梯度更新到原模型中
      # sum of natural loss and adversarial loss
      '''
      for v, gn, gp in zip(self.model.arch_parameters(), dalpha_natural, dalpha_pgd):
        if v.grad is None:
          v.grad = Variable((gn + gp).data)
        else:
          v.grad.data.copy_((gn + gp).data)
      '''

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(v, alpha=R)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(v, alpha=2*R)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(v, alpha=R)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  # Todo： decide gamma_star
  def _compute_gamma_star(self, u1, u2):
    u2_sub_u1 = u2 - u1
    result = torch.dot(u2_sub_u1, u2) / torch.dot(u2_sub_u1, u2_sub_u1)
    gamma_star = torch.clamp(result, 0.0, 1.0)

    return gamma_star