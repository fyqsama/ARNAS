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
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, input_search):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer, input_search)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, input_search):
    # 复制一个模型
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    # 分别计算每个目标对架构参数的损失
    #unrolled_loss_natural = unrolled_model._loss(input_search, target_valid)
    unrolled_loss_pgd = unrolled_model._loss(input_valid, target_valid)
   
    
 
    # 分别计算每个目标对架构参数的梯度
    #grads_natural = torch.autograd.grad(unrolled_loss_natural, unrolled_model.arch_parameters(), retain_graph=True)
    #grads_pgd = torch.autograd.grad(unrolled_loss_pgd, unrolled_model.arch_parameters(), retain_graph=True)
    #u1 = _concat(grads_natural)
    #u2 = _concat(grads_pgd)

    #计算gamma_star
    #gamma_star = self._compute_gamma_star(u1, u2)
    #hh = gamma_star * u1 + (1 - gamma_star) * u2
    #hh = torch.dot(hh, hh)
    
    
    #gamma_star.volatile = False
    # 计算Lval
    #unrolled_loss = gamma_star * unrolled_loss_natural + (1 - gamma_star) * unrolled_loss_pgd
    unrolled_loss = unrolled_loss_pgd
    # 反向传播
    unrolled_loss.backward()
    # 得到对架构参数的梯度
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    # 得到对模型参数的梯度
    vector = [v.grad.data for v in unrolled_model.parameters()]
    # 计算二阶近似
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    # 计算最终梯度
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    # 把计算出的梯度更新到原模型中
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

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
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  # Todo： decide gamma_star
  def _compute_gamma_star(self, u1, u2):
    u2_sub_u1 = u2 - u1
    result = torch.dot(u2_sub_u1, u2) / torch.dot(u2_sub_u1, u2_sub_u1)
    gamma_star = torch.clamp(result, 0.0, 1.0)

    return gamma_star