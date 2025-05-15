import os
from typing import Tuple

import torch
import torch.nn as nn
from dotwiz import DotWiz

import models.utils.helper_modules as M
import models.utils.learning as L
from models.utils.hyperparams import (Inhibition, InputProcessing,
                                      LearningRule, WeightGrowth)
from utils.experiment_utils.experiment_constants import Focus


class SoftHebbLayer(nn.Module):
    def __init__(
        self,
        K: float,
        focus: Focus,
        epsilon: float,
        inputdim: int,
        outputdim: int,
        w_lr: float = 0.003,
        b_lr: float = 0.003,
        l_lr: float = 0.003,
        device=None,
        is_output_layer=False,
        initial_weight_norm: float = 0.01,
        triangle: bool = False,
        initial_lambda: float = 4.0,
        inhibition: Inhibition = Inhibition.RePU,
        learningrule: LearningRule = LearningRule.SoftHebb,
        preprocessing: InputProcessing = InputProcessing.No,
        weight_growth: WeightGrowth = WeightGrowth.Default,
    ):
        super(SoftHebbLayer, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.input_dim: int = inputdim
        self.output_dim: int = outputdim

        self.K = K
        self.focus = focus
        self.epsilon = epsilon
        self.triangle: bool = triangle
        self.w_lr: float = w_lr
        self.l_lr: float = l_lr
        self.b_lr: float = b_lr
        self.lamb = nn.Parameter(
            torch.tensor(initial_lambda, device=device), requires_grad=False
        )
        self.is_output_layer: bool = is_output_layer
        self.learningrule: LearningRule = learningrule
        assert learningrule in [
            LearningRule.SoftHebb,
            LearningRule.SoftHebbOutputContrastive,
        ]
        self.inhibition: Inhibition = inhibition
        self.weight_growth: WeightGrowth = weight_growth
        self.preprocessing: InputProcessing = preprocessing
        if preprocessing == InputProcessing.Whiten:
            self.bn = M.BatchNorm(inputdim, device=device)

        self.weight = nn.Parameter(
            torch.randn((outputdim, inputdim), device=device), requires_grad=False
        )
        self.weighted_sum = self.weight
        self.logprior = nn.Parameter(
            torch.zeros(outputdim, device=device), requires_grad=False
        )

        self.initial_weight_norm = initial_weight_norm
        self.set_weight_norms_to(initial_weight_norm)

    def set_weight_norms_to(self, norm: float):
        weights = self.weight
        weights_norm = self.get_weight_norms(weights)
        new_weights = norm * weights / (weights_norm + 1e-10)
        self.weight = nn.Parameter(new_weights, requires_grad=False)

    def get_weight_norms(self, weights):
        return torch.norm(weights, p=2, dim=1, keepdim=True)

    def get_f_target(self):
        min_val = self.weighted_sum.min()
        max_val = self.weighted_sum.max()
        target = (self.weighted_sum - min_val) / (max_val - min_val + 1e-8)
        target = target.mean()
        return max(0.66, target.mean())

    def a(self, x):
        # batch_size, dim = x.shape
        # it is expected that x has L2 norm of 1.
        weight_norms = self.get_weight_norms(self.weight)
        W = self.weight / (weight_norms + 1e-9)
        cos_sims = torch.matmul(x, W.T)
        return cos_sims

    def u(self, a):
        # batch_size, dim = a.shape
        if self.inhibition == Inhibition.RePU:
            if self.triangle:
                setpoint = a.mean()
            else:
                setpoint = 0
            u = torch.relu(a - setpoint)
        elif self.inhibition == Inhibition.Softmax:
            u = torch.exp(a)
        else:
            raise NotImplementedError(
                f"{self.inhibition} is not an implemented inhibition method."
            )
        return u

    def y(self, a):
        if self.inhibition == Inhibition.Softmax:
            y = torch.softmax(self.lamb * a + self.logprior)
        elif self.inhibition == Inhibition.RePU:
            u = self.u(a)
            un = u / (torch.max(u) + 1e-9)  # normalize for numerical stability
            ulamb = un**self.lamb * torch.exp(self.logprior)
            y = ulamb / (torch.sum(ulamb, dim=1, keepdim=True) + 1e-9)
        else:
            raise NotImplementedError(
                f"{self.inhibition} is not an implemented inhibition method."
            )
        return y

    def inference(self, x):
        x_norms = torch.norm(x, dim=1, keepdim=True)
        x_n = x / (x_norms + 1e-9)
        a = self.a(x_n)
        u = self.u(a)
        y = self.y(a)
        return DotWiz(xn=x_n, a=a, u=u, y=y)

    def learn_weights(self, inference_output, target=None):
        supervised = self.learningrule == LearningRule.SoftHebbOutputContrastive
        f_target = self.get_f_target()
        delta_w, self.K = L.update_softhebb_w(
            self.K,
            self.focus,
            inference_output.y,
            inference_output.xn,
            inference_output.a,
            self.weight,
            f_target,
            self.epsilon,
            self.inhibition,
            inference_output.u,
            target=target,
            supervised=supervised,
            weight_growth=self.weight_growth,
        )
        delta_b = L.update_softhebb_b(
            inference_output.y, self.logprior, target=target, supervised=supervised
        )
        delta_l = L.update_softhebb_lamb(
            inference_output.y,
            inference_output.a,
            inhibition=self.inhibition,
            lamb=self.lamb.item(),
            in_dim=self.input_dim,
            target=target,
            supervised=supervised,
        )
        new_weight = self.weight + self.w_lr * delta_w
        self.weighted_sum += new_weight
        self.weight.data = new_weight

        new_bias = self.logprior + self.b_lr * delta_b
        # normalize bias to be proper prior, i.e. sum exp Prior = 1
        norm_cste = torch.log(torch.exp(new_bias).sum())
        self.logprior.data = new_bias - norm_cste

        new_lambda = self.lamb + self.l_lr * delta_l
        self.lamb.data = new_lambda

    def forward(self, x, target=None):
        inference_output = self.inference(x)
        if self.training:
            self.learn_weights(inference_output, target=target)
        return inference_output.y


class SoftNeuralNet(nn.Module):
    def __init__(self, device, hsize):
        super(SoftNeuralNet, self).__init__()
        self.layers = nn.ModuleDict()
        self.iteration = 3
        self.output_dim = 10
        self.device = device
        self.hsize = hsize

    def add_layer(self, name, layer):
        self.layers[name] = layer

    def forward(self, x, clamped=None):
        for layer in self.layers.values():
            x = layer.forward(x, target=clamped)
        return x

    def forward_test(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def set_iteration(self, i):
        self.iteration = i

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def set_training_layers(self, layers_to_train):
        for layer in self.layers.values():
            if layer in layers_to_train:
                layer.train()
            else:
                layer.eval()
