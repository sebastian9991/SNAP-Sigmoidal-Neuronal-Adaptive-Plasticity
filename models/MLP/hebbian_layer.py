import os
from typing import Tuple

import matplotlib.pyplot as plt
import models.utils.helper_modules as M
import models.utils.learning as L
import numpy as np
import torch
import torch.nn as nn
from dotwiz import DotWiz
from models.hyperparams import (Inhibition, InputProcessing, LearningRule,
                                WeightGrowth)
from utils.experiment_constants import Focus


class SoftHebbLayer(nn.Module):
    def __init__(
        self,
        K: float,
        focus: Focus,
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

        print(f"OUTPUT DIM is {outputdim}")
        self.K = K
        self.focus = focus
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
        delta_w, self.wn = L.update_softhebb_w(
            self.K,
            self.focus,
            inference_output.y,
            inference_output.xn,
            inference_output.a,
            self.weight,
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

    def plot_wn_distribution(self, epoch, count):
        """
        Plot the weight norm distribution stored in self.wn.
        Call this method at the end of an epoch.
        """
        if self.wn is None:
            print("No weight norms recorded for this epoch.")
            return

        # Convert self.wn to a numpy array
        wn_np = self.wn.cpu().numpy().flatten()
        # Remove nan-values (matplot plt.() can't deal with Nan indexes)
        wn_np = wn_np[~np.isnan(wn_np)]

        # Create a folder for the plots
        plot_folder = os.path.join(
            os.getcwd(), f"Focus_{self.focus}_with_K_{self.K}_wn_plots"
        )
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        # Determine layer type for the filename
        layer_type = "output" if self.is_output_layer else "hidden"
        plot_filename = os.path.join(
            plot_folder, f"Task_{count}_wn_distribution_{layer_type}_epoch_{epoch}.png"
        )

        plt.figure()
        plt.hist(wn_np, bins=50, alpha=0.75)
        plt.title(
            f"Weight Norm Distribution ({layer_type.capitalize()} Layer) - Epoch {epoch} - K = {self.K}"
        )
        plt.xlabel("Weight Norm")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved weight norm plot: {plot_filename}")
