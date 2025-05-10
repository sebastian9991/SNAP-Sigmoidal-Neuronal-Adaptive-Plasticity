import sys

import scipy
import torch

from models.utils.hyperparams import Inhibition
from utils.experiment_utils.experiment_constants import Focus, WeightGrowth


def update_weight_softhebb(
    input, preactivation, output, weight, target=None, inhibition=Inhibition.RePU
):
    # input_shape = batch, in_dim
    # output_shape = batch, out_dim = preactivation_shape
    # weight_shape = out_dim, in_dim
    b, indim = input.shape
    b, outdim = output.shape
    multiplicative_factor = 1
    W = weight
    if inhibition == Inhibition.RePU:
        u = torch.relu(preactivation)
        multiplicative_factor = multiplicative_factor / (u + 1e-9)
    elif inhibition == Inhibition.Softmax:
        u = preactivation
    # deltas = multiplicative_factor * output * (input - torch.matmul(torch.relu(u), W).reshape(b, indim))
    deltas = (multiplicative_factor * output).reshape(b, outdim, 1) * (
        input - torch.matmul(torch.relu(u), W)
    ).reshape(b, 1, indim)
    delta = torch.mean(deltas, dim=0)
    return delta


def softhebb_input_difference(x, a, normalized_weights):
    # Here x is assumed to have an L2 norm of 1
    # same for normalized_weights[i].
    # output has shape: batch, out_dim, in_dim
    batch_dim, in_dim = x.shape
    batch_dim, out_dim = a.shape
    in_space_diff = x.reshape(batch_dim, 1, in_dim) - a.reshape(
        batch_dim, out_dim, 1
    ) * normalized_weights.reshape(1, out_dim, in_dim)
    return in_space_diff


def update_softhebb_w(
    K,
    focus,
    y,
    normed_x,
    a,
    weights,
    inhibition: Inhibition,
    u=None,
    target=None,
    supervised=False,
    weight_growth: WeightGrowth = WeightGrowth.DEFAULT,
):
    if focus == Focus.NEURON:
        weight_norms = torch.norm(weights, dim=1, keepdim=True)
        normed_weights = weights / (weight_norms + 1e-9)
        print("Shape of y:", y.shape)
        sys.stdout.flush()
        batch_dim, out_dim = y.shape
        wn = weight_norms.unsqueeze(0)
        factor = 1 / (wn + 1e-9)
        if weight_growth == WeightGrowth.LINEAR:
            factor *= 1
        elif weight_growth == WeightGrowth.SIGMOID:
            factor *= wn / K * (1 - wn / K)
        elif weight_growth == WeightGrowth.EXPONENTIAL:
            factor *= wn / K
        else:
            raise NotImplementedError(f"Weight growth {weight_growth}, invalid.")
        if inhibition == Inhibition.RePU:
            indicator = (a > 0).float()
            factor = (
                factor
                * indicator.reshape(batch_dim, out_dim, 1)
                / (a.reshape(batch_dim, out_dim, 1) + 1e-9)
            )
        if supervised:
            y_part = (target - y).reshape(batch_dim, out_dim, 1)
        else:
            y_part = y.reshape(batch_dim, out_dim, 1)

        delta_w = (
            factor * y_part * softhebb_input_difference(normed_x, a, normed_weights)
        )
        delta_w = torch.mean(
            delta_w, dim=0
        )  # average the delta weights over the batch dim
    elif focus == Focus.SYNAPSE:
        batch_dim, out_dim = y.shape
        print("Shape of y:", y.shape)
        w = torch.abs(weights)  # Element-wise absoluate value for |Wij|
        weight_norms = torch.norm(weights, dim=1, keepdim=True)
        wn = weight_norms.unsqueeze(0)  # Keeping this to make return consistent
        factor = 1 / (w / K + 1e-9)
        if weight_growth == WeightGrowth.LINEAR:
            factor *= torch.ones_like(weights)
        elif weight_growth == WeightGrowth.SIGMOID:
            factor *= w / K * (1 - w / K)
        elif weight_growth == WeightGrowth.EXPONENTIAL:
            factor *= w / K
        else:
            raise NotImplementedError(f"Weight growth {weight_growth}, invalid.")

        if inhibition == Inhibition.RePU:
            indicator = (a > 0).float()
            factor = (
                factor
                * indicator.reshape(batch_dim, out_dim, 1)
                / (a.reshape(batch_dim, out_dim, 1) + 1e-9)
            )
        if supervised:
            y_part = (target - y).reshape(batch_dim, out_dim, 1)
        else:
            y_part = y.reshape(batch_dim, out_dim, 1)
        delta_w = factor * y_part * softhebb_input_difference(normed_x, a, weights)
        delta_w = torch.mean(
            delta_w, dim=0
        )  # Average the delta weights over the batch dim
    else:
        raise NotImplementedError(f"Focus paramater {focus} is not valid.")

    return delta_w, wn / K


def update_softhebb_b(y, logprior, target=None, supervised=False):
    priors = torch.softmax(logprior, dim=0).unsqueeze(0)
    if supervised:
        delta_b = target - priors
    else:
        delta_b = y - priors
    delta_b = torch.mean(delta_b, dim=0)  # mean over batch dim

    return delta_b


def update_softhebb_lamb(
    y, a, inhibition: Inhibition, lamb=None, in_dim=None, target=None, supervised=False
):
    if inhibition == Inhibition.Softmax:
        v = a
    elif inhibition == Inhibition.RePU:
        u = torch.relu(a)
        v = (u > 0).float() * torch.log(u + 1e-7)
    else:
        raise NotImplementedError(
            f"{inhibition} not implemented type of inhibition in update λ."
        )

    if supervised:
        delta_l = torch.sum((target - y) * v, dim=1)
    else:
        if inhibition == Inhibition.Softmax:
            k = scipy.special.iv(in_dim / 2, lamb) / scipy.special.iv(
                in_dim / 2 - 1, lamb
            )
        elif inhibition == Inhibition.RePU:
            k = 0.5 * (
                scipy.special.psi(0.5 * (lamb + 1))
                - scipy.special.psi(0.5 * (lamb + in_dim))
            )
        else:
            raise NotImplementedError(
                f"{inhibition} not implemented type of inhibition in update λ."
            )
        delta_l = torch.sum(y * v, dim=1) - k
    delta_l = torch.mean(delta_l)  # mean over batch dim
    return delta_l
