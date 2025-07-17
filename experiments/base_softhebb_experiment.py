import math
import os
import time
from typing import Tuple, Type, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from interfaces.experiment import Experiment
from interfaces.network import Network
from layers.base.data_setup_layer import DataSetupLayer
from layers.input_layer import InputLayer
from utils.experiment_utils.experiment_constants import (DataSets,
                                                         ExperimentPhases,
                                                         Purposes)
from utils.experiment_utils.experiment_logger import *
from utils.experiment_utils.experiment_parser import *
from utils.experiment_utils.experiment_timer import *
from utils.path.path import get_root_dir
from utils.plotting.plot_accuracy_list import *
from utils.plotting.plot_weights import plot_weight_grid, plot_weight_heatmap

# def set_global_seed(seed: int = 42):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def plot_misclassified_examples_activations_logits(
    model, data_loader, device, epoch, save_dir, max_samples=10
):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get raw activations (a) and predictions
            inference_out = model.layers["SoftHebbian1"].inference(inputs)
            a = inference_out.a
            lamb = model.layers["SoftHebbian1"].lamb
            logprior = model.layers["SoftHebbian1"].logprior.to(inputs.device)

            logits = (lamb * a + logprior)

            y_pred = model.layers["SoftHebbian2"](inference_out.y)
            pred_labels = y_pred.argmax(dim=1)

            for i in range(inputs.size(0)):
                if pred_labels[i] != labels[i]:
                    misclassified.append(
                        (
                            inputs[i].cpu(),
                            pred_labels[i].item(),
                            labels[i].item(),
                            a[i].cpu(),         # cosine similarity
                            logits[i].cpu(),    # logits for softmax
                        )
                    )
                if len(misclassified) >= max_samples:
                    break
            if len(misclassified) >= max_samples:
                break

    # Prepare save paths
    img_save_dir = os.path.join(save_dir, "images")
    act_save_dir = os.path.join(save_dir, "activations")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(act_save_dir, exist_ok=True)

    # --- Plot 1: Input images with labels ---
    fig1, axs1 = plt.subplots(1, max_samples, figsize=(2 * max_samples, 2.5))
    axs1 = axs1.flatten()

    for i, (x, y_pred, y_true, _, _) in enumerate(misclassified):
        x_img = x.view(28, 28) if x.numel() == 784 else x.squeeze()
        axs1[i].imshow(x_img, cmap="gray")
        axs1[i].set_title(f"{y_pred} / {y_true}", fontsize=9)
        axs1[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{img_save_dir}/epoch_{epoch}_misclassified_inputs.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Cosine similarity activations (a) ---
    fig2, axs2 = plt.subplots(1, max_samples, figsize=(2.5 * max_samples, 2.5))
    axs2 = axs2.flatten()

    for i, (_, _, _, a_val, _) in enumerate(misclassified):
        a_np = a_val.numpy()
        h_len = len(a_np)
        side = int(math.sqrt(h_len))

        if side * side != h_len:
            padded = np.zeros((side + 1) ** 2)
            padded[:h_len] = a_np
            a_np = padded
            side += 1

        grid = a_np.reshape(side, side)
        norm_grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)

        im = axs2[i].imshow(norm_grid, cmap="viridis")
        axs2[i].set_title(f"Sample {i+1}", fontsize=7)
        axs2[i].axis("off")

        cbar = fig2.colorbar(im, ax=axs2[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(f"{act_save_dir}/epoch_{epoch}_activations_cosine_a.png", dpi=200)
    plt.close(fig2)

    # --- Plot 3: Logits heatmaps ---
    fig3, axs3 = plt.subplots(1, max_samples, figsize=(2.5 * max_samples, 2.5))
    axs3 = axs3.flatten()

    for i, (_, _, _, _, logits_val) in enumerate(misclassified):
        logit_np = logits_val.numpy()
        h_len = len(logit_np)
        side = int(math.sqrt(h_len))

        if side * side != h_len:
            padded = np.zeros((side + 1) ** 2)
            padded[:h_len] = logit_np
            logit_np = padded
            side += 1

        grid = logit_np.reshape(side, side)
        norm_grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)

        im = axs3[i].imshow(norm_grid, cmap="plasma")
        axs3[i].set_title(f"Sample {i+1}", fontsize=7)
        axs3[i].axis("off")

        cbar = fig3.colorbar(im, ax=axs3[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(f"{act_save_dir}/epoch_{epoch}_logits_heatmaps.png", dpi=200)
    plt.close(fig3)

    # --- Optional: Add mean logit and activation maps later if useful ---



def plot_misclassified_examples_activations(
    model, data_loader, device, epoch, save_dir, max_samples=10
):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get raw activations (a) and predictions
            inference_out = model.layers["SoftHebbian1"].inference(inputs)
            a = inference_out.a.cpu()  # raw cosine similarity activations
            y_pred = model.layers["SoftHebbian2"](inference_out.y)  # still use y
            pred_labels = y_pred.argmax(dim=1)

            for i in range(inputs.size(0)):
                if pred_labels[i] != labels[i]:
                    misclassified.append(
                        (
                            inputs[i].cpu(),
                            pred_labels[i].item(),
                            labels[i].item(),
                            a[i],  # now storing cosine similarity a[i], not y[i]
                        )
                    )
                if len(misclassified) >= max_samples:
                    break
            if len(misclassified) >= max_samples:
                break

    # Prepare save paths
    img_save_dir = os.path.join(save_dir, "images")
    act_save_dir = os.path.join(save_dir, "activations")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(act_save_dir, exist_ok=True)

    # --- Plot 1: Input images with labels ---
    fig1, axs1 = plt.subplots(1, max_samples, figsize=(2 * max_samples, 2.5))
    axs1 = axs1.flatten()

    for i, (x, y_pred, y_true, _) in enumerate(misclassified):
        x_img = x.view(28, 28) if x.numel() == 784 else x.squeeze()
        axs1[i].imshow(x_img, cmap="gray")
        axs1[i].set_title(f"{y_pred} / {y_true}", fontsize=9)
        axs1[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{img_save_dir}/epoch_{epoch}_misclassified_inputs.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Individual activation heatmaps + average, each with a colorbar ---
    fig2, axs2 = plt.subplots(
        1, max_samples + 1, figsize=(2.5 * (max_samples + 1), 2.5)
    )
    axs2 = axs2.flatten()

    all_h = []

    for i, (_, _, _, h_val) in enumerate(misclassified):
        h_np = h_val.numpy()
        all_h.append(h_np)

        h_len = len(h_np)
        side = int(math.sqrt(h_len))

        if side * side == h_len:
            grid = h_np.reshape(side, side)
        else:
            new_len = (side + 1) ** 2
            padded = np.zeros(new_len)
            padded[:h_len] = h_np
            grid = padded.reshape(side + 1, side + 1)

        norm_grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)

        im = axs2[i].imshow(norm_grid, cmap="viridis")
        axs2[i].axis("off")
        axs2[i].set_title(f"Sample {i+1}", fontsize=7)

        cbar = fig2.colorbar(im, ax=axs2[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    # --- Add mean activation plot ---
    all_h_np = np.stack(all_h)  # [num_samples, h_dim]
    avg_h = np.mean(all_h_np, axis=0)

    h_len = len(avg_h)
    side = int(math.sqrt(h_len))

    if side * side == h_len:
        grid_avg = avg_h.reshape(side, side)
    else:
        new_len = (side + 1) ** 2
        padded = np.zeros(new_len)
        padded[:h_len] = avg_h
        grid_avg = padded.reshape(side + 1, side + 1)

    norm_avg = (grid_avg - grid_avg.min()) / (grid_avg.max() - grid_avg.min() + 1e-9)
    im_avg = axs2[-1].imshow(norm_avg, cmap="viridis")
    axs2[-1].axis("off")
    axs2[-1].set_title("Mean", fontsize=8)

    cbar = fig2.colorbar(im_avg, ax=axs2[-1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(
        f"{act_save_dir}/epoch_{epoch}_activations_grid_avg_colorbars.png", dpi=200
    )
    plt.close(fig2)

    # --- Plot 3: Side-by-side input vs activation image for each misclassified sample ---
    fig3, axs3 = plt.subplots(max_samples, 2, figsize=(4, 2.5 * max_samples))

    for i, (x, _, _, h_val) in enumerate(misclassified):
        x_img = x.view(28, 28).numpy()
        h_np = h_val.numpy()
        h_len = len(h_np)
        side = int(math.sqrt(h_len))
        if side * side == h_len:
            h_img = h_np.reshape(side, side)
        else:
            new_len = (side + 1) ** 2
            padded = np.zeros(new_len)
            padded[:h_len] = h_np
            h_img = padded.reshape(side + 1, side + 1)

        norm_h_img = (h_img - h_img.min()) / (h_img.max() - h_img.min() + 1e-9)

        axs3[i, 0].imshow(x_img, cmap="gray")
        axs3[i, 0].set_title("Input", fontsize=8)
        axs3[i, 0].axis("off")

        im = axs3[i, 1].imshow(norm_h_img, cmap="viridis")
        axs3[i, 1].set_title("Cosine Similarity a", fontsize=8)
        axs3[i, 1].axis("off")

        cbar = fig3.colorbar(im, ax=axs3[i, 1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(f"{act_save_dir}/epoch_{epoch}_x_vs_h_reshaped.png", dpi=200)
    plt.close(fig3)



def plot_misclassified_examples_original(
    model, data_loader, device, epoch, save_dir, max_samples=10
):

    model.eval()
    misclassified = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get activations and predictions
            h = model.layers["SoftHebbian1"](inputs)
            y_pred = model.layers["SoftHebbian2"](h)
            pred_labels = y_pred.argmax(dim=1)

            for i in range(inputs.size(0)):
                if pred_labels[i] != labels[i]:
                    misclassified.append(
                        (
                            inputs[i].cpu(),
                            pred_labels[i].item(),
                            labels[i].item(),
                            h[i].cpu(),
                        )
                    )
                if len(misclassified) >= max_samples:
                    break
            if len(misclassified) >= max_samples:
                break

    # Prepare save paths
    img_save_dir = os.path.join(save_dir, "images")
    act_save_dir = os.path.join(save_dir, "activations")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(act_save_dir, exist_ok=True)

    # --- Plot 1: Input images with labels ---
    fig1, axs1 = plt.subplots(1, max_samples, figsize=(2 * max_samples, 2.5))
    axs1 = axs1.flatten()

    for i, (x, y_pred, y_true, _) in enumerate(misclassified):
        x_img = x.view(28, 28) if x.numel() == 784 else x.squeeze()
        axs1[i].imshow(x_img, cmap="gray")
        axs1[i].set_title(f"{y_pred} / {y_true}", fontsize=9)
        axs1[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{img_save_dir}/epoch_{epoch}_misclassified_inputs.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Individual activation heatmaps + average, each with a colorbar ---
    fig2, axs2 = plt.subplots(
        1, max_samples + 1, figsize=(2.5 * (max_samples + 1), 2.5)
    )
    axs2 = axs2.flatten()

    all_h = []

    for i, (_, _, _, h_val) in enumerate(misclassified):
        h_np = h_val.numpy()
        all_h.append(h_np)

        h_len = len(h_np)
        side = int(math.sqrt(h_len))

        if side * side == h_len:
            grid = h_np.reshape(side, side)
        else:
            new_len = (side + 1) ** 2
            padded = np.zeros(new_len)
            padded[:h_len] = h_np
            grid = padded.reshape(side + 1, side + 1)

        norm_grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)

        im = axs2[i].imshow(norm_grid, cmap="viridis")
        axs2[i].axis("off")
        axs2[i].set_title(f"Sample {i+1}", fontsize=7)

        # Add individual colorbar
        cbar = fig2.colorbar(im, ax=axs2[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    # --- Add mean activation plot ---
    all_h_np = np.stack(all_h)  # [num_samples, h_dim]
    avg_h = np.mean(all_h_np, axis=0)

    h_len = len(avg_h)
    side = int(math.sqrt(h_len))

    if side * side == h_len:
        grid_avg = avg_h.reshape(side, side)
    else:
        new_len = (side + 1) ** 2
        padded = np.zeros(new_len)
        padded[:h_len] = avg_h
        grid_avg = padded.reshape(side + 1, side + 1)

    norm_avg = (grid_avg - grid_avg.min()) / (grid_avg.max() - grid_avg.min() + 1e-9)
    im_avg = axs2[-1].imshow(norm_avg, cmap="viridis")
    axs2[-1].axis("off")
    axs2[-1].set_title("Mean", fontsize=8)

    # Add colorbar to mean subplot
    cbar = fig2.colorbar(im_avg, ax=axs2[-1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(
        f"{act_save_dir}/epoch_{epoch}_activations_grid_avg_colorbars.png", dpi=200
    )
    plt.close(fig2)

    # --- Plot 3: Side-by-side input vs activation image for each misclassified sample ---
    fig3, axs3 = plt.subplots(max_samples, 2, figsize=(4, 2.5 * max_samples))

    for i, (x, _, _, h_val) in enumerate(misclassified):
        # Prepare input image
        x_img = x.view(28, 28).numpy()

        # Prepare activation image
        h_np = h_val.numpy()
        h_len = len(h_np)
        side = int(math.sqrt(h_len))
        if side * side == h_len:
            h_img = h_np.reshape(side, side)
        else:
            new_len = (side + 1) ** 2
            padded = np.zeros(new_len)
            padded[:h_len] = h_np
            h_img = padded.reshape(side + 1, side + 1)

        norm_h_img = (h_img - h_img.min()) / (h_img.max() - h_img.min() + 1e-9)

        # Plot input
        axs3[i, 0].imshow(x_img, cmap="gray")
        axs3[i, 0].set_title("Input", fontsize=8)
        axs3[i, 0].axis("off")

        # Plot reshaped activation
        im = axs3[i, 1].imshow(norm_h_img, cmap="viridis")
        axs3[i, 1].set_title("Reshaped h", fontsize=8)
        axs3[i, 1].axis("off")

        # Add colorbar for each reshaped activation
        cbar = fig3.colorbar(im, ax=axs3[i, 1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(f"{act_save_dir}/epoch_{epoch}_x_vs_h_reshaped.png", dpi=200)
    plt.close(fig3)



def plot_misclassified_examples_debugger(
    model, data_loader, device, epoch, save_dir, max_samples=10
):

    model.eval()
    misclassified = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            print(f"\n[Epoch {epoch}] Debug Info")
            print(f"  inputs.shape: {inputs.shape}")
            print(f"  inputs.min(): {inputs.min().item():.4f}")
            print(f"  inputs.max(): {inputs.max().item():.4f}")
            print(f"  inputs.mean(): {inputs.mean().item():.4f}")

            # --- Inference out ---
            layer1 = model.layers.get("SoftHebbian1", None)
            inference_out = layer1.inference(inputs)
            a = inference_out.a
            y = inference_out.y

            lamb = getattr(layer1, "lamb", None)
            logprior = getattr(layer1, "logprior", None)
            lamb_device = lamb.device if isinstance(lamb, torch.Tensor) else inputs.device
            if logprior is not None:
                logits = lamb * a + logprior.to(lamb_device)
            else:
                logits = torch.Tensor([-1])

            # --- Predictions ---
            y_pred = model.layers["SoftHebbian2"](y)
            pred_labels = y_pred.argmax(dim=1)

            # --- a stats (cosine similarity) ---
            print(f"  a.shape: {a.shape}")
            print(f"  a.min(): {a.min().item():.4f}")
            print(f"  a.max(): {a.max().item():.4f}")
            print(f"  a.mean(): {a.mean().item():.4f}")

            # --- logits stats ---
            print(f"  logits.shape: {logits.shape}")
            print(f"  logits.min(): {logits.min().item():.4f}")
            print(f"  logits.max(): {logits.max().item():.4f}")
            print(f"  logits.mean(): {logits.mean().item():.4f}")

            # --- Output y_pred stats ---
            print(f"  y_pred.shape: {y_pred.shape}")
            print(f"  y_pred.min(): {y_pred.min().item():.4f}")
            print(f"  y_pred.max(): {y_pred.max().item():.4f}")
            print(f"  y_pred.mean(): {y_pred.mean().item():.4f}")
            print(f"  pred_labels: {pred_labels.tolist()}")
            print(f"  true_labels: {labels.tolist()}")

            # --- lamb and logprior ---
            if isinstance(lamb, torch.Tensor):
                print(f"  lamb: {lamb.item():.4f}")
            if isinstance(logprior, torch.Tensor):
                print(f"  logprior.shape: {logprior.shape}")
                print(f"  logprior.min(): {logprior.min().item():.4f}")
                print(f"  logprior.max(): {logprior.max().item():.4f}")
                print(f"  logprior.mean(): {logprior.mean().item():.4f}")

            for i in range(inputs.size(0)):
                if pred_labels[i] != labels[i]:
                    misclassified.append(
                        (
                            inputs[i].cpu(),
                            pred_labels[i].item(),
                            labels[i].item(),
                            y[i].cpu(),  # NOTE: still using h = y here
                        )
                    )
                if len(misclassified) >= max_samples:
                    break
            if len(misclassified) >= max_samples:
                break    # Prepare save paths
    img_save_dir = os.path.join(save_dir, "images")
    act_save_dir = os.path.join(save_dir, "activations")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(act_save_dir, exist_ok=True)

    # --- Plot 1: Input images with labels ---
    fig1, axs1 = plt.subplots(1, max_samples, figsize=(2 * max_samples, 2.5))
    axs1 = axs1.flatten()

    for i, (x, y_pred, y_true, _) in enumerate(misclassified):
        x_img = x.view(28, 28) if x.numel() == 784 else x.squeeze()
        axs1[i].imshow(x_img, cmap="gray")
        axs1[i].set_title(f"{y_pred} / {y_true}", fontsize=9)
        axs1[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{img_save_dir}/epoch_{epoch}_misclassified_inputs.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Individual activation heatmaps + average, each with a colorbar ---
    fig2, axs2 = plt.subplots(
        1, max_samples + 1, figsize=(2.5 * (max_samples + 1), 2.5)
    )
    axs2 = axs2.flatten()

    all_h = []

    for i, (_, _, _, h_val) in enumerate(misclassified):
        h_np = h_val.numpy()
        all_h.append(h_np)

        h_len = len(h_np)
        side = int(math.sqrt(h_len))

        if side * side == h_len:
            grid = h_np.reshape(side, side)
        else:
            new_len = (side + 1) ** 2
            padded = np.zeros(new_len)
            padded[:h_len] = h_np
            grid = padded.reshape(side + 1, side + 1)

        norm_grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)

        im = axs2[i].imshow(norm_grid, cmap="viridis")
        axs2[i].axis("off")
        axs2[i].set_title(f"Sample {i+1}", fontsize=7)

        # Add individual colorbar
        cbar = fig2.colorbar(im, ax=axs2[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    # --- Add mean activation plot ---
    all_h_np = np.stack(all_h)  # [num_samples, h_dim]
    avg_h = np.mean(all_h_np, axis=0)

    h_len = len(avg_h)
    side = int(math.sqrt(h_len))

    if side * side == h_len:
        grid_avg = avg_h.reshape(side, side)
    else:
        new_len = (side + 1) ** 2
        padded = np.zeros(new_len)
        padded[:h_len] = avg_h
        grid_avg = padded.reshape(side + 1, side + 1)

    norm_avg = (grid_avg - grid_avg.min()) / (grid_avg.max() - grid_avg.min() + 1e-9)
    im_avg = axs2[-1].imshow(norm_avg, cmap="viridis")
    axs2[-1].axis("off")
    axs2[-1].set_title("Mean", fontsize=8)

    # Add colorbar to mean subplot
    cbar = fig2.colorbar(im_avg, ax=axs2[-1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(
        f"{act_save_dir}/epoch_{epoch}_activations_grid_avg_colorbars.png", dpi=200
    )
    plt.close(fig2)

    # --- Plot 3: Side-by-side input vs activation image for each misclassified sample ---
    fig3, axs3 = plt.subplots(max_samples, 2, figsize=(4, 2.5 * max_samples))

    for i, (x, _, _, h_val) in enumerate(misclassified):
        # Prepare input image
        x_img = x.view(28, 28).numpy()

        # Prepare activation image
        h_np = h_val.numpy()
        h_len = len(h_np)
        side = int(math.sqrt(h_len))
        if side * side == h_len:
            h_img = h_np.reshape(side, side)
        else:
            new_len = (side + 1) ** 2
            padded = np.zeros(new_len)
            padded[:h_len] = h_np
            h_img = padded.reshape(side + 1, side + 1)

        norm_h_img = (h_img - h_img.min()) / (h_img.max() - h_img.min() + 1e-9)

        # Plot input
        axs3[i, 0].imshow(x_img, cmap="gray")
        axs3[i, 0].set_title("Input", fontsize=8)
        axs3[i, 0].axis("off")

        # Plot reshaped activation
        im = axs3[i, 1].imshow(norm_h_img, cmap="viridis")
        axs3[i, 1].set_title("Reshaped h", fontsize=8)
        axs3[i, 1].axis("off")

        # Add colorbar for each reshaped activation
        cbar = fig3.colorbar(im, ax=axs3[i, 1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(f"{act_save_dir}/epoch_{epoch}_x_vs_h_reshaped.png", dpi=200)
    plt.close(fig3)

def plot_misclassified_examples(
    model, data_loader, device, epoch, save_dir, max_samples=10
):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            print(f"\n[Epoch {epoch}] Batch {batch_idx}")
            print(f"  inputs.shape: {inputs.shape}")
            print(f"  inputs.min(): {inputs.min().item():.4f}")
            print(f"  inputs.max(): {inputs.max().item():.4f}")
            print(f"  inputs.mean(): {inputs.mean().item():.4f}")

            # Get inference activations and predictions
            inference_out = model.layers["SoftHebbian1"].inference(inputs)
            a = inference_out.a  # Cosine similarities
            h = model.layers["SoftHebbian1"].y(a)  # Output from inhibition function

            print(f"  a.shape (cosine sim): {a.shape}")
            print(f"  a.min(): {a.min().item():.4e}")
            print(f"  a.max(): {a.max().item():.4e}")
            print(f"  a.mean(): {a.mean().item():.4e}")
            print(f"  a[0][:10]: {a[0][:10].tolist()}  # First 10 cosine sim of sample 0")

            print(f"  h.shape: {h.shape}")
            print(f"  h.min(): {h.min().item():.4e}")
            print(f"  h.max(): {h.max().item():.4e}")
            print(f"  h.mean(): {h.mean().item():.4e}")
            print(f"  h[0][:10]: {h[0][:10].tolist()}  # First 10 activations of sample 0")

            row_means = h.mean(dim=1)  # Average activation per input sample
            print(f"  h row-wise mean: {[round(v.item(), 6) for v in row_means]}")

            y_pred = model.layers["SoftHebbian2"](h)
            pred_labels = y_pred.argmax(dim=1)

            for i in range(inputs.size(0)):
                if pred_labels[i] != labels[i]:
                    misclassified.append(
                        (
                            inputs[i].cpu(),
                            pred_labels[i].item(),
                            labels[i].item(),
                            h[i].cpu(),
                        )
                    )
                if len(misclassified) >= max_samples:
                    break
            if len(misclassified) >= max_samples:
                break

class BaseSoftExperiment(Experiment):
    """
    CLASS
    Experiment for base training and testing of model
    @instance attr.
        Experiment ATTR.
            model (Network): model used in experiment
            batch_size (int): size of each batch of data
            epochs (int): number of epochs to train
            test_sample (int): interval at which testing will be done
            device (str): device that will be used for CUDA
            local_machine (bool): where code is ran
            experiment_type (ExperimentTypes): what type of experiment to be ran

            START_TIME (float): start time of experiment
            END_TIMER (float): end of experiment
            DURATION (float): duration of experiment
            TRAIN_TIME (float): training time
            TEST_ACC_TIME (float): testing time
            TRAIN_ACC_TIME (float): testing time
            EXP_NAME (str): experiment name
            RESULT_PATH (str): where result files will be created
            PRINT_LOG (logging.Logger): print log
            TEST_LOG (logging.Logger): log with all test accuracy results
            TRAIN_LOG (logging.Logger): log with all trainning accuracy results
            PARAM_LOG (logging.Logger): parameter log for experiment
            DEBUG_LOG (logging.Logger): debugging
            EXP_LOG (logging.Logger): logging of experiment process
        OWN ATTR.
            data_name (str): name of dataset
            train_data (str): path to train data
            train_label (str): path to train label
            train_fname (str): path to train filename
            test_data (str): path to test data
            test_label (str): path to test label
            test_fname (str): path to test filename

            SAMPLES (int): number of samples seen in training

            train_data_set (TensorDataset): training dataset
            train_data_loader (DataLoader): training dataloader
            test_data_set (TensorDataset): testing dataset
            test_data_loader (DataLoader): testing dataloader
    """

    def __init__(self, model: Network, args: argparse.Namespace, name: str) -> None:
        """
        CONTRUCTOR METHOD
        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
        """
        super().__init__(model, args, name)
        self.SAMPLES: int = 0

        # set_global_seed(args.seed)

        dataset_mapping = {member.name.upper(): member for member in DataSets}
        self.dataset = dataset_mapping[self.data_name.upper()]

        self.train_data = args.train_data
        self.train_label = args.train_label
        self.test_data = args.test_data
        self.test_label = args.test_label
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.classes = args.classes
        self.train_fname = args.train_fname
        self.test_fname = args.test_fname

        input_layer: InputLayer = DataSetupLayer()
        input_class: Type[InputLayer] = globals()[input_layer.__class__.__name__]

        self.train_data_set: TensorDataset = input_class.setup_data(
            self.train_data,
            self.train_label,
            self.train_fname,
            self.train_size,
            self.dataset,
        )
        self.train_data_loader: DataLoader = DataLoader(
            self.train_data_set, batch_size=self.batch_size, shuffle=True
        )
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")

        self.test_data_set: TensorDataset = input_class.setup_data(
            self.test_data,
            self.test_label,
            self.test_fname,
            self.test_size,
            self.dataset,
        )
        self.test_data_loader: DataLoader = DataLoader(
            self.test_data_set, batch_size=self.batch_size, shuffle=True
        )
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader.")

        self.DEBUG_LOG.info(f"DEVICE USED: {self.device}")
        ##Debugging:
        self.weight_list = []
        self.lambda_over_epochs = {name: [] for name in self.model.layers.keys()}

    def _base_train(
        self,
        train_data_loader: DataLoader,
        epoch: int,
        dname: str,
        visualize: bool = True,
    ) -> None:
        """
        METHOD
        Base training of model for 1 epoch
        @param
            train_data_loader: dataloader with the training data
            epoch : training epoch current training loop is at
            dname: dataset name
            visualize: if the weights of model should be visualized
        @return
            None
        """

        train_epoch_start: float = self.TRAIN_TIME

        train_start: float = time.time()
        self.EXP_LOG.info(f"Started 'base_train' function with {dname.upper()}.")

        train_batches_per_epoch: int = len(train_data_loader)
        self.EXP_LOG.info(
            f"This training batch is epoch #{epoch} with {train_batches_per_epoch} batches of size {self.batch_size} in this epoch."
        )

        for idx, (inputs, labels) in enumerate(
            tqdm(train_data_loader, desc="Training batch", leave=False)
        ):

            # # Test model at intervals of samples seen
            # if self.check_test(self.SAMPLES):
            #     # Pause train timer and add to total time
            #     train_pause_time: float = time.time()
            #     self.TRAIN_TIME += train_pause_time - train_start
            #
            #     self._testing(
            #         self.test_data_loader,
            #         Purposes.TEST_ACCURACY,
            #         self.data_name,
            #         ExperimentPhases.BASE,
            #     )
            #
            #     self._testing(
            #         self.train_data_loader,
            #         Purposes.TRAIN_ACCURACY,
            #         self.data_name,
            #         ExperimentPhases.BASE,
            #     )
            #
            #     train_start = time.time()

            inputs, labels = (
                inputs.to(self.device).float(),
                one_hot(labels, self.model.output_dim)
                .squeeze()
                .to(self.device)
                .float(),
            )

            self.model.train()
            self.model(inputs, clamped=labels)

            self.SAMPLES += 1

        train_end: float = time.time()
        self.TRAIN_TIME += train_end - train_start
        train_epoch_end: float = self.TRAIN_TIME
        training_time = train_epoch_end - train_epoch_start

        self.EXP_LOG.info(
            f"Training of epoch #{epoch} took {time_to_str(training_time)}."
        )
        total_norm = (
            torch.nn.utils.parameters_to_vector(self.model.parameters()).norm(2).item()
        )
        self.weight_list.append(total_norm)
        self.WEIGHT_LOG.info(
            f"Model weight L2 norm after epoch #{epoch}: {total_norm:.4f}"
        )
        self.EXP_LOG.info("Completed 'base_train' function.")
        for name, layer in self.model.layers.items():
            self.lambda_over_epochs[name].append(layer.lamb.item())
            self.LAMBDA_LOG.info(
                f"Model lambda after epoch #{epoch}: {layer.lamb.item()}"
            )

    def _base_test(
        self,
        test_data_loader: DataLoader,
        purpose: Purposes,
        dname: str,
        visualize: bool = True,
    ) -> float:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            dname: dataset name
            last: is it final test
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        test_start: float = time.time()
        self.EXP_LOG.info(f"Started 'base_test' function with {dname.upper()}.")

        test_batches_per_epoch = len(test_data_loader)
        self.EXP_LOG.info(
            f"This testing is done after samples seen #{self.SAMPLES} with {test_batches_per_epoch} batches of size {self.batch_size} in this epoch."
        )

        self.model.eval()
        self.EXP_LOG.info("Set the model to testing mode.")

        final_accuracy: float = 0

        with torch.no_grad():
            correct: int = 0
            total: int = len(test_data_loader) * self.batch_size

            for inputs, labels in test_data_loader:
                # Move input and targets to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Inference
                predictions: torch.Tensor = self.model(inputs)

                # Evaluates performance of model on testing dataset
                correct += (
                    (predictions.argmax(-1) == labels).type(torch.float).sum().item()
                )

            final_accuracy = round(correct / total, 4)

        test_end = time.time()
        testing_time = test_end - test_start

        if purpose == Purposes.TEST_ACCURACY:
            self.TEST_ACC_TIME += testing_time
        if purpose == Purposes.TRAIN_ACCURACY:
            self.TRAIN_ACC_TIME += testing_time

        self.EXP_LOG.info(f"Completed testing with {correct} out of {total}.")
        self.EXP_LOG.info("Completed 'base_test' function.")
        self.EXP_LOG.info(
            f"Testing ({purpose.value.lower()} acc) of sample #{self.SAMPLES} took {time_to_str(testing_time)}."
        )

        if purpose == Purposes.TEST_ACCURACY:
            self.TEST_LOG.info(
                f"Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Test Accuracy: {final_accuracy}"
            )
        if purpose == Purposes.TRAIN_ACCURACY:
            self.TRAIN_LOG.info(
                f"Samples Seen: {self.SAMPLES} || Dataset: {dname.upper()} || Train Accuracy: {final_accuracy}"
            )

        return final_accuracy

    def _training(
        self,
        train_data_loader: DataLoader,
        epoch: int,
        dname: str,
        phase: ExperimentPhases,
        visualize: bool = True,
    ) -> None:
        """
        METHOD
        Train model for 1 epoch
        @param
            train_data_loader: dataloader containing the training dataset
            epoch: epoch number of training iteration that is being tested on
            dname: dataset name
            phase: which part of experiment -> which training to do
        @return
            None
        """
        if phase == ExperimentPhases.BASE:
            self._base_train(train_data_loader, epoch, dname, visualize)

    def _testing(
        self,
        test_data_loader: DataLoader,
        purpose: Purposes,
        dname: str,
        phase: ExperimentPhases,
        visualize: bool = True,
    ) -> float:
        """
        METHOD
        Test model with test dataset and determine its accuracy
        @param
            test_data_loader: dataloader containing the testing dataset
            purpose: name of set for logging purposes (test/train)
            dname: dataset name
            phase: which part of experiment -> which test to do
        @return
            accuracy: float value between [0, 1] to show accuracy model got on test
        """
        if phase == ExperimentPhases.BASE:
            return self._base_test(test_data_loader, purpose, dname, visualize)
        else:
            return 0

    def _param_start_log(self):
        self.EXP_LOG.info("Started logging of experiment parameters.")

        self.PARAM_LOG.info(
            f"Experiment Type: {self.experiment_type.value.lower().capitalize()}"
        )
        self.PARAM_LOG.info(f"Device: {self.device.upper()}")
        self.PARAM_LOG.info(
            f"Start time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.START_TIME))}"
        )

        self.EXP_LOG.info("Completed logging of experiment parameters.")

    def _param_end_log(self):
        self.PARAM_LOG.info(
            f"End time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.END_TIME))}"
        )
        self.PARAM_LOG.info(
            f"Runtime of experiment: {time_to_str(self.DURATION if self.DURATION != None else 0)}"
        )
        self.PARAM_LOG.info(
            f"Total train time of experiment: {time_to_str(self.TRAIN_TIME)}"
        )
        self.PARAM_LOG.info(
            f"Total test time (test acc) of experiment: {time_to_str(self.TEST_ACC_TIME)}"
        )
        self.PARAM_LOG.info(
            f"Total test time (train acc) of experiment: {time_to_str(self.TRAIN_ACC_TIME)}"
        )

    def _final_test_log(self, results) -> None:
        test_acc, train_acc = results
        self.PARAM_LOG.info(
            f"Training accuracy of model after training for {self.epochs} epochs: {train_acc}"
        )
        self.PARAM_LOG.info(
            f"Testing accuracy of model after training for {self.epochs} epochs: {test_acc}"
        )

    ################################################################################################
    # Running Experiment
    ################################################################################################
    def _experiment(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        torch.device(self.device)
        root_dir = get_root_dir()
        weight_matrix_save_path = f"{root_dir}/plots/weight_matrix/{self.EXP_NAME}"

        self.EXP_LOG.info("Started training and testing loops.")
        training_acc = []
        testing_acc = []

        for epoch in tqdm(range(0, self.epochs), desc="Epochs"):
            self._training(
                self.train_data_loader, epoch, self.data_name, ExperimentPhases.BASE
            )

            # if epoch + 1 == 500:
            #     model_path = os.path.join(
            #         get_root_dir(), "checkpoints", f"{self.EXP_NAME}_epoch500.pt"
            #     )
            #     os.makedirs(os.path.dirname(model_path), exist_ok=True)
            #     torch.save(self.model.state_dict(), model_path)
            #     self.EXP_LOG.info(f"Saved model weights at epoch 500 to: {model_path}")

            # if (epoch + 1) % 50 == 0:
            #     for name, layer in self.model.layers.items():
            #         if hasattr(layer, "weight"):
            #             save_dir = os.path.join(weight_matrix_save_path, name)
            #             if name == "SoftHebbian1":
            #                 plot_weight_grid(layer.weight, name, epoch + 1, save_dir)
            #             else:
            #                 plot_weight_heatmap(layer.weight, name, epoch + 1, save_dir)
            #
            #     plot_misclassified_examples_original(
            #         self.model,
            #         self.test_data_loader,
            #         self.device,
            #         epoch + 1,
            #         save_dir=f"{root_dir}/plots/misclassified/{self.EXP_NAME}",
            #         max_samples=20,
            #     )
            testing_acc.append(
                self._testing(
                    self.test_data_loader,
                    Purposes.TEST_ACCURACY,
                    self.data_name,
                    ExperimentPhases.BASE,
                )
            )
            training_acc.append(
                self._testing(
                    self.train_data_loader,
                    Purposes.TRAIN_ACCURACY,
                    self.data_name,
                    ExperimentPhases.BASE,
                )
            )

        self.EXP_LOG.info("Completed training of model.")
        self.EXP_LOG.info("Visualize weights of model after training.")
        return (
            training_acc,
            testing_acc,
            self.weight_list,
            self.lambda_over_epochs["SoftHebbian1"],
        )

    def _final_test(self) -> Tuple[float, ...]:
        test_acc: float = self._testing(
            self.test_data_loader,
            Purposes.TEST_ACCURACY,
            self.data_name,
            ExperimentPhases.BASE,
            visualize=False,
        )
        train_acc: float = self._testing(
            self.train_data_loader,
            Purposes.TRAIN_ACCURACY,
            self.data_name,
            ExperimentPhases.BASE,
            visualize=False,
        )

        self.EXP_LOG.info("Completed final testing methods.")
        return (test_acc, train_acc)
