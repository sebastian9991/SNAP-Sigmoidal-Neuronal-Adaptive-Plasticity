import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_weight_heatmap(
    weight_matrix: torch.Tensor, layer_name: str, epoch: int, save_dir: str
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    weight_np = weight_matrix.detach().cpu().numpy()
    abs_weights = np.abs(weight_np) + 1e-10  # add epsilon to avoid log(0)

    # LogNorm will stretch small values and compress large ones
    norm = mcolors.LogNorm(vmin=abs_weights.min(), vmax=abs_weights.max())

    plt.figure(figsize=(10, 6))
    sns.heatmap(abs_weights, cmap="viridis", norm=norm, cbar=True)
    plt.title(f"Log-Scaled Weight Heatmap - {layer_name} - Epoch {epoch}")
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")

    cbar = plt.gca().collections[0].colorbar
    cbar.set_label("log(|weight| + ε)")

    filename = os.path.join(save_dir, f"weights_{layer_name}_epoch_{epoch}_log.png")
    plt.savefig(filename)
    plt.close()


def plot_mnist_weight_filters(
    weight_matrix: torch.Tensor, layer_name: str, epoch: int, save_dir: str
):
    """
    Visualizes each neuron's weight as a 28x28 MNIST-style image.
    """
    full_path = os.path.join(save_dir, layer_name, "mnist_like_filters")
    os.makedirs(full_path, exist_ok=True)

    num_filters = weight_matrix.shape[0]  # One image per output neuron
    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))

    for i, ax in enumerate(axes):
        weights_2d = weight_matrix[i].detach().cpu().numpy().reshape(28, 28)
        ax.imshow(weights_2d, cmap="jet")
        ax.axis("off")

    plt.suptitle(f"{layer_name} Weights as MNIST Filters - Epoch {epoch}")
    filename = os.path.join(full_path, f"mnist_filters_{layer_name}_epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()


def plot_weight_grid(
    weight_tensor: torch.Tensor,
    layer_name: str,
    epoch: int,
    save_dir: str,
    square_dim: int = 28,
):
    os.makedirs(save_dir, exist_ok=True)

    output_dim, input_dim = weight_tensor.shape
    target_dim = square_dim * square_dim

    # Pad or truncate to square
    if input_dim < target_dim:
        pad = target_dim - input_dim
        weight_tensor = torch.cat(
            [
                weight_tensor,
                torch.zeros((output_dim, pad), device=weight_tensor.device),
            ],
            dim=1,
        )
    elif input_dim > target_dim:
        weight_tensor = weight_tensor[:, :target_dim]

    grid_cols = math.ceil(math.sqrt(output_dim))  # e.g., 32
    grid_rows = math.ceil(output_dim / grid_cols)  # e.g., 31

    fig, axes = plt.subplots(
        grid_rows, grid_cols, figsize=(grid_cols, grid_rows), dpi=100
    )

    # Flatten and only use the first 992 axes
    axes = axes.flatten()
    for i in range(output_dim):
        weight_matrix = (
            weight_tensor[i].detach().cpu().numpy().reshape(square_dim, square_dim)
        )
        ax = axes[i]
        ax.imshow(weight_matrix, cmap="viridis")
        ax.axis("off")

    for j in range(output_dim, len(axes)):
        axes[j].remove()  # or axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(f"Weights of {layer_name} at epoch {epoch}", fontsize=14)
    plt.savefig(os.path.join(save_dir, f"{layer_name}_epoch_{epoch}_grid.png"))
    plt.close(fig)


def plot_misclassified_samples(misclassified, title, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(20, 3, figsize=(6, 40))  # 20 samples × (x, y, h)

    for i, (x, y, h) in enumerate(misclassified):
        x_img = x.reshape(28, 28)
        h_img = h.reshape(31, 32)  # reshaped SoftHebbian1 activation

        axes[i, 0].imshow(x_img, cmap="gray")
        axes[i, 0].set_title("Input x")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(y.unsqueeze(0), cmap="Blues", aspect="auto")
        axes[i, 1].set_title(f"Predicted y = {y.item()}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(h_img, cmap="viridis")
        axes[i, 2].set_title("Activation h")
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    fig.savefig(os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()
