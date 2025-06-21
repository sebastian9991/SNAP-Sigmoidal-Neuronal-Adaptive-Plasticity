import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_weight_heatmap(
    weight_matrix: torch.Tensor,
    layer_name: str,
    epoch: int,
    save_dir: str,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    sns.heatmap(weight_matrix.detach().cpu().numpy(), cmap="viridis", cbar=True)
    plt.title(f"Weight Heatmap - {layer_name} - Epoch {epoch}")
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")
    filename = f"{save_dir}/weights_{layer_name}_epoch_{epoch}.png"
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
