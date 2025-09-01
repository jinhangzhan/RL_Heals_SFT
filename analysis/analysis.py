import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# --------------------------------------------------------------------------------
# Example model loading (placeholder).
# You need to replace it with your actual model & load logic.
# --------------------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint_path):
    """
    Loads the model from a given checkpoint path.
    Replace this with the actual loading code for your LLaMA or other model.
    """
    # Example:
    # model = MyModelClass.load_from_checkpoint(checkpoint_path)
    # model.eval()
    # return model
    pass


# --------------------------------------------------------------------------------
# 1) Loss Dynamics Analysis
# --------------------------------------------------------------------------------
def parse_training_logs(log_file):
    """
    Parses a training log file (CSV or JSON or custom) and returns
    data frames or arrays for plotting (e.g. steps, training loss, validation loss).
    """
    # Placeholder for log parsing.
    # Suppose your training log is a CSV with columns: step, train_loss, val_loss, ...
    df = pd.read_csv(log_file)
    return df


def plot_loss_dynamics(df, output_path=None):
    """
    Plots training vs. validation loss over steps.
    """
    steps = df['step'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values

    plt.figure()
    plt.plot(steps, train_loss, label='Training Loss')
    plt.plot(steps, val_loss, label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    plt.show()


# --------------------------------------------------------------------------------
# 2) Gradient Norms (or Parameter Updates) Analysis
# --------------------------------------------------------------------------------
def compute_gradient_norms(model, data_loader, device='cuda'):
    """
    Computes gradient norms for a few batches (placeholder).
    In practice, you'd do this during training or right after backprop.
    """
    # This is just a stub.
    # You'd typically integrate gradient norm recording inside your training loop.
    grad_norms = []
    for batch in data_loader:
        # forward pass
        # loss = model(...)
        # loss.backward()
        # norms = []
        # for param in model.parameters():
        #     if param.grad is not None:
        #         norms.append(param.grad.data.norm(2).item())
        # grad_norms.append(np.mean(norms))
        pass
    return grad_norms


def plot_gradient_norms(grad_norms, output_path=None):
    plt.figure()
    plt.plot(range(len(grad_norms)), grad_norms, label='Average Gradient Norm')
    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    plt.show()


# --------------------------------------------------------------------------------
# 3) PCA / t-SNE on Hidden States
# --------------------------------------------------------------------------------
def get_hidden_states(model, data_loader, device='cuda', num_samples=100):
    """
    Runs forward passes on a subset of data to collect hidden states.
    """
    # This is placeholder code. Modify it to return actual hidden states from your model.
    hidden_states_list = []
    labels_list = []

    count = 0
    for batch in data_loader:
        # inputs, labels = batch
        # inputs = inputs.to(device)
        # with torch.no_grad():
        #     outputs = model(inputs, output_hidden_states=True)
        #     # For example, take the last hidden state
        #     last_hidden = outputs.hidden_states[-1]  # shape: [batch_size, seq_len, hidden_dim]
        #     # Flatten or pool it
        #     representation = last_hidden.mean(dim=1)  # shape: [batch_size, hidden_dim]
        # hidden_states_list.append(representation.cpu().numpy())
        # labels_list.append(labels.cpu().numpy())

        count += 1
        if count >= num_samples:
            break

    # Combine into arrays
    # hidden_states = np.concatenate(hidden_states_list, axis=0)
    # label_array = np.concatenate(labels_list, axis=0)
    hidden_states = np.random.randn(num_samples, 768)  # Dummy data
    label_array = np.random.randint(0, 2, size=(num_samples,))  # Dummy labels
    return hidden_states, label_array


def pca_analysis(hidden_states, n_components=2):
    """
    Performs PCA on hidden states.
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(hidden_states)
    return reduced, pca


def plot_pca_results(reduced, labels, output_path=None):
    """
    Plots the PCA 2D projection.
    """
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)  # color by label
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if output_path:
        plt.savefig(output_path)
    plt.show()


def tsne_analysis(hidden_states, n_components=2, perplexity=30.0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, init='pca')
    reduced = tsne.fit_transform(hidden_states)
    return reduced


def plot_tsne_results(reduced, labels, output_path=None):
    """
    Plots the t-SNE 2D projection.
    """
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if output_path:
        plt.savefig(output_path)
    plt.show()


# --------------------------------------------------------------------------------
# 4) SVD Analysis on Weights or Activations
# --------------------------------------------------------------------------------
def svd_analysis_on_weights(model):
    """
    Demonstrates an SVD on a chosen weight matrix.
    Replace 'model.some_layer.weight' with whichever layer you want to analyze.
    """
    # Example for a single layer:
    # weight_matrix = model.some_layer.weight.data.cpu().numpy()
    # u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
    # return s
    pass


def plot_singular_values(s, output_path=None):
    plt.figure()
    plt.plot(range(len(s)), s, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    if output_path:
        plt.savefig(output_path)
    plt.show()


# --------------------------------------------------------------------------------
# 5) Attention Analysis
# --------------------------------------------------------------------------------
def analyze_attention_distributions(model, data_loader, device='cuda', num_samples=10):
    """
    Computes some statistics of attention distributions, e.g. average entropy per head.
    This is highly dependent on how your model returns attention weights.
    """
    # Placeholder. Here is a rough outline:
    # entropies = []
    # count = 0
    # for batch in data_loader:
    #     if count >= num_samples:
    #         break
    #     with torch.no_grad():
    #         outputs = model(inputs, output_attentions=True)
    #         attentions = outputs.attentions  # list of [batch_size, num_heads, seq_len, seq_len]
    #         # compute entropy of each head
    #         for layer_attn in attentions:
    #             # shape: [batch_size, heads, seq_len, seq_len]
    #             # ...
    #     count += 1
    # return entropies
    pass


def plot_attention_stats(entropies, output_path=None):
    """
    Plots average attention entropy or other metrics across heads/layers.
    """
    # Example:
    plt.figure()
    plt.bar(range(len(entropies)), entropies)
    plt.xlabel('Head')
    plt.ylabel('Entropy')
    if output_path:
        plt.savefig(output_path)
    plt.show()


# --------------------------------------------------------------------------------
# MAIN ANALYSIS LOOP EXAMPLE
# --------------------------------------------------------------------------------
def main_analysis(
        checkpoint_root="/path/gp_l_sft",
        checkpoints=[0, 1000, 2000, 3000, 4000, 5000, 6000],
        log_file="training_log.csv",
        device="cuda"
):
    """
    High-level analysis loop that:
      1) Parses training log and plots loss
      2) Iterates over a list of checkpoints
      3) For each checkpoint, loads the model, does hidden-state analysis (PCA, t-SNE), SVD, etc.
    """
    # 1. Loss dynamics
    if os.path.exists(log_file):
        df = parse_training_logs(log_file)
        plot_loss_dynamics(df, output_path="loss_dynamics.png")
    else:
        print(f"Log file {log_file} not found, skipping loss dynamics analysis.")

    # 2. For each checkpoint, do the deeper analyses
    for ckpt_step in checkpoints:
        ckpt_path = os.path.join(checkpoint_root, f"checkpoint-{ckpt_step}")
        print(f"Loading checkpoint from: {ckpt_path} ...")

        model = load_model_from_checkpoint(ckpt_path)
        if model is None:
            print(f"Failed to load model at {ckpt_path}, skipping.")
            continue

        model.to(device)
        model.eval()

        # 2.1 Hidden state PCA / t-SNE
        # Replace the placeholder data loader with your actual set.
        dummy_data_loader = None
        hidden_states, labels = get_hidden_states(model, dummy_data_loader, device=device, num_samples=100)

        # PCA
        pca_reduced, pca_model = pca_analysis(hidden_states, n_components=2)
        plot_pca_results(pca_reduced, labels, output_path=f"pca_ckpt_{ckpt_step}.png")

        # t-SNE
        tsne_reduced = tsne_analysis(hidden_states, n_components=2, perplexity=30.0)
        plot_tsne_results(tsne_reduced, labels, output_path=f"tsne_ckpt_{ckpt_step}.png")

        # 2.2 SVD on weights
        # s = svd_analysis_on_weights(model)
        # if s is not None:
        #     plot_singular_values(s, output_path=f"svd_ckpt_{ckpt_step}.png")

        # 2.3 Attention analysis
        # attn_stats = analyze_attention_distributions(model, dummy_data_loader, device=device)
        # if attn_stats is not None:
        #     plot_attention_stats(attn_stats, output_path=f"attn_ckpt_{ckpt_step}.png")

    print("Analysis complete.")


if __name__ == "__main__":
    main_analysis()