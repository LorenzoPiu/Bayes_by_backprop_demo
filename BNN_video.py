#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:34:32 2024

@author: lorenzo piu
"""

#------------------------------------------------------------------------------
#                        Bayesian Neural Networks with Pytorch
#------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import imageio

# Define the Variational Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0e-3):
        super(BayesianLinear, self).__init__()
        # Variational parameters for the weight distribution (mean and log of standard deviation)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-4))
        
        # Variational parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-4))
        
        # Prior standard deviation
        self.prior_std = prior_std

    def forward(self, x):
        # Sample weights and biases using reparameterization trick
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        # Sampling weights
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # KL divergence between learned distribution and the prior
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        # KL for weights
        weight_kl = torch.sum(
            torch.log(self.prior_std / weight_sigma) + 
            (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * self.prior_std ** 2) - 0.5
        )
        
        # KL for biases
        bias_kl = torch.sum(
            torch.log(self.prior_std / bias_sigma) + 
            (bias_sigma ** 2 + self.bias_mu ** 2) / (2 * self.prior_std ** 2) - 0.5
        )
        
        return weight_kl + bias_kl
    
# Define the Bayesian NN
class BayesianNN(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons, output_dim):
        super(BayesianNN, self).__init__()
        
        # Create a list to hold the layers
        self.layers = nn.ModuleList()
        
        # Input layer (first layer)
        self.layers.append(BayesianLinear(input_dim, n_neurons))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(BayesianLinear(n_neurons, n_neurons))
        
        # Output layer (last layer)
        self.layers.append(BayesianLinear(n_neurons, output_dim))
    
    def forward(self, x):
        # Pass input through all layers except the last (activation applied only to hidden layers)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # Output layer (no activation function)
        return self.layers[-1](x)
    
    def kl_divergence(self):
        # Sum KL divergence across all layers
        return sum(layer.kl_divergence() for layer in self.layers)
    
# Loss function
def gaussian_NLL(y_true, y_pred, sigma):
    """
    Computes the Negative Log-Likelihood (NLL) assuming a Gaussian distribution.

    Parameters:
    - y_true: Torch tensor of true values (N, ).
    - y_pred: Torch tensor of predicted values (N, ).
    - sigma: Torch tensor of standard deviations for predictions (N, ).

    Returns:
    - nll: Negative log-likelihood value (scalar).
    """
    # Input validation
    if not isinstance(y_true, torch.Tensor):
        raise TypeError("y_true must be a PyTorch tensor.")
    if not isinstance(y_pred, torch.Tensor):
        raise TypeError("y_pred must be a PyTorch tensor.")
    if not isinstance(sigma, torch.Tensor):
        raise TypeError("sigma must be a PyTorch tensor.")
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true and y_pred must have the same shape. Got {y_true.shape} and {y_pred.shape}.")
    if y_true.shape != sigma.shape:
        raise ValueError(f"Shape mismatch: y_true and sigma must have the same shape. Got {y_true.shape} and {sigma.shape}.")

    # Clipping sigma to avoid log(0) or division by zero
    epsilon = 1e-15
    sigma = torch.clamp(sigma, min=epsilon)

    # Compute NLL
    nll = (
        0.5 * torch.sum(((y_true - y_pred) ** 2) / (sigma ** 2)) + 
        torch.sum(torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi))))
    ) / y_true.size(0)

    return nll

def loss_function(output, sigma, target, kl_divergence, kl_weight=1.0):
    # Negative log likelihood (mean squared error as an example)
    log_likelihood = gaussian_NLL(output, target, sigma)
    
    # Total loss: log likelihood + KL divergence
    return log_likelihood + kl_weight * kl_divergence

def plot_and_save_figure(x_test, mean_pred, uncertainty, 
                         x_train, y_train, 
                         loss_total, epoch_list,
                         name='fig.png', epoch=None):
    """
    Creates a composite figure with:
      - Left column:
          * Top: Loss curves (KL, NLL, Total)
          * Bottom: Prediction plot with 95% CI and training points.
      - Right column: Three Monte Carlo KDE plots for zoomed-in values.
    
    The right column is set to be half as wide as the left column.
    
    Note: This function assumes that `model` and `n_samples` are available in the current scope.
    """
    
    # Define a gridspec with 3 rows and 2 columns.
    # Left column: width ratio 2, right column: width ratio 1.
    # We use 3 rows so that the loss plot (top left) takes 1 row and the confidence interval plot (bottom left)
    # spans the remaining 2 rows. The right column will have one KDE plot per row.
    fig = plt.figure(constrained_layout=True, figsize=(12, 9))
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[2, 1])
    
    # Left column subplots
    ax_loss = fig.add_subplot(gs[0, 0])         # Top left: Loss curves (smaller)
    ax_ci   = fig.add_subplot(gs[1:, 0])         # Bottom left: Confidence interval and prediction plot (larger)
    
    # Right column: three KDE plots (one per row)
    ax_kde = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    
    # --------------------------
    # Plot Loss Curves in ax_loss
    # --------------------------
    # ax_loss.plot(epoch, loss_kl, label='KL Loss')
    # ax_loss.plot(epoch, loss_nll, label='NLL Loss')
    ax_loss.plot(epoch_list, loss_total) #, label='Total Loss')
    ax_loss.set_yscale('log')
    # ax_loss.legend(fontsize=10)
    # ax_loss.set_title("Loss Curves", fontsize=12)
    ax_loss.set_xlabel("Epoch", fontsize=20, weight='bold')
    ax_loss.set_ylabel("Loss", fontsize=20, weight='bold')
    
    # ----------------------------------------------
    # Plot Confidence Interval & Predictions in ax_ci
    # ----------------------------------------------
    # Compute the uncertainty bounds
    lower_bound = mean_pred.detach().numpy().flatten() - 1.96 * uncertainty.detach().numpy().flatten()
    upper_bound = mean_pred.detach().numpy().flatten() + 1.96 * uncertainty.detach().numpy().flatten()
    x_test_np = x_test.numpy().flatten()
    
    # Fill between for uncertainty (95% CI)
    ax_ci.fill_between(x_test_np, lower_bound, upper_bound, color='lightblue', 
                       alpha=0.5, label='Uncertainty (95% CI)')
    
    # Plot mean prediction and its uncertainty borders
    ax_ci.plot(x_test_np, mean_pred.detach().numpy().flatten(), color='blue', 
             label='Mean Prediction', linewidth=2)
    ax_ci.plot(x_test_np, lower_bound, color='blue', linewidth=0.5)
    ax_ci.plot(x_test_np, upper_bound, color='blue', linewidth=0.5)
    
    # Plot training points
    ax_ci.scatter(x_train, y_train, c='red', marker='o', s=10, alpha=0.6, 
                  edgecolor='darkred', label='Training points')
    
    # For zoomed-in points, choose specific x-values.
    # Instead of directly indexing with floats, find the closest indices for the desired zoom values.
    zoom_values = [-0.6, -0.2, 1.2]
    indices = []
    for val in zoom_values:
        idx_val = (np.abs(x_test_np - val)).argmin()
        indices.append(idx_val)
    
    # Scatter these zoom-in points
    ax_ci.scatter(x_test_np[indices], mean_pred.detach().numpy().flatten()[indices], 
                  marker='d', c='orange', s=80, edgecolor='black', zorder=4)
    
    # Set axis ticks and limits
    ax_ci.set_xticks([-2, -1, 0, 1, 2])
    ax_ci.set_yticks([-1.5, -0.5, 0.5, 1.5, 2.5])
    ax_ci.set_xlim([-2, 2])
    ax_ci.set_ylim([-1.5, 2.0])
    ax_ci.set_xlabel('Input x', fontsize=20, labelpad=10, weight='bold')
    ax_ci.set_ylabel('Prediction', fontsize=20, labelpad=10, weight='bold')
    # if epoch is not None:
        # ax_ci.set_title(f'Training Epoch: {epoch}', fontsize=16, pad=10)
    
    ax_ci.legend(loc='upper center', fontsize=18, frameon=True)
    
    # -----------------------------------------
    # Plot Monte Carlo KDE plots in the right column
    # -----------------------------------------
    for i, val in enumerate(zoom_values):
        # Find the closest index to the desired zoom value
        idx_val = (np.abs(x_test_np - val)).argmin()
        x_val = x_test[idx_val]
        
        # Use the global model and n_samples to compute predictions at x_val
        with torch.no_grad():
            preds = torch.stack([model(x_val) for _ in range(n_samples)])
            
        # Determine the number of bins for the histogram
        range_val = preds.max() - preds.min()
        n_bins = int(17 * range_val.item())
        if n_bins < 1:
            n_bins = 10  # fallback if range is too small
        
        # Convert predictions to a DataFrame for seaborn
        preds_df = pd.DataFrame(preds.numpy().flatten(), columns=['Predictions'])
        
        # Plot histogram and KDE on the corresponding right-axis subplot
        sns.histplot(data=preds_df, y='Predictions', color='blue', bins=n_bins, 
                     kde=True, edgecolor="darkblue", alpha=0.3, linewidth=1, ax=ax_kde[i])
        
        ax_kde[i].set_title(f"x={float(x_val.detach().numpy()):.2f}", fontsize=20, pad=10, weight='bold')
        ax_kde[i].set_xlabel("Count", fontsize=20, labelpad=5, weight='bold')
        ax_kde[i].set_ylabel("Predictions", fontsize=20, labelpad=5, weight='bold')
        
        # Adjust x-ticks: here we set ticks at 0, half the max, and the max
        max_x = ax_kde[i].get_xlim()[1]
        ax_kde[i].set_xticks([0, max_x/2, max_x])
        ax_kde[i].set_yticks([-0.5, 0, 0.5, 1, 1.5])
        ax_kde[i].set_ylim([-0.5, 1.5])
    
    # Save the combined figure to file
    plt.savefig(name, dpi=300)
    plt.close(fig)

def predict_with_uncertainty(model, x, n_samples=1000):
    model.eval()
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n_samples)])
    mean_pred = preds.mean(0)
    # Variance across the samples. Uncomment to divide by the mean value
    uncertainty = preds.std(0)# /torch.abs(mean_pred)  
    # This implementation is far from being memory efficient; 
    # compute mean and standard deviation with an incremental algorithm when the number of samples and/or observations becomes too large
    return mean_pred, uncertainty

# %% Training
n_samples = 100
input_dim = 1
n_neurons = 32
n_layers  = 2
output_dim = 1
x_train = torch.rand(300, input_dim) -0.5
x_train = 1.5*x_train/((torch.max(x_train)-torch.min(x_train))/2)  # Scale between -1.5 and 1.5
x_train = x_train[(x_train>0.3)|(x_train<-0.1)]
x_train = x_train.reshape(len(x_train),1) # reshape column vector
y_train = -0.5 * np.sin(3*x_train) + x_train*0.2*torch.randn(*x_train.shape)  

# Model, optimizer
model = BayesianNN(input_dim, n_layers, n_neurons, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2500# 5000
kl_weight = 5e-8  # Weight for KL-divergence regularization

x_test = torch.Tensor([np.linspace(-2,2,201)]).t()

if not os.path.exists('frames/'):
    os.mkdir('frames')
    
loss_list = []
epoch_list = []

for epoch in range(num_epochs):

    # with torch.no_grad(): 
    #     y, sigma = predict_with_uncertainty(model, x_test)
    
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    # output = model(x_train)
    preds = torch.stack([model(x_train) for _ in range(n_samples)])
    mean_pred = preds.mean(0)
    uncertainty = preds.std(0)# /torch.abs(mean_pred)  # Variance across the samples divided by the output value
    
    # Compute KL divergence
    kl_div = model.kl_divergence()
    
    # Compute total loss
    loss = loss_function(mean_pred, uncertainty, y_train, kl_div, kl_weight)
    loss_list.append(loss.detach().numpy())
    epoch_list.append(epoch)
    
    # Backpropagation and optimization
    loss.backward()
    optimizer.step()
    
    # Plot and save figure
    if epoch % 20 == 0:
        with torch.no_grad(): 
            y, sigma = predict_with_uncertainty(model, x_test)
        
        plot_and_save_figure(x_test, y, sigma, 
                             x_train, y_train, 
                             loss_total=loss_list, 
                             epoch_list=epoch_list, 
                             name=f'frames/epoch{epoch:04d}', 
                             epoch=epoch)
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


        
from moviepy import ImageSequenceClip
import os

frame_dir = 'frames'
frame_files = sorted([
    os.path.join(frame_dir, f)
    for f in os.listdir(frame_dir) if f.endswith('.png')
])

clip = ImageSequenceClip(frame_files, fps=10)
clip.write_videofile("output_video.mp4", codec='libx264')
