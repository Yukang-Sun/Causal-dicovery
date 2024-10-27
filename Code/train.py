
import torch
import torch.optim as optim
import torch.nn as nn
from model import Transformer
import numpy as np
from pcmci_utils import run_pcmci
import torch.nn.functional as F

def train_model(train_loader, num_epochs, PCMCI_matrix=None):
    d_model = 128
    nhead = 4
    num_layers = 3
    dim_feedforward = 512
    input_dim = 4
    output_dim = 4
    dilation = 2
    dropout = 0.1

    model = Transformer(num_layers, d_model, nhead, dim_feedforward, input_dim, output_dim, dilation, dropout)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    def regularization_loss(attention_matrix, PCMCI_matrix):
        if PCMCI_matrix is None:
            return 0  # No regularization if PCMCI result is not available

        # Average all attention heads (batch_size, seq_len, seq_len)
        avg_attention_matrix = attention_weights.mean(dim=1)

        # If tau_weights are not provided, use uniform weighting
        tau_time = PCMCI_matrix.shape[0]  # Get the number of lag steps
        tau_weights = None
        if tau_weights is None:
            tau_weights = [1 / tau_time] * tau_time  # Default to uniform weighting

        # Initialize total loss
        total_loss = 0

        # Iterate over each lag step and calculate the difference between the attention matrix and PCMCI matrix
        for tau in range(tau_time):
            PCMCI_matrix_tau = PCMCI_matrix[tau]  # Select the causal matrix at lag tau

            # If the attention matrix and PCMCI matrix sizes do not match, adjust attention matrix size
            num_vars = PCMCI_matrix_tau.shape[0]
            if avg_attention_matrix.shape[1] > num_vars:
                attention_submatrix = avg_attention_matrix[:, :num_vars, :num_vars]
            elif avg_attention_matrix.shape[1] < num_vars:
                attention_submatrix = F.interpolate(avg_attention_matrix, size=(num_vars, num_vars), mode='bilinear')
            else:
                attention_submatrix = avg_attention_matrix

            # If PCMCI matrix is a numpy array, convert it to torch.Tensor
            if isinstance(PCMCI_matrix_tau, np.ndarray):
                PCMCI_matrix_tau = torch.tensor(PCMCI_matrix_tau, dtype=torch.float32, device=attention_matrix.device)

            # Calculate regularization loss for this lag step
            loss_tau = torch.norm(attention_submatrix - PCMCI_matrix_tau)
            total_loss += tau_weights[tau] * loss_tau  # Weight by tau_weights

        return total_loss

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            predicted_output, attention_weights = model(batch_X)
            mse_loss = loss_fn(predicted_output, batch_Y)
            attention_loss = regularization_loss(attention_weights.mean(dim=1), PCMCI_matrix)
            lambda_reg = 0.1
            loss = mse_loss + lambda_reg * attention_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    print("Training complete!")
