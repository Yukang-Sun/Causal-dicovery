
import torch
import torch.optim as optim
import torch.nn as nn
from model import Transformer
import numpy as np
from pcmci_utils import run_pcmci
import torch.nn.functional as F
from config import *

def train_model(train_loader, num_epochs, PCMCI_matrix=None):
    example_batch = next(iter(train_loader))
    input_dim = example_batch[0].shape[-1]
    output_dim = example_batch[1].shape[-1]

    if DATASET_NAME == 'Dream3':
        model = Transformer(
            num_layers=TRANSFORMER_CONFIG["DREAM3"]["num_layers"],
            d_model=TRANSFORMER_CONFIG["DREAM3"]["d_model"],
            nhead=TRANSFORMER_CONFIG["DREAM3"]["nhead"],
            dim_feedforward=TRANSFORMER_CONFIG["DREAM3"]["dim_feedforward"],
            input_dim=TRANSFORMER_CONFIG["DREAM3"]["input_dim"],  # 3
            output_dim=TRANSFORMER_CONFIG["DREAM3"]["output_dim"],  # 8
            dilation=TRANSFORMER_CONFIG["DREAM3"]["dilation"],
            dropout=TRANSFORMER_CONFIG["DREAM3"]["dropout"],
        )
    elif DATASET_NAME == 'Housing_consumption':
        model = Transformer(
            num_layers=TRANSFORMER_CONFIG["Household"]["num_layers"],
            d_model=TRANSFORMER_CONFIG["Household"]["d_model"],
            nhead=TRANSFORMER_CONFIG["Household"]["nhead"],
            dim_feedforward=TRANSFORMER_CONFIG["Household"]["dim_feedforward"],
            input_dim=TRANSFORMER_CONFIG["Household"]["input_dim"],  # 3
            output_dim=TRANSFORMER_CONFIG["Household"]["output_dim"],  # 8
            dilation=TRANSFORMER_CONFIG["Household"]["dilation"],
            dropout=TRANSFORMER_CONFIG["Household"]["dropout"],)

    elif DATASET_NAME == 'Mackey_glass':
        model = Transformer(
            num_layers=TRANSFORMER_CONFIG["Mackey_glass"]["num_layers"],
            d_model=TRANSFORMER_CONFIG["Mackey_glass"]["d_model"],
            nhead=TRANSFORMER_CONFIG["Mackey_glass"]["nhead"],
            dim_feedforward=TRANSFORMER_CONFIG["Mackey_glass"]["dim_feedforward"],
            input_dim=TRANSFORMER_CONFIG["Mackey_glass"]["input_dim"],  # 3
            output_dim=TRANSFORMER_CONFIG["Mackey_glass"]["output_dim"],  # 8
            dilation=TRANSFORMER_CONFIG["Mackey_glass"]["dilation"],
            dropout=TRANSFORMER_CONFIG["Mackey_glass"]["dropout"],)

    elif DATASET_NAME == 'Lorenz96':
        model = Transformer(
            num_layers=TRANSFORMER_CONFIG["Lorenz96"]["num_layers"],
            d_model=TRANSFORMER_CONFIG["Lorenz96"]["d_model"],
            nhead=TRANSFORMER_CONFIG["Lorenz96"]["nhead"],
            dim_feedforward=TRANSFORMER_CONFIG["Lorenz96"]["dim_feedforward"],
            input_dim=TRANSFORMER_CONFIG["Lorenz96"]["input_dim"],  # 3
            output_dim=TRANSFORMER_CONFIG["Lorenz96"]["output_dim"],  # 8
            dilation=TRANSFORMER_CONFIG["Lorenz96"]["dilation"],
            dropout=TRANSFORMER_CONFIG["Lorenz96"]["dropout"],)

    # model = Transformer(NUM_LAYERS, D_MODEL, NHEAD, DIM_FEEDFORWARD, input_dim, output_dim, DILATION, DROPOUT_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    def regularization_loss(attention_matrix, PCMCI_matrix):
        if PCMCI_matrix is None:
            return 0  # No regularization if PCMCI result is not available
        # Attention Matrix shape : [batch_size, n_head, seq_len, seq_len]
        avg_attention_matrix = attention_matrix.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]
        tau_time = PCMCI_matrix.shape[2]  # Number of lag steps
        tau_weights = [1 / tau_time] * tau_time  # Uniform weighting

        total_loss = 0


        for tau in range(tau_time):
            PCMCI_matrix_tau = PCMCI_matrix[:, :, tau]  # Select the causal matrix for lag tau
            num_vars = PCMCI_matrix_tau.shape[0]

            # Adjust attention matrix size to match PCMCI matrix
            if avg_attention_matrix.shape[1] > num_vars:
                attention_submatrix = avg_attention_matrix[:, :num_vars, :num_vars]
                attention_submatrix = attention_submatrix.mean(dim=0)
            elif avg_attention_matrix.shape[1] < num_vars:
                # Add batch and channel dimensions for interpolation
                avg_attention_matrix_reshaped = avg_attention_matrix.unsqueeze(1)
                attention_submatrix = F.interpolate(
                    avg_attention_matrix_reshaped, size=(num_vars, num_vars), mode="bilinear"
                ).squeeze(1)
                attention_submatrix = attention_submatrix.mean(dim=0)
            else:
                attention_submatrix = avg_attention_matrix
                attention_submatrix = attention_submatrix.mean(dim=0)

            # Convert PCMCI matrix to tensor if needed
            if isinstance(PCMCI_matrix_tau, np.ndarray):
                PCMCI_matrix_tau = torch.tensor(
                    PCMCI_matrix_tau, dtype=torch.float32, device=attention_matrix.device
                )

            # Calculate the Frobenius norm difference
            loss_tau = torch.norm(attention_submatrix - PCMCI_matrix_tau)
            total_loss += tau_weights[tau] * loss_tau

        return total_loss

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            # Ensure target dimensions match the model output

            predicted_output, attention_weights = model(batch_X)
            # predicted_output = predicted_output[:, -1, :]
            mse_loss = loss_fn(predicted_output, batch_Y)
            attention_loss = regularization_loss(attention_weights, PCMCI_matrix)

            loss = mse_loss + LAMBDA_REG * attention_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    print("Training complete!")
    return model