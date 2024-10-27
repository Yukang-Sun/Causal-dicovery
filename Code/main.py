
from data_processing import load_data, preprocess_data, generate_multidimensional_mackey_glass, generate_lorenz96, load_housing_consumption
from train import train_model
from pcmci_utils import run_pcmci
import torch

def main():
    data = generate_multidimensional_mackey_glass(100)
    X_tensor, Y_tensor = preprocess_data(data)

    # PCMCI causal discovery on the processed data
    pcmci_results = run_pcmci(data)

    # Training the model (initially without PCMCI results)
    PCMCI_matrix = None  # Use None for the first round of training
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_tensor, Y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )  # Placeholder: You need to create a DataLoader or any iterable data structure

    train_model(train_loader, num_epochs=10, PCMCI_matrix=PCMCI_matrix)

    # Optionally, add PCMCI results after first epoch
    PCMCI_matrix = pcmci_results['val_matrix']  # Example: results from PCMCI
    train_model(train_loader, num_epochs=10, PCMCI_matrix=PCMCI_matrix)

if __name__ == "__main__":
    main()
