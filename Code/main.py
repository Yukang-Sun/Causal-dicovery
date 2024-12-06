import numpy as np

from data_processing import preprocess_mackey_glass, generate_multidimensional_mackey_glass, generate_lorenz96, preprocess_household, preprocess_dream3,preprocess_lorenz96
from train import train_model
from pcmci_utils import run_pcmci, extract_submatrix
from config import *
import torch
import pandas as pd

def main():
    if DATASET_NAME == "Mackey_glass":
        data = generate_multidimensional_mackey_glass(DATA_SIZE)
        X_train, X_test, Y_train, Y_test = preprocess_mackey_glass(data)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


        train_loader = torch.utils.data.DataLoader(

            dataset=torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor),

            batch_size=BATCH_SIZE,

            shuffle=True,

        )

        test_loader = torch.utils.data.DataLoader(

            dataset=torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor),

            batch_size=BATCH_SIZE,

            shuffle=False,

        )

        print("Start for First Transformer")

        # ===================

        # 1. Train Transformer

        # ===================

        PCMCI_matrix = None  # No causal matrix for the initial round

        transformer_model = train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)

        # ==========================

        # 2. Predict with Transformer

        # ==========================

        predicted_output, _ = transformer_model(X_train_tensor)  # Get predicted output
        predicted_output_df = pd.DataFrame(predicted_output.detach().numpy().reshape(-1, predicted_output.shape[-1]))
        X_train_df = pd.DataFrame(X_train.reshape(-1, X_train.shape[-1]))
        # X_train_df = pd.DataFrame(X_train_df)

        unique_indices = list(range(SEQ_LEN - 1, X_train_df.shape[0], SEQ_LEN))
        X_train_df = X_train_df.iloc[unique_indices]
        predicted_output_df = predicted_output_df.iloc[unique_indices]

        combined_train = pd.concat([X_train_df, predicted_output_df], axis=1)

        # combined_train = torch.cat((X_train_tensor, predicted_output), dim=-1)

        print("Start for PCMCI")

        pcmci_results = run_pcmci(combined_train)

        # ===========================

        # 4. Retrain Transformer with PCMCI Matrix

        # ===========================

        PCMCI_matrix = pcmci_results['val_matrix']

        print("Start for Second Transformer")

        train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)

        print("End for Training")

    elif DATASET_NAME == "Lorenz96":
        _, data = generate_lorenz96()
        X_train, X_test, Y_train, Y_test = preprocess_lorenz96(data)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


        train_loader = torch.utils.data.DataLoader(

            dataset=torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor),

            batch_size=BATCH_SIZE,

            shuffle=True,

        )

        test_loader = torch.utils.data.DataLoader(

            dataset=torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor),

            batch_size=BATCH_SIZE,

            shuffle=False,

        )

        print("Start for First Transformer")

        # ===================

        # 1. Train Transformer

        # ===================

        PCMCI_matrix = None  # No causal matrix for the initial round

        transformer_model = train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)

        # ==========================

        # 2. Predict with Transformer

        # ==========================

        predicted_output, _ = transformer_model(X_train_tensor)  # Get predicted output
        predicted_output_df = pd.DataFrame(predicted_output.detach().numpy().reshape(-1, predicted_output.shape[-1]))
        X_train_df = pd.DataFrame(X_train.reshape(-1, X_train.shape[-1]))

        unique_indices = list(range(SEQ_LEN - 1, X_train_df.shape[0], SEQ_LEN))
        X_train_df = X_train_df.iloc[unique_indices]
        predicted_output_df = predicted_output_df.iloc[unique_indices]

        combined_train = pd.concat([X_train_df, predicted_output_df], axis=1)


        print("Start for PCMCI")

        pcmci_results = run_pcmci(combined_train)

        # ===========================

        # 4. Retrain Transformer with PCMCI Matrix

        # ===========================

        PCMCI_matrix = pcmci_results['val_matrix']

        print("Start for Second Transformer")

        train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)

        print("End for Training")

    elif DATASET_NAME == "Dream3":
        seq_len = 8  # Number of time steps
        X_train, X_test, Y_train, Y_test = preprocess_dream3()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        print("Start for First Transformer")
        # ===================
        # 1. Train Transformer
        # ===================
        # Initial training of Transformer without PCMCI regularization
        PCMCI_matrix = None  # No causal matrix for the initial round
        transformer_model = train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)

        # ==========================
        # 2. Predict with Transformer
        # ==========================
        # Use the trained Transformer to predict `gat1` on the training data
        # transformer_model = train_model.model  # Retrieve the trained Transformer model
        predicted_output, _ = transformer_model(X_train_tensor)  # Get predicted output
        predicted_gat1_train = predicted_output.detach().numpy()  # Convert to NumPy

        # Remove the last dimension to make it 2D
        # predicted_gat1_train = predicted_gat1_train.squeeze(-1)

        # Create a DataFrame for predicted values
        # predicted_gat1_train_df = pd.DataFrame(predicted_gat1_train, columns=Y_train.columns, index=Y_train.index)


        # ========================
        # 3. Combine data for PCMCI
        # ========================
        # Merge `wt`, `gcn4`, `leu3` features with the predicted `gat1`
        # X_train_df = pd.DataFrame(X_train.reshape(-1, 3))
        combined_data = np.concatenate((predicted_gat1_train, X_train), axis=-1)
        # combined_train = pd.concat([X_train, predicted_gat1_train_df], axis=1)

        # Run PCMCI causal discovery
        print("Start for First PCMCI")
        global_matrix = np.zeros((4, 4))
        for gene_index in range(combined_data.shape[0]):
            gene_matrix = combined_data[gene_index]
            pcmci_results_matrix = run_pcmci(gene_matrix)['val_matrix']
            global_matrix += pcmci_results_matrix
        PCMCI_matrix = global_matrix / combined_data.shape[0]
        # ===========================
        # 4. Retrain Transformer with PCMCI Matrix
        # ===========================
        # PCMCI_matrix = pcmci_results['val_matrix']
        print("Start for Second Transformer")
        train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)
        print("End for Training")


    elif DATASET_NAME == "Housing_consumption":

        X_train, X_test, Y_train, Y_test = preprocess_household()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


        train_loader = torch.utils.data.DataLoader(

            dataset=torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor),

            batch_size=BATCH_SIZE,

            shuffle=True,

        )

        test_loader = torch.utils.data.DataLoader(

            dataset=torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor),

            batch_size=BATCH_SIZE,

            shuffle=False,

        )

        print("Start for First Transformer")

        # ===================

        # 1. Train Transformer

        # ===================

        PCMCI_matrix = None  # No causal matrix for the initial round

        transformer_model = train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)

        # ==========================

        # 2. Predict with Transformer

        # ==========================

        predicted_output, _ = transformer_model(X_train_tensor)  # Get predicted output
        predicted_output_df = pd.DataFrame(predicted_output.detach().numpy().reshape(-1, predicted_output.shape[-1]))
        X_train_df = pd.DataFrame(X_train.reshape(-1, X_train.shape[-1]))
        # X_train_df = pd.DataFrame(X_train_df)

        unique_indices = list(range(SEQ_LEN-1, X_train_df.shape[0], SEQ_LEN))
        X_train_df = X_train_df.iloc[unique_indices]
        predicted_output_df = predicted_output_df.iloc[unique_indices]

        combined_train = pd.concat([X_train_df, predicted_output_df], axis=1)

        # var_name = combined_train.columns.tolist()
        # combined_train = torch.cat((X_train_tensor, predicted_output), dim=-1)
        # X_train_var = pd.DataFrame(X_train).columns.tolist()

        print("Start for PCMCI")

        pcmci_results = run_pcmci(combined_train)

        # ===========================

        # 4. Retrain Transformer with PCMCI Matrix

        # ===========================

        PCMCI_matrix = extract_submatrix(pcmci_results)



        print("Start for Second Transformer")

        train_model(train_loader, num_epochs=NUM_EPOCHS, PCMCI_matrix=PCMCI_matrix)

        print("End for Training")


if __name__ == "__main__":
    main()
