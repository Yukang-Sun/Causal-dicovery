
# PCMCI Transformer Causal Discovery Project

This project focuses on performing causal discovery using a Transformer model combined with PCMCI (Peter and Clark Momentary Conditional Independence) analysis.

## Project Structure

The project is organized into the following files:

1. **data_processing.py**: Contains functions to load and preprocess data.
2. **model.py**: Defines the Transformer model used for causal discovery.
3. **train.py**: Contains the training loop for the model, including the loss function that integrates PCMCI results.
4. **pcmci_utils.py**: Provides utility functions for running PCMCI analysis using the `CMIknn` method.
5. **main.py**: The main script that ties everything together and runs the data processing, PCMCI analysis, and model training.

## Data 

1. **Lorenz96 and Mackey_glass**
2. **household_power_consumption**: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption 
3. **Dream3**: https://www.synapse.org/Synapse:syn3033083/wiki/74370


## How to Run the Project

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the main Python script:

```bash
python main.py
```

## PCMCI Methods

- **GPDC**: Used for linear or easily non-linear data
- **CMIknn**: Used for non-linear data analysis, leveraging k-nearest neighbors to estimate conditional mutual information.

## PCMCI Parameter
- **Significance Testing**: Different significance testing methods are available, such as `analytic` (fast, for simple data) and `shuffle_test` (slower but more accurate for complex data).

## Dependencies
```bash
pip install -r requirements.txt
```



