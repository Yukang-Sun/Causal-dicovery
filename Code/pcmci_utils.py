import numpy as np
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.gpdc import GPDC
from tigramite import data_processing as pp
from config import *


def run_pcmci(data):
    if not isinstance(data, np.ndarray):
        data_np = data.values  # Convert to ndarray
    else:
        data_np = data


    # Initialize Tigramite DataFrame
    pp_data = pp.DataFrame(data_np)
    # GDPC: linear or simply non-linear data; CMIknn: complicated non-linear data but expensive
    if PCMCI_METHOD == 'GPDC':
        gpdc = GPDC(significance=SIGNIFICANCE_METHOD)  # analytic or 'shuffle_test'/'bootstrap'
        pcmci = PCMCI(dataframe=pp_data, cond_ind_test=gpdc, verbosity=PC_VERBOSITY)
    elif PCMCI_METHOD == 'CMIknn':
        cmiknn = CMIknn(significance=SIGNIFICANCE_METHOD)
        pcmci = PCMCI(dataframe=pp_data, cond_ind_test=cmiknn, verbosity=PC_VERBOSITY)
    results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA_LEVEL)
    return results

def extract_submatrix(results):
    """
    Extract a submatrix from the causal matrix for specific variables.

    Parameters:
    - results: Output from PCMCI, e.g., results['val_matrix']
    - var_names: List of all variable names
    - selected_vars: List of variables to extract, e.g., ['b', 'c', 'd']

    Returns:
    - submatrix: The extracted submatrix for the selected variables
    """
    val_matrix = results['val_matrix']

    # Get indices of the selected variables
    # indices = [var_names.index(var) for var in selected_vars]


    # Extract the submatrix
    submatrix = val_matrix[:-1, :-1]
    return submatrix

