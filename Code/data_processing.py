import numpy as np
from scipy.integrate import odeint
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *

def load_data():
    # Placeholder for data loading and preprocessing
    # Replace with actual data loading logic
    data = np.random.rand(100, 4)  # Example dummy data
    return data

def preprocess_mackey_glass(data):

    Y = data[:, 0]
    X = data[:, 1:]

    scaler = StandardScaler()
    X_scald = scaler.fit_transform(X)

    X_seq, Y_seq = create_sequences(X_scald, Y, SEQ_LEN)

    X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test


def generate_multidimensional_mackey_glass(num_samples, tau=TAU, delta_t=DELTA_T, num_attributes=NUM_ATTRIBUTE):
    data = np.zeros((num_samples, num_attributes))
    data[:tau] = np.random.rand(tau, num_attributes)

    beta = BETA
    gamma = GAMMA
    n = MACKEY_N

    for t in range(tau, num_samples):

        data[t, 0] = data[t - 1, 0] + (beta * data[t - tau, 0] / (1 + data[t - tau, 0] ** n) - gamma * data[t - 1, 0]) * delta_t

        data[t, 1] = data[t - 1, 1] + (0.25 * data[t - tau, 0] / (1 + data[t - tau, 0] ** n) - gamma * data[t - 1, 1]) * delta_t

        data[t, 2] = data[t - 1, 2] + (0.15 * data[t - tau, 1] / (1 + data[t - tau, 1] ** n) - gamma * data[t - 1, 2]) * delta_t + np.random.normal(0, 0.05)

        data[t, 3] = data[t - 1, 3] + (0.1 * data[t - tau, 2] / (1 + data[t - tau, 2] ** n) + 0.2 * data[t - 1, 3] - gamma * data[t - 1, 3]) * delta_t

    return data


def generate_lorenz96(N=LORENZ96_N, F=LORENZ96_F, x0=None, t_end=LORENZ96_T_END, t_points=LORENZ96_T_POINTS):
    """
    Generates data using the Lorenz 96 model.

    Parameters:
    - N (int): Number of variables (dimension of the system)
    - F (float): Forcing term
    - x0 (np.array or None): Initial state, if None defaults to equilibrium with a small perturbation
    - t_end (float): End time for integration
    - t_points (int): Number of time points to integrate

    Returns:
    - t (np.array): Time vector
    - lorenz96_data (np.array): Integrated Lorenz 96 data
    """

    def L96(x, t):
        """Lorenz 96 model with constant forcing."""
        d = np.zeros(N)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    # Set initial state if not provided
    if x0 is None:
        x0 = F * np.ones(N)
        x0[0] += 0.01  # Small perturbation to the first variable

    # Time vector
    t = np.linspace(0, t_end, t_points)

    # Integrate the Lorenz 96 model
    lorenz96_data = odeint(L96, x0, t)

    return t, lorenz96_data

def preprocess_lorenz96(data):

    X = data[:, 1:]
    Y = data[:, 0]

    scaler = StandardScaler()
    X_scald = scaler.fit_transform(X)

    X_seq, Y_seq = create_sequences(X_scald, Y, SEQ_LEN)

    X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test


def create_sliding_window(data, sequence_length=4):
    X = []
    Y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        Y.append(
            data[i + 1 : i + sequence_length + 1]
        )
    return np.array(X), np.array(Y)


def preprocess_dream3():
    expression_data_path = "../Data/dream3/DREAM3_GeneExpressionChallenge_ExpressionData_UPDATED.txt"
    target_list_path = "../Data/dream3/DREAM3_GeneExpressionChallenge_TargetList.txt"

    expression_df = pd.read_csv(expression_data_path, sep="\t")
    target_list_df = pd.read_csv(target_list_path, sep="\t")

    target_probe_ids = target_list_df["probeID"].dropna().tolist()

    non_target_data = expression_df[~expression_df["ProbeID"].isin(target_probe_ids)]

    non_target_data = non_target_data[0:30]
    non_target_data = non_target_data.iloc[:, 3:]

    non_target_data_array = non_target_data.values
    reshaped_data = non_target_data_array.reshape(non_target_data_array.shape[0], 8, 4).transpose(0, 2, 1).reshape(-1, 4)
    non_target_data = pd.DataFrame(reshaped_data, columns=["wt", "gat1", "gcn4", "leu3"])

    feature_columns = [col for col in non_target_data.columns if any(cond in col for cond in ["wt", "gcn4", "leu3"])]
    target_columns = [col for col in non_target_data.columns if "gat1" in col]

    X = non_target_data[feature_columns].dropna().values
    Y = non_target_data[target_columns].dropna().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq = X_scaled.reshape(-1, 8, 3)
    Y_seq = Y.reshape(-1, 8, 1)

    Y_seq = np.where(Y_seq == "PREDICT", np.nan, Y_seq)  # 替换无效值
    Y_seq = Y_seq.astype(float)  # 转换为 float 类型
    Y_seq = np.nan_to_num(Y_seq)  # 将 NaN 替换为 0 或其他默认值

    X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.3, random_state=42)

    return X_train, X_test, Y_train, Y_test

def preprocess_household():
    file_path = "../Data/household_power_consumption.txt"
    data = pd.read_csv(file_path, sep=";", na_values=["?", "NA"])

    # Clean the data, change to Datetime
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

    # Delete Date and Time column
    data.drop(['Date', 'Time'], axis=1, inplace=True)

    # DropNa
    data.dropna(inplace=True)

    feature_columns = [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]
    target_column = "Global_active_power"

    data = data[0:100]

    X = data[feature_columns].values
    Y = data[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, Y_seq = create_sequences(X_scaled, Y, SEQ_LEN)

    X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test


def create_sequences(features, targets, seq_len):
    X_seq, Y_seq = [], []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i:i + seq_len])
        Y_seq.append(targets[i:i + seq_len].reshape(-1, 1))
    return np.array(X_seq), np.array(Y_seq)