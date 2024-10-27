import numpy as np
from scipy.integrate import odeint
import pandas as pd
import torch


def load_data():
    # Placeholder for data loading and preprocessing
    # Replace with actual data loading logic
    data = np.random.rand(100, 4)  # Example dummy data
    return data

def preprocess_data(data):
    # Placeholder for data preprocessing
    # Replace with actual preprocessing logic
    sequence_length = 4
    X, Y = create_sliding_window(data, sequence_length)
    # X, Y = create_sliding_window(mackey_data, sequence_length)
    # 归一化数据到 [0, 1]
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    return X_tensor, Y_tensor


def generate_multidimensional_mackey_glass(num_samples, tau=17, delta_t=0.1, num_attributes=4):
    # 初始化数据矩阵，维度为 (num_samples, num_attributes)
    data = np.zeros((num_samples, num_attributes))

    # 随机初始化前几个时间步的数据
    data[:tau] = np.random.rand(tau, num_attributes)

    # 定义系统的参数
    beta = 0.2
    gamma = 0.1
    n = 10

    # 按照因果关系生成后续样本
    for t in range(tau, num_samples):
        # 变量A的变化
        data[t, 0] = data[t - 1, 0] + (beta * data[t - tau, 0] / (1 + data[t - tau, 0] ** n) - gamma * data[t - 1, 0]) * delta_t

        # 变量B的变化，受A的滞后值以及B自身的影响
        data[t, 1] = data[t - 1, 1] + (0.25 * data[t - tau, 0] / (1 + data[t - tau, 0] ** n) - gamma * data[t - 1, 1]) * delta_t

        # 变量C的变化，受B的滞后值和随机噪声的影响
        data[t, 2] = data[t - 1, 2] + (0.15 * data[t - tau, 1] / (1 + data[t - tau, 1] ** n) - gamma * data[t - 1, 2]) * delta_t + np.random.normal(0, 0.05)

        # 变量D的变化，受C和D的延迟值的共同影响
        data[t, 3] = data[t - 1, 3] + (0.1 * data[t - tau, 2] / (1 + data[t - tau, 2] ** n) + 0.2 * data[t - 1, 3] - gamma * data[t - 1, 3]) * delta_t

    return data


def generate_lorenz96(N=4, F=10, x0=None, t_end=10, t_points=1000):
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


def load_housing_consumption():
    data = pd.read_csv("household_power_consumption.txt", sep=';', parse_dates={'Datetime': ['Date', 'Time']},
                na_values='?', low_memory=False)
    hs_data = data.dropna()

    return hs_data

def create_sliding_window(data, sequence_length=4):
    X = []
    Y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        Y.append(
            data[i + 1 : i + sequence_length + 1]
        )
    return np.array(X), np.array(Y)