import os
import json
import pandas as pd
from pathlib import Path
import csv
from qvarnet import load_model_from_results
import matplotlib.pyplot as plt
import numpy as np


def load_simulation_results(root_dir):
    all_runs = []

    # 1. Setup FileSystem
    files = list(Path(root_dir).rglob("metrics.json"))

    # 2. Process Files
    for file_path in files:
        str_path = str(file_path)
        folder = os.path.dirname(str_path)
        config_path = os.path.join(folder, "config.json")

        # Helper to handle local
        def open_file(p):
            return open(p, "r")

        try:
            with open_file(str_path) as f:
                metrics = json.load(f)

            config = {}
            exists = os.path.exists(config_path)

            if exists:
                with open_file(config_path) as f:
                    config = json.load(f)

            all_runs.append({**config, **metrics, "path": folder})
        except Exception as e:
            print(f"Skipping {folder} due to error: {e}")

    df = pd.json_normalize(all_runs, sep="_")
    return df


def separate_input_dim_and_hidden_dim(df):
    # Create new columns for input_dim and hidden_dim
    df["input_dim"] = df["model_architecture"].apply(lambda x: int(x[0]))
    df["hidden_dim"] = df["model_architecture"].apply(lambda x: x[1:-1])
    df.drop(columns=["model_architecture"], inplace=True)
    return df


def add_energy_per_particle(df):
    df["energy_per_particle"] = df["total_energy"] / df["input_dim"]
    df["std_energy_per_particle"] = df["std"] / df["input_dim"]
    df["best_score_per_particle"] = df["best_score"] / df["input_dim"]
    return df


def clean_non_informative_columns(df):
    for column in df.columns:
        if df[column].dtype == list:
            df[column] = df[column].apply(lambda x: str(x))
        if df.groupby(column).ngroups > 1:
            # print(f"Column '{column}' has {df.groupby(column).size().nunique()} unique group sizes.")
            continue
        else:
            print(f"Column '{column}' has only 1 unique group size, dropping.")
            df = df.drop(columns=[column])
    return df


def get_energy_history(path):
    # path is found in the path column of the dataframe, and it is the path to the folder containing the energy_history.csv file
    print(f"Reading energy history from: {path}")

    with open(os.path.join(path, "energy_history.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        energy_history = [float(row[1]) for row in reader]
    return energy_history


def load_model_from_path(path):
    model, jax_params, input_dim = load_model_from_results(path)
    return model, jax_params, input_dim


def plot_wavefunction_one_particle(
    model, params, in_dim, x_range=(-5, 5), num_points=200
):
    x_vec = np.linspace(x_range[0], x_range[1], num_points)

    psi_values_list = []

    for i in range(in_dim):
        # 1. Create a base matrix of zeros: Shape (100, DoF)
        coords = np.zeros((num_points, in_dim))

        # 2. Inject the 'x' range into the i-th column
        coords[:, i] = x_vec

        # 3. Forward pass through your VMC model
        psi_values = model.apply(params, coords)
        psi_values_list.append(psi_values)

    psi_values_array = np.array(psi_values_list)  # Shape: (DoF, num_points)
    psi_values_mean = np.mean(psi_values_array, axis=0).squeeze()  # Average over DoF
    psi_values_std = np.std(
        psi_values_array, axis=0
    ).squeeze()  # Std deviation over DoF
    print("Shape of psi_values_array:", psi_values_array.shape)
    print("Shape of psi_values_mean:", psi_values_mean.shape)
    print("Shape of psi_values_std:", psi_values_std.shape)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vec, psi_values_mean, label="Mean Wavefunction")
    plt.fill_between(
        x_vec,
        psi_values_mean - psi_values_std,
        psi_values_mean + psi_values_std,
        alpha=0.3,
        label="Std Dev",
    )
    plt.title("Wavefunction vs Position for One Particle")
    plt.xlabel("Position (x)")
    plt.ylabel("Wavefunction (ψ)")
    plt.legend()
    plt.grid()
    plt.show()


def localize_element_in_dataframe(
    df,
    input_dim=None,
    hidden_dims=None,
    lr=None,
    chain_length=None,
    step_size=None,
    n_chains=None,
):
    """
    Localizes the row in the DataFrame that corresponds to the given parameters.

    Parameters:
    - df: The DataFrame containing the simulation results.
    - input_dim: The input dimension to search for.
    - hidden_dims: The hidden dimensions to search for (as a tuple).
    - lr: The learning rate to search for.
    - chain_length: The chain length to search for.
    - step_size: The step size to search for.
    - n_chains: The number of chains to search for.

    Returns:
    - A DataFrame row corresponding to the given parameters, or None if not found.
    """
    # Filter the DataFrame based on the provided parameters

    filtered_df = df.copy()
    for param, value in zip(
        [
            "input_dim",
            "hidden_dim",
            "optimizer_learning_rate",
            "sampler_chain_length",
            "sampler_step_size",
            "training_batch_size",
        ],
        [input_dim, hidden_dims, lr, chain_length, step_size, n_chains],
    ):
        if value is not None:
            print(f"Filtering for {param} = {value}")
            filtered_df = filtered_df[filtered_df[param] == value]
        else:
            print(f"No filter applied for {param}")

    if filtered_df.empty:
        print("No matching row found.")
        return None
    else:
        print(f"Found {len(filtered_df)} matching rows. Returning all of them.")
        return filtered_df
