# utils.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt




def load_params(json_path="params.json"):
    """
    Load parameters from a JSON file and return them as a dictionary.
    """
    with open(json_path, "r") as f:
        params = json.load(f)
    
    return params


def load_scenario_data(scenario_idx, base_path, num_snps, replicate_idx):
    file_path = f"{base_path}/scenario_{scenario_idx}/ts_{scenario_idx}_{replicate_idx}.npz"
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        return None, None
    snp_data = data["SNP"]
    pos_data = data["POS"]
    if snp_data.shape[1] < num_snps:
        return None, None
    return snp_data[:, :num_snps], pos_data[:num_snps]


def data_loading(params):
    parameters_df = pd.read_csv(params["parameters_csv"])
    rg_sc_train = range(*params["rg_sc_train"])
    rg_sc_test = range(*params["rg_sc_test"])

    X_train, Y_train, X_test, Y_test = [], [], [], []
    P_train, P_test = [], []
    scenarios_train, scenarios_test = [], []

    for scenario_idx in tqdm(rg_sc_train, desc="Training Scenarios"):
        for replicate_idx in range(params["num_replicates"]):
            X_data, P_data = load_scenario_data(scenario_idx, params["train_base_path"], params["num_snps"], replicate_idx)
            if X_data is None:
                continue
            X_population1 = X_data[:params["nb_gen"], :, np.newaxis]
            X_population2 = X_data[params["nb_gen"]:, :, np.newaxis]
            X_train.append([X_population1, X_population2])
            P_train.append(P_data[:, np.newaxis])
            Y_train.append(parameters_df.loc[scenario_idx, params["target"]].values)
            scenarios_train.append(scenario_idx)

    for scenario_idx in tqdm(rg_sc_test, desc="Test Scenarios"):
        for replicate_idx in range(params["num_replicates"]):
            X_data, P_data = load_scenario_data(scenario_idx, params["test_base_path"], params["num_snps"], replicate_idx)
            if X_data is None:
                continue
            X_population1 = X_data[:params["nb_gen"], :, np.newaxis]
            X_population2 = X_data[params["nb_gen"]:, :, np.newaxis]
            X_test.append([X_population1, X_population2])
            P_test.append(P_data[:, np.newaxis])
            Y_test.append(parameters_df.loc[scenario_idx, params["target"]].values)
            scenarios_test.append(scenario_idx)

    X_train, X_test = np.array(X_train), np.array(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)
    P_train, P_test = np.array(P_train), np.array(P_test)

    X_train = (X_train - X_train.mean(axis=(0, 1), keepdims=True)) / X_train.std(axis=(0, 1), keepdims=True)
    X_test = (X_test - X_test.mean(axis=(0, 1), keepdims=True)) / X_test.std(axis=(0, 1), keepdims=True)

    P_train = (P_train - P_train.mean()) / P_train.std()
    P_test = (P_test - P_test.mean()) / P_test.std()

    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)

    P_train = P_train[:, np.newaxis, :, np.newaxis]
    P_test = P_test[:, np.newaxis, :, np.newaxis]

    return X_train, Y_train_scaled, X_test, Y_test_scaled, P_train, P_test, scenarios_train, scenarios_test, scaler_Y

def save_training_plots(history_path, output_dir):
    # Load training history
    with open(history_path, "r") as f:
        history = json.load(f)

    # Create figure
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history["mae"], label="Training MAE")
    if "val_mae" in history:
        plt.plot(history["val_mae"], label="Validation MAE")
    plt.title("MAE over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()

    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()


def saving_conv(model, history, 
                Y_train_scaled, Y_test_scaled, 
                Y_train_pred, Y_test_pred,
                scenarios_train, scenarios_test, 
                output_dir):
    
    output_dir = os.path.join(output_dir, "replicates")
    os.makedirs(output_dir, exist_ok=True)

    model_save_path = os.path.join(output_dir, "model.keras")  # Change extension to .keras
    model.save(model_save_path)

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)

    train_results = pd.DataFrame({
        'scenario_train': scenarios_train,
        'Y_train_scaled': list(Y_train_scaled), 
        'Y_train_pred': list(Y_train_pred)
    })
    
    test_results = pd.DataFrame({
        'scenario_test': scenarios_test,
        'Y_test_scaled': list(Y_test_scaled), 
        'Y_test_pred': list(Y_test_pred)
    })
    
    # Save to CSV
    train_results.to_csv(os.path.join(output_dir, "train_results.csv"), index=False)
    test_results.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

    # Save training curves as PNG
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Training MAE")
    if "val_mae" in history.history:
        plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("MAE over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()  

    # Scatter plot predicted vs observed values for each target (only test set)
    num_targets = Y_test_scaled.shape[1]  # Number of target variables
    
    for target_idx in range(num_targets):
        plt.figure(figsize=(8, 6))

        plt.scatter(Y_test_scaled[:, target_idx], Y_test_pred[:, target_idx], color='red', alpha=0.6, label="Test")

        min_val = min(np.min(Y_test_scaled[:, target_idx]), np.min(Y_test_pred[:, target_idx]))
        max_val = max(np.max(Y_test_scaled[:, target_idx]), np.max(Y_test_pred[:, target_idx]))
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label="Ideal (y = x)")

        plt.title(f"Predicted vs Observed (Test Set) - Target {target_idx + 1}")
        plt.xlabel(f"Observed (Target {target_idx + 1})")
        plt.ylabel(f"Predicted (Target {target_idx + 1})")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_test_pred_vs_obs_target_{target_idx + 1}.png"))
        plt.close()

