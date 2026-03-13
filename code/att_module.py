import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_attention_data(Y_pred, Y_true, scenarios, n_replicates=20):
    X_grouped, Y_grouped = [], []
    scenario_ids = np.unique(scenarios)

    for scenario_id in scenario_ids:
        idx = np.where(scenarios == scenario_id)[0]
        if len(idx) != n_replicates:
            continue
        preds = Y_pred[idx].reshape(n_replicates, -1)
        target = np.mean(Y_true[idx], axis=0).reshape(-1)
        X_grouped.append(preds)
        Y_grouped.append(target)

    return np.stack(X_grouped), np.stack(Y_grouped)

def train_att_model(Y_train_pred, Y_train_scaled, scenarios_train,
                          Y_test_pred, Y_test_scaled, scenarios_test,
                          hidden_dim=64, n_queries=6, lr=0.0005, n_epochs=2000, verbose = False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = Y_train_pred.shape[1]

    # prepare data
    X_train_att, Y_train_att = prepare_attention_data(Y_train_pred, Y_train_scaled, scenarios_train)
    X_test_att, Y_test_att = prepare_attention_data(Y_test_pred, Y_test_scaled, scenarios_test)
    X_train_sub, X_val_sub, Y_train_sub, Y_val_sub = train_test_split(X_train_att, Y_train_att, test_size=0.2, random_state=42)

    # Tensors
    X_train_tensor = torch.tensor(X_train_sub, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train_sub, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_sub, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val_sub, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_att, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_att, dtype=torch.float32).to(device)

    # Layers
    key_linear1 = nn.Linear(input_dim, hidden_dim).to(device)
    key_relu = nn.ReLU()
    key_linear2 = nn.Linear(hidden_dim, hidden_dim).to(device)
    queries = nn.Parameter(torch.randn(n_queries, hidden_dim, device=device))
    output_linear = nn.Linear(n_queries, Y_train_tensor.shape[1]).to(device)

    # model
    model_params = list(key_linear1.parameters()) + list(key_linear2.parameters()) + list(output_linear.parameters()) + [queries]
    optimizer = torch.optim.Adam(model_params, lr=lr)
    criterion = nn.MSELoss()

    def forward(x):
        k = key_relu(key_linear1(x))
        k = key_linear2(k)
        attn_scores = torch.einsum('qh,bth->bqt', queries, k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_sums = torch.einsum('bqt,btd->bq', attn_weights, x)
        return output_linear(weighted_sums)

    # training
    for epoch in range(n_epochs):
        model_output = forward(X_train_tensor)
        loss = criterion(model_output, Y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_output = forward(X_val_tensor)
            val_loss = criterion(val_output, Y_val_tensor)

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

    # Pred
    with torch.no_grad():
        test_preds = forward(X_test_tensor).cpu().numpy()
        test_targets = Y_test_tensor.cpu().numpy()

    return test_preds, test_targets

def plot_pred_vs_obs(Y_true, Y_pred, target_parameters, title, output_dir):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_scatter.png")

    plt.figure(figsize=(15, 10))
    for i, param in enumerate(target_parameters):
        plt.subplot(2, 2, i + 1)
        plt.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.5)
        plt.plot([min(Y_true[:, i]), max(Y_true[:, i])], [min(Y_true[:, i]), max(Y_true[:, i])], 'r--')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.title(f'{param} - {title}')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Plots saved")


def print_metrics(Y_true, Y_pred, target_parameters, output_dir):
    import os
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    metrics = []

    for i, param in enumerate(target_parameters):
        rmse = np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i]))
        mae = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
        metrics.append({'Parameter': param, 'RMSE': rmse, 'MAE': mae})

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print("Metrics saved")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_att_results(Y_true_scaled, Y_pred_scaled, target_names, output_dir):

    os.makedirs(output_dir, exist_ok=True)


    results_df = pd.DataFrame(Y_true_scaled, columns=[f"true_{t}" for t in target_names])
    for i, t in enumerate(target_names):
        results_df[f"pred_{t}"] = Y_pred_scaled[:, i]
    
    results_csv_path = os.path.join(output_dir, "test_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)


    for i, param in enumerate(target_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(Y_true_scaled[:, i], Y_pred_scaled[:, i], alpha=0.5)
        min_val = min(Y_true_scaled[:, i].min(), Y_pred_scaled[:, i].min())
        max_val = max(Y_true_scaled[:, i].max(), Y_pred_scaled[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.title(f'{param} - Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}_scatter.png"))
        plt.close()
