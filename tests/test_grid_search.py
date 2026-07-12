"""
KaCGM Automated Grid Search
-----------------------------
PURPOSE OF THIS TEST:
This script performs an automated hyperparameter tuning loop over different values of .
'beta_weight' to find the optimal balance between MSE (prediction, R2) and HSIC (causal independence).
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt
from src.models.kan import kan_model_mixed  , hsic_loss
from copy import deepcopy
from pathlib import Path
from sklearn.metrics import r2_score

def set_determinism(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #Force Pytorch to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True

set_determinism(42)

#1. Create a simple Directed Acyclic Graph (DAG): X -> Y
dag = nx.DiGraph()
dag.add_edges_from([('X', 'Y')])

#2. Generate continuous synthetic data 
N = 300
X = np.random.uniform(-2, 2, size=N)
Y = X**3 + np.random.normal(0, 0.4, size=N)
df = pd.DataFrame({'X': X, 'Y': Y})

#DEFINE THE GRID OF BETAS TO TEST
betas_to_test= [0.0,0.1,0.5, 1, 5, 10, 20, 100.0, 500.0]
grid_results = []

#=============================================================
# STARTING AUTOMATED GRID SEARCH LOOP
#=============================================================
print("\n---Starting Automated Grid Search---")

#3. Define hyperparameters simulating the project's configuration

for beta in betas_to_test:
    print(f"Training KAN with alpha_weight = 1.0 and beta_weight ={beta}...")
    node_type_map = {"X": "continuous", "Y": "continuous"}
    num_classes_map ={} #Empty since we are dealing with continuous variables
    kan_hyperparameters = {
        "hidden_dim": 4, #Added hidden dimensions so KAN can capture the cubic curve
        "batch_size": 100,
        "steps": 400,
        "lr": 0.02,
        "lamb": 0.0,
        "mult_kan": False,
        "loss": "hybrid" if beta > 0 else "mse",  # Use hybrid loss if beta > 0, otherwise use mse
        "early_stop": False,
        "verbose": 0, 
        "node_types": node_type_map,
        "num_classes": num_classes_map,
        "alpha_weight": 1.0, #Explicitly set alpha_weight
        "beta_weight": beta  #Explicitly set beta_weight
    }

    #In KaCGM, only non-root nodes require KAN hyperparameters definitions.
    #Root nodes like 'X' are handled automatically as distributions.
    params ={"Y": deepcopy(kan_hyperparameters)}

    model = kan_model_mixed(dag, deepcopy(params))
    model.fit(df)

    #=============================================================
    # VISUAL EVALUATION OF RESIDUALS
    #=============================================================
    print("\n---Visual Evaluation of Residuals---")
    x_np= df["X"].to_numpy().copy()
    y_real = df["Y"].to_numpy().copy()

    # Get model predictions 
    kan_predictor_Y = model.models['Y']
    y_pred = kan_predictor_Y.predict(x_np.reshape(-1, 1)).flatten()

    residuals_np = y_real - y_pred

    x_tensor = torch.from_numpy(x_np).float().reshape(-1, 1)
    residuals_tensor = torch.from_numpy(residuals_np).float()

    #Calculate target validation metrics
    r2=r2_score(y_real, y_pred)
    final_hsic = hsic_loss(x_tensor, residuals_tensor, sigma=1.0).item()
    print(f"Results for Beta={beta} -> R2 score: {r2:.4f} | Final HSIC: {final_hsic:.6f}")

    grid_results.append({
        "beta": beta,
        "r2":r2,
        "hsic":final_hsic
    })

df_results = pd.DataFrame(grid_results)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))

#Plot 1: Beta vs R2 (prediction quality)
ax1.plot(df_results["beta"], df_results["r2"], marker='o', color='red', linewidth=2, label="Prediction Accuracy (R2)")
ax1.set_xscale('symlog', linthresh=0.1)
ax1.set_title('Prediction Performance')
ax1.set_xlabel("Beta weight (log scale)")
ax1.set_ylabel("R2 score")
ax1.legend()
ax1.grid(True, alpha=0.3)

#Plot 2: Beta vs HSIC (Independence quality)
ax2.plot(df_results["beta"], df_results["hsic"], marker='s', color='purple', linewidth=2, label = "Casual Independence (HSIC)")
ax2.set_xscale('symlog', linthresh=0.1)
ax2.set_title('Residual Independence')
ax2.set_xlabel("Beta weight (log scale)")
ax2.set_ylabel("final HSIC Value")
ax2.legend()
ax2.grid(True, alpha=0.3)

#=============================================================
# SAVING AND PLOTTING TO A DEDICATED DIRECTORY
#=============================================================
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

plt.tight_layout()
output_path = plots_dir / f"grid_search_metrics.png"
plt.savefig(output_path, dpi=300)
print(f"Success! Graph saved at {output_path}")

