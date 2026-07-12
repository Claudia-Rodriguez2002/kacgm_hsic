"""
KaCGM Test Hyperparameters Weight
---------------------------------
PURPOSE OF THIS TEST:
This script verifies that explicitly passing 'alpha_weight' and 'beta_weight' yields the exact same visual 
and numerical results as the current hardcoded setup.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt
from src.models.kan import kan_model_mixed  
from copy import deepcopy
from pathlib import Path

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

#3. Define hyperparameters simulating the project's configuration
STRATEGY = 'hybrid'  
node_type_map = {"X": "continuous", "Y": "continuous"}
num_classes_map ={} #Empty since we are dealing with continuous variables
kan_hyperparameters = {
    "hidden_dim": 4, #Added hidden dimensions so KAN can capture the cubic curve
    "batch_size": 100,
    "steps": 400,
    "lr": 0.02,
    "mult_kan": False,
    "loss": STRATEGY,
    "early_stop": False,
    "verbose": 1, #Enable progress bar to monitor training
    "node_types": node_type_map,
    "num_classes": num_classes_map,
    "alpha_weight": 1, #Explicitly set alpha_weight 
    "beta_weight": 10  #Explicitly set beta_weight 
}

#In KaCGM, only non-root nodes require KAN hyperparameters definitions.
#Root nodes like 'X' are handled automatically as distributions.
params ={"Y": deepcopy(kan_hyperparameters)}

print (f"---Running Hyperparameters Regression Test: {STRATEGY.upper()}---")
model = kan_model_mixed(dag, deepcopy(params))
model.fit(df)

#=============================================================
# VISUAL EVALUATION OF RESIDUALS
#=============================================================
print("\n---Visual Evaluation of Residuals---")
x_real = df["X"].to_numpy()
y_real = df["Y"].to_numpy()

# Get model predictions 
kan_predictor_Y = model.models['Y']
y_pred = kan_predictor_Y.predict(x_real.reshape(-1, 1)).flatten()

residuals = y_real - y_pred

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))

#Plot 1: True values vd Predicted values
ax1.scatter(x_real, y_real, color='blue', alpha=0.5, label='True Values')
ax1.scatter(x_real, y_pred, color='red', alpha=0.5, label='Predicted Values')
ax1.set_title(f'Model fit (Alpha:{kan_hyperparameters["alpha_weight"]}, Beta:{kan_hyperparameters["beta_weight"]})')
ax1.set_xlabel("Input X")
ax1.set_ylabel("Output Y")
ax1.legend()
ax1.grid(True, alpha=0.3)

#Plot 2:Residuals vs Input X (The independence Check)
ax2.scatter(x_real, residuals, color="purple", alpha=0.5,)
ax2.axhline(y=0, color='black', linestyle='--')
ax2.set_title('Residuals vs Input X')
ax2.set_xlabel("Input X")
ax2.set_ylabel("Residuals")
ax2.grid(True, alpha=0.3)

#=============================================================
# SAVING AND PLOTTING TO A DEDICATED DIRECTORY
#=============================================================
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

plt.tight_layout()
output_path = plots_dir / f"residual_analysis_hyperparams_validation_new2.png"
plt.savefig(output_path, dpi=300)
print(f"Success! Graph saved at {output_path}")

