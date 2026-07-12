"""
KaCGM Refactoring Test Suite
-----------------------------
PURPOSE OF THIS TEST:
The main goal is to perform a Regression Test on the 'kan_model_mixed' class after
modifying its loss function. The idea is to ensure that the new refactored training
loop (which now supports 'mse', 'hsic' and 'hsic_mse' loss functions) behaves
exactly the same as the original implementation when using the 'mse' loss function.

HOW TO CHECK IF IT IS WORKING:
1. Run this script before making any changes to 'src/models/kan.py' and save the 
printed "Check-value" sum.
2. Modify the loss function in 'src/models/kan.py' to support 'hsic' and 'hsic_mse'.
3. Run this script again and compare the new "Check-value" sum with the previously 
saved one. If they match, the refactoring is successful and the new implementation 
is consistent.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from src.models.kan import kan_model_mixed  
from copy import deepcopy

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
N = 200
X = np.random.uniform(-2, 2, size=N)
Y = 2 * X + np.random.normal(0, 0.5, size=N) 
df = pd.DataFrame({'X': X, 'Y': Y})

#3. Define hyperparameters simulating the project's configuration
node_type_map = {"X": "continuous", "Y": "continuous"}
num_classes_map ={} #Empty since we are dealing with continuous variables
kan_hyperparameters = {
    "hidden_dim": 0,
    "batch_size": 50,
    "steps": 15,
    "lr": 0.05,
    "mult_kan": False,
    "loss": "mse",
    "early_stop": False,
    "verbose": 0,
    "node_types": node_type_map,
    "num_classes": num_classes_map
}

#In KaCGM, only non-root nodes require KAN hyperparameters definitions.
#Root nodes like 'X' are handled automatically as distributions.
params ={"Y": deepcopy(kan_hyperparameters)}

print ("---Starting Regression Test training---")
model = kan_model_mixed(dag, deepcopy(params))
model.fit(df)

#4. Generate steady state data points to verify the mathematical outputs
X_test = np.linspace(-2, 2, 5).reshape(-1, 1)

kan_predictor_Y = model.models['Y']
predictions = kan_predictor_Y.predict(X_test)

#5. Show the results and compute a check-value for regression testing
print ("\n---Regression Test Results---")
print(f"First 3 predictions of the model: {predictions.flatten()[:3]}")
print(f"Check value: {predictions.sum():.6f}")

