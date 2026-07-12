"""
KaCGM Test HSIC
-----------------------------
PURPOSE OF THIS TEST:
This script explicitly tests the new 'hsic' and 'hybrid' loss functions implemented 
in the 'kan_model_mixed' class. It trains the KAN network using statistical independence 
criteria and outputs a visual analysis of the residuals.

"""
import pytest
import torch
import numpy as np
from src.models.kan import hsic_loss

def seed_all(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def test_hsic_loss_perfect_dependence():
    """
    Test 1: Strong linear dependency.
    If residuals are completely determined by X, HSIC should be significantly above 0.
    """
    seed_all()
    num_samples = 500
    
    # Generate random features for a single parent
    X = torch.randn(num_samples, 1)
    # Residuals are perfectly dependent on X
    residuals = X.clone()
    
    # Compute HSIC loss using the Median Heuristic (sigma=None)
    loss_val = hsic_loss(X, residuals, sigma=None).item()
    
    print(f"\n[Test 1] Strong dependence HSIC: {loss_val:.6f}")
    # Assert that the independence criterion detects a strong relationship
    assert loss_val > 0.05, f"HSIC should be high for dependent variables, got {loss_val}"

def test_hsic_loss_statistical_independence():
    """
    Test 2: Complete statistical independence.
    If X and residuals are drawn from independent Gaussian distributions, 
    the empirical HSIC should be very close to 0.
    """
    seed_all()
    num_samples = 1000  # More samples reduce empirical estimation noise
    
    # Generate independent distributions
    X = torch.randn(num_samples, 1)
    residuals = torch.randn(num_samples, 1)
    
    # Compute HSIC loss
    loss_val = hsic_loss(X, residuals, sigma=None).item()
    
    print(f"[Test 2] Independence HSIC: {loss_val:.6f}")
    # Empirically, for N=1000 independent samples, HSIC is usually < 0.005
    assert loss_val < 0.005, f"HSIC should be close to 0 for independent variables, got {loss_val}"

def test_hsic_loss_multiple_parents():
    """
    Test 3: Triangle graph simulation (Multiple Parents).
    Verifies that the function correctly processes multiple conditioning variables.
    - Case A: Residuals depend on a combination of both parents (High HSIC).
    - Case B: Residuals are independent of both parents (Low HSIC).
    """
    seed_all()
    num_samples = 500
    
    # Simulate two parents (e.g., X and Y for the Z node in your triangle graph)
    X1 = torch.randn(num_samples, 1)
    X2 = torch.randn(num_samples, 1)
    parents = torch.cat([X1, X2], dim=1)  # Shape: (num_samples, 2)
    
    # --- Case A: High Dependency ---
    # Residuals are a non-linear combination of both parents
    dependent_residuals = torch.sin(X1) + X2**2
    loss_dependent = hsic_loss(parents, dependent_residuals, sigma=None).item()
    
    # --- Case B: Full Independence ---
    # Residuals are completely unrelated noise
    independent_residuals = torch.randn(num_samples, 1)
    loss_independent = hsic_loss(parents, independent_residuals, sigma=None).item()
    
    print(f"[Test 3] Multiple Parents - Dependent HSIC: {loss_dependent:.6f}")
    print(f"[Test 3] Multiple Parents - Independent HSIC: {loss_independent:.6f}")
    
    # Assertions
    assert loss_dependent > 0.02, f"HSIC should detect joint dependence, got {loss_dependent}"
    assert loss_independent < 0.005, f"HSIC should be near 0 for joint independence, got {loss_independent}"
    assert loss_dependent > loss_independent, "Dependent HSIC must be significantly higher than Independent HSIC"

def test_hsic_loss_small_batch_safety():
    """
    Test 4: Boundary condition for small batch sizes.
    If the batch size is less than 2, the function should gracefully return 0.0.
    """
    X = torch.randn(1, 2)
    residuals = torch.randn(1, 1)
    
    loss_val = hsic_loss(X, residuals, sigma=None).item()
    assert loss_val == 0.0, f"HSIC should return 0.0 for batches smaller than 2 samples, got {loss_val}"


# --- EVALUACIÓN DE CASOS EXACTOS ---

# Pruebas para el Caso 1: Una variable constante
X_random = torch.randn(5, 1)
res_constant = torch.full((5, 1), 3.14) # Todos los residuos son 3.14

val_caso1 = hsic_loss(X_random, res_constant, sigma=None).item()
print(f"Caso 1 (Constante) -> Resultado obtenido: {val_caso1:.6f} | Esperado: 0.000000")


# Pruebas para el Caso 2: Dos muestras cualesquiera distintas
# Usamos valores totalmente arbitrarios
X_2samples = torch.tensor([[1.5], [9.2]])
res_2samples = torch.tensor([[-0.4], [3.1]])

val_caso2 = hsic_loss(X_2samples, res_2samples, sigma=None).item()
print(f"Caso 2 (2 Muestras)  -> Resultado obtenido: {val_caso2:.6f} | Esperado: 0.154818")