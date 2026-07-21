# NMSE normalization of the MSE loss

This note documents why and how the MSE term in the KAN training loop is
normalized by `Var(Y)` (turning it into **NMSE**, the normalized MSE), and how
that term is combined with the HSIC independence loss in the `hybrid` strategy.

Relevant code: [`src/models/kan.py`](../src/models/kan.py), method
`kan_predictor.custom_fit`.

---

## 1. The problem

The training loop supports three loss strategies:

- `mse` — ordinary regression loss.
- `hsic` — normalized HSIC (`nHSIC`) between the inputs `X` and the residuals,
  an independence loss. `nHSIC` is already bounded in `[0, 1]` (0 = independence,
  the goal; 1 = maximal dependence) and is **unit-free**.
- `hybrid` — a weighted sum `alpha * MSE + beta * nHSIC`.

For the `hybrid` sum to make sense, both terms must live on a **comparable
scale**. Raw MSE does not: its magnitude depends on the units and variance of the
target `Y`, so a single fixed `alpha`/`beta` pair would weight nodes differently
depending on how large their targets happen to be. We therefore need to put MSE
onto the same scale-free footing as `nHSIC`.

An earlier version divided the MSE by `mse_inicial` — the MSE of the *first
mini-batch of the first epoch*. That pins the loss curve to exactly 1 at step 0,
but it is a **noisy, non-deterministic anchor** (it depends on the random weight
initialization and the batch order) and it has no absolute meaning. This note
replaces it with `Var(Y)`.

---

## 2. What NMSE is

Divide the MSE by the variance of the target:

```
NMSE = E[(y - ŷ)²] / Var(Y) = 1 - R²
```

This is the **fraction of variance unexplained** (FVU). Its endpoints are
meaningful in absolute terms:

- `NMSE = 1` → the model does no better than predicting the constant mean of `Y`
  (i.e. it has learned nothing about the parents).
- `NMSE = 0` → perfect fit.
- `NMSE > 1` → the model is *worse* than predicting the mean (can happen early in
  training or on a bad test split).

### Why the constant-mean predictor gives `MSE = Var(Y)`

For any constant predictor `ŷ = c`, add and subtract the mean `μ = E[Y]`:

```
MSE(c) = E[(Y - c)²]
       = E[(Y - μ)²] + 2(μ - c)·E[Y - μ] + (μ - c)²
       = Var(Y) + (μ - c)²          (the middle term is 0 because E[Y - μ] = 0)
```

So the best constant is `c = μ` (the mean), and its error is exactly `Var(Y)` —
this is just the definition of variance (mean squared deviation from the mean).
`Var(Y)` is therefore the MSE of the "dumbest non-trivial" model, which makes it
the natural baseline to normalize against.

### How far NMSE can actually drop

A predictor that uses the parents `X` can beat the mean baseline. By the law of
total variance:

```
Var(Y) = E[Var(Y | X)]  +  Var(E[Y | X])
         └ irreducible ┘    └ explainable ┘
            noise            by the parents
```

The best possible predictor is the conditional mean `ŷ = E[Y | X]`, whose MSE is
`E[Var(Y | X)]`. In an additive-noise model `Y = f(parents) + noise`, that floor
is `Var(noise)`, so a perfectly trained mechanism reaches:

```
optimal NMSE = Var(noise) / Var(Y)   (the fraction of Y that is pure noise, not 0)
```

The leftover is exactly the residual fed to HSIC — which should be *independent*
of the parents once the true `f` is recovered.

---

## 3. Why `Var(Y)` is a good normalizer

**1. Scale invariance (the key property).** Rescale a node's target `Y → c·Y`.
Then `MSE → c²·MSE` and `Var(Y) → c²·Var(Y)`, so NMSE is **unchanged**. This is
what lets a single fixed `alpha`/`beta` be meaningful across nodes whose targets
have different variances. Raw MSE does not have this invariance.

**2. It matches HSIC's status.** `nHSIC` is already unit-free and absolute
(0 = independence, 1 = max dependence). Var(Y)-normalized MSE is *also* unit-free
and absolute (0 = perfect, 1 = mean-level), and with the **same "0 = good"
convention**. Both terms of `alpha·NMSE + beta·nHSIC` now sit on the same
`[0, 1]`-ish scale, so `alpha`/`beta` are honest balancing weights rather than
hidden scale-correction factors.

**3. Deterministic, stable anchor.** `Var(Y)` is a fixed property of the data,
computed once. `mse_inicial` was stochastic (random init + first batch), and the
untrained KAN is not guaranteed to sit near the mean predictor, so `mse_inicial`
could be well above `Var(Y)` and jitter run to run. A normalizer should be a
*reference baseline*, not a point on the optimization trajectory.

**4. Cross-node / cross-run comparability.** Two nodes both at `NMSE = 0.1`
genuinely explain 90% of their variance each — a statement you can put in a
paper table. "10% of the initial loss" is not comparable across nodes or seeds.

---

## 4. Implementation

In `custom_fit`, the anchor is computed **once before the training loop** from the
training labels:

```python
eps_norm = 1e-8
y_train_full = dataset['train_label'].float()
if y_train_full.dim() <= 1:
    y_var = torch.var(y_train_full, unbiased=False)
else:
    y_var = torch.var(y_train_full, dim=0, unbiased=False).mean()
y_var = torch.clamp(y_var.detach(), min=eps_norm)
```

Details:

- **`unbiased=False`** (divide by `N`, not `N-1`) matches `nn.MSELoss`'s mean
  reduction, so `NMSE == 1` exactly for the constant-mean predictor.
- **Multi-output safe.** For a `(n,)` target it uses the plain variance; for
  `(n, d)` it averages the per-column variances, consistent with how `MSELoss`
  averages over all elements.
- **`detach()`** keeps the anchor out of the autograd graph (it is a constant),
  and **`clamp(min=eps_norm)`** guards against near-constant/degenerate nodes.

The same `y_var` is then used at all four normalization sites — train and test,
`mse` and `hybrid`:

```python
# train, mse
train_loss = mse_part / y_var

# train, hybrid
train_loss = alpha * mse_part / y_var + beta_weight * hsic_part

# test, mse
test_loss = mse_part_test / y_var

# test, hybrid
test_loss = alpha * mse_part_test / y_var + beta_weight * hsic_part_test
```

The **test loss reuses the training-set `y_var`** (it is never recomputed on the
test split), so the train and test curves are on the same scale.

The `hsic` branch is left untouched: `nHSIC` keeps its absolute `[0, 1]` meaning
and does not need an anchor.

---

## 5. Caveats

**This is an in-loss change, not just cosmetic.** The optimized objective is:

```
loss = train_loss + lamb · reg
```

The regularization term `lamb · reg` is **not** divided by `y_var`. Dividing only
the MSE by a constant `c = Var(Y)` is algebraically:

```
mse/c + lamb·reg  =  (1/c) · ( mse + (c·lamb)·reg )
```

so it multiplies the **effective regularization strength** by `c` relative to the
data-fit term. With **Adam** (approximately invariant to scaling the whole loss),
the overall `1/c` prefactor is largely absorbed, and the *real* remaining effect
is the shift `lamb → c·lamb` in the fit-vs-regularization balance.

- Because the data is standardized, `Var(Y) ≈ 1` per node, so in practice this is
  close to a no-op — but it is now a **fixed, reproducible** factor instead of the
  noisy per-run `mse_inicial`.
- If you want the pure-`mse` objective to be *exactly* untouched, optimize the raw
  MSE and divide by `y_var` only when logging/plotting. For `hybrid`, the
  normalization *belongs* in the loss (that is the whole point of making MSE and
  HSIC comparable), so it stays there.

**Curves start near 1, not exactly at 1.** With the old anchor the curve was
pinned to exactly 1 at step 0. With `Var(Y)` it starts at the true initial NMSE
(`initial MSE / Var(Y)`), which can be slightly above 1 if the untrained KAN is
worse than the mean predictor. This is the correct, honest behavior.

---

## 6. Better / alternative solutions

`Var(Y)` is the best *simple, static* choice. More ambitious options, in
increasing order of effort:

- **A. Standardize `Y` per node and use raw MSE.** Then `Var(Y) = 1` by
  construction and the division disappears. Cleanest if you already standardize —
  the normalization becomes implicit in preprocessing.
- **B. Running (EMA) normalization.** Normalize each loss by an exponential moving
  average of its own recent magnitude, so *both* terms stay `O(1)` throughout
  training, not only at init. More robust, but reintroduces nondeterminism.
- **C. Uncertainty-based weighting** (Kendall, Gal & Cipolla, 2018). Learn scalars
  `σ₁`, `σ₂` and use `L = MSE/(2σ₁²) + HSIC/(2σ₂²) + log σ₁ + log σ₂`. Auto-balances
  the two objectives; adds two trainable parameters.
- **D. Constrained / Lagrangian formulation (most principled for the causal
  goal).** The real objective is "fit the mechanism well *subject to* residual
  independence", i.e. `min MSE s.t. HSIC ≤ ε`. Solve with an augmented Lagrangian /
  an adaptive multiplier `λ`. You then choose a **tolerance `ε`** on residual
  dependence (a direct causal interpretation) instead of tuning `alpha`/`beta`, and
  the magnitude mismatch is handled automatically. More code; needs a stable
  multiplier update and a noise-robust HSIC estimate.
- **E. Pareto / multi-objective sweep.** The `alpha0.1…0.9` grid already samples
  the trade-off frontier between fit and independence; reporting the front is often
  more informative for the paper than a single weighted-sum operating point.

**Recommendation:** keep the `Var(Y)` normalization as the default (small change,
removes the noisy anchor, makes `alpha`/`beta` comparable across nodes and runs),
and keep HSIC as the raw normalized value. If the causal story needs
strengthening later, the constrained/Lagrangian formulation (D) is the direction
that is genuinely *better* rather than just cleaner.
