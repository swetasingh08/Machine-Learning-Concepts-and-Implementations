
# Locally Weighted Regression (LWR)

### Non-Parametric Local Regression from Scratch

This project implements **Locally Weighted Regression (LWR)** — a non-parametric, instance-based learning algorithm that fits a local linear model for each prediction point. Unlike global regression which fits one model to all data, LWR adapts to local structure, capturing complex non-linear relationships without assuming a global functional form.


---

## 📑 Table of Contents

- [Overview](#-overview)
- [Theoretical Foundations](#-theoretical-foundations)
- [Quick Start](#-quick-start)
- [Algorithm Mechanics](#-algorithm-mechanics)
- [Implementation Details](#-implementation-details)
- [Mathematical Deep Dive](#-mathematical-deep-dive)
- [The Role of Tau (Bandwidth)](#-the-role-of-tau-bandwidth)
- [Interpreting the Visualization](#-interpreting-the-visualization)
- [Comparison with Other Regression Methods](#-comparison-with-other-regression-methods)
- [Limitations & Extensions](#-limitations--extensions)
- [Troubleshooting](#-troubleshooting)

---

## 📝 Overview

### The Problem with Global Regression

Traditional linear regression fits a single global model:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

This works well when the relationship between features and target is approximately linear. But real-world data often exhibits complex, non-linear patterns that a single straight line (or hyperplane) cannot capture.

**Consider this scenario:** Housing prices might increase linearly with square footage in one neighborhood but plateau in another. A global linear model would average these effects, missing local patterns entirely.

### The Local Approach

Locally Weighted Regression takes a fundamentally different approach:

> **Instead of fitting one global model, fit a separate local model for each prediction point, using only the most relevant (nearby) training data.**

This is analogous to making predictions by consulting the most similar historical examples rather than applying a one-size-fits-all rule.

**Key insight:** The algorithm answers "What is the predicted value at point x₀?" by:
1. Finding training points close to x₀
2. Giving more weight to points that are closer
3. Fitting a weighted linear regression using these local weights
4. Using that local model to predict at x₀

---

## 🧠 Theoretical Foundations

### The Bias-Variance Tradeoff in LWR

LWR navigates the bias-variance tradeoff through its locality:

| Model Type | Bias | Variance | Flexibility |
|------------|------|----------|-------------|
| **Global Linear Regression** | High (assumes linearity) | Low (few parameters) | Low |
| **LWR (large tau)** | Moderate | Moderate | Moderate |
| **LWR (small tau)** | Low (adapts locally) | High (many effective parameters) | High |
| **KNN Regression** | Low | High (depends on K) | High |

LWR occupies a middle ground: more flexible than global regression but more structured than pure nearest-neighbor methods. The tau (bandwidth) parameter controls this tradeoff.

### Instance-Based vs Model-Based Learning

| Property | Model-Based (Linear Regression) | Instance-Based (LWR) |
|----------|--------------------------------|----------------------|
| **Training** | Computes and stores parameters | Stores all training data |
| **Prediction** | Fast (matrix multiplication) | Slow (local model per query) |
| **Adaptability** | Must retrain for new data | Naturally incorporates new data |
| **Model Complexity** | Fixed by parameter count | Varies with data density |
| **Interpretability** | Global coefficients | Local fits (harder to summarize) |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- NumPy library
- Matplotlib library

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/locally-weighted-regression.git
cd locally-weighted-regression

# Install dependencies
pip install numpy matplotlib

# Run the demonstration
python lwr_demo.py
```

### Expected Output

The script generates a 2×2 grid of plots showing LWR fits for four different tau (bandwidth) values:
- **tau = 10.0:** Very smooth, near-linear fit (high bias)
- **tau = 1.0:** Moderate smoothness, captures main curve
- **tau = 0.1:** Flexible fit, follows data closely
- **tau = 0.01:** Highly wiggly, overfitting to noise

---

## 🔧 Algorithm Mechanics

### The LWR Algorithm

```
Algorithm: Locally Weighted Regression

Input:  Training data (X, Y)
        Query point x₀
        Bandwidth parameter tau

Output: Predicted value ŷ for x₀

1. For each training point xᵢ:
      Compute weight wᵢ = K(x₀, xᵢ, tau)
      where K is the radial (Gaussian) kernel

2. Solve weighted linear regression:
      β = argmin Σ wᵢ · (yᵢ - xᵢᵀβ)²
   
   This gives:
      β = (XᵀWX)⁻¹XᵀWY
   
   where W = diag(w₁, w₂, ..., wₙ)

3. Predict: ŷ = x₀ᵀβ
```

### Step-by-Step Walkthrough

#### Step 1: Compute Weights

For a query point x₀, compute how much attention to pay to each training point:

```python
weights = radial_kernel(x0, X, tau)
```

Points close to x₀ receive weights near 1.0; distant points receive weights near 0.0.

#### Step 2: Weighted Linear Regression

Fit a linear model where the importance of each training point is proportional to its weight:

```python
xw = X.T * weights          # Weighted design matrix
beta = np.linalg.pinv(xw @ X) @ xw @ Y  # Weighted normal equation
```

This is the standard normal equation β = (XᵀX)⁻¹XᵀY, but with weights incorporated.

#### Step 3: Local Prediction

Use the locally-fitted model to predict at x₀:

```python
y_pred = x0 @ beta
```

### Visualizing the Process

For a query point at x₀ = 1.5:

```
Training Data:          Weights (tau=0.5):        Local Fit:
    |                       |                       |
Y   |  ·  ·                 |  ·  ·                 |     /
    | ·   ·  ·              | ·   ·  ·              |    /←── local line
    |  ·  ·  ·●             |  ·  ·  ·◉             |   /  ◉ query point
    |   ·  ·                |   ·  ·                |  /
    |      ·                |      ·                | /
    +------------- X        +------------- X        +------------- X
                              ◉ = high weight
                              · = low weight
```

---

## 💻 Implementation Details

### Code Structure

```
lwr_demo.py
├── radial_kernel(x0, X, tau)
│   └── Computes Gaussian weights for all training points
├── local_regression(x0, X, Y, tau)
│   ├── Add bias terms to x0 and X
│   ├── Compute weights via radial kernel
│   ├── Solve weighted normal equation
│   └── Return predicted value
├── Dataset Generation
│   ├── Create 1000 points along X = [-3, 3]
│   ├── Generate Y = log(|X² - 1| + 0.5)
│   └── Add Gaussian noise (σ=0.1)
├── plot_lwr(tau, subplot_index)
│   ├── Compute predictions across domain
│   ├── Scatter plot of training data
│   └── Line plot of LWR fit
└── Main Visualization
    └── 2×2 grid for four tau values
```

### The Radial (Gaussian) Kernel

```python
def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))
```

**Mathematical form:**
```
K(x₀, xᵢ, τ) = exp(-||xᵢ - x₀||² / (2τ²))
```

**Properties:**
- When xᵢ = x₀: K = exp(0) = 1.0 (maximum weight)
- When ||xᵢ - x₀|| = τ: K = exp(-1/2) ≈ 0.607
- When ||xᵢ - x₀|| = 2τ: K = exp(-2) ≈ 0.135
- When ||xᵢ - x₀|| → ∞: K → 0.0

**Why Gaussian?**
- Smooth, differentiable everywhere
- Naturally bounded between 0 and 1
- Single parameter (τ) controls locality
- Drops off smoothly rather than abruptly

### Local Regression Function

```python
def local_regression(x0, X, Y, tau):
    x0 = np.r_[1, x0]                # Add bias term: [1, x0]
    X = np.c_[np.ones(len(X)), X]    # Add bias column to design matrix
    
    weights = radial_kernel(x0, X, tau)
    xw = X.T * weights                # Weighted design matrix
    
    beta = np.linalg.pinv(xw @ X) @ xw @ Y  # Solve via pseudo-inverse
    return x0 @ beta                  # Predict
```

**Why add bias?**
The bias term (column of 1s) allows the local linear model to have an intercept. Without it, the fitted line would be forced through the origin.

**Why pseudo-inverse?**
`np.linalg.pinv` computes the Moore-Penrose pseudo-inverse, which handles cases where `xw @ X` is singular or near-singular (common when few points have non-zero weight).

### Dataset Generation

```python
X = np.linspace(-3, 3, num=1000)
Y = np.log(np.abs(X**2 - 1) + 0.5)
X += np.random.normal(scale=0.1, size=n)
```

**The underlying function:** `Y = log(|X² - 1| + 0.5)`

This function has interesting features:
- Two minima near X = ±1 (where X² - 1 ≈ 0)
- Logarithm compresses large values
- Non-linear and non-polynomial
- Cannot be well-fit by a single global linear model

**Added noise:** Gaussian noise with σ=0.1 makes the problem realistic — the true function is obscured by random variation.

---

## 📐 Mathematical Deep Dive

### Weighted Least Squares Derivation

The weighted least squares problem minimizes:

```
J(β) = Σ(i=1 to n) wᵢ · (yᵢ - xᵢᵀβ)²
```

In matrix form:

```
J(β) = (Y - Xβ)ᵀW(Y - Xβ)
```

Where W = diag(w₁, w₂, ..., wₙ).

**Taking the derivative and setting to zero:**

```
∂J/∂β = -2XᵀW(Y - Xβ) = 0
XᵀWXβ = XᵀWY
β = (XᵀWX)⁻¹XᵀWY
```

This is the weighted normal equation. It reduces to ordinary least squares when W = I (all weights equal to 1).

### The Effective Number of Parameters

Unlike global linear regression with a fixed number of parameters, LWR's effective complexity varies:

```
df(τ) ≈ Σ(i=1 to n) hᵢᵢ(τ)
```

Where hᵢᵢ are the diagonal elements of the hat matrix H = X(XᵀWX)⁻¹XᵀW.

- Large τ: df → 2 (intercept + slope, like global regression)
- Small τ: df → n (one parameter per data point, overfitting)

### Connection to Kernel Density Estimation

LWR can be viewed as estimating the conditional expectation:

```
ŷ(x₀) = Σ wᵢ(x₀) · yᵢ / Σ wᵢ(x₀)
```

This is the Nadaraya-Watson estimator — a kernel-weighted average. The local linear version used here reduces bias at boundaries.

---

## 🎯 The Role of Tau (Bandwidth)

Tau is the single most important hyperparameter in LWR. It controls the size of the local neighborhood used for each prediction.

### Tau as Smoothing Parameter

```
τ → 0:  Each prediction uses only points extremely close to x₀
        → Very flexible, follows noise (overfitting)

τ → ∞:  All points receive equal weight
        → Equivalent to global linear regression (underfitting)
```

### Visual Comparison

| Tau | Behavior | Bias | Variance | When Appropriate |
|-----|----------|------|----------|------------------|
| **10.0** | Near-linear, very smooth | High | Low | Strong prior that relationship is simple |
| **1.0** | Captures main trend, smooth curves | Moderate | Moderate | Good default for this dataset |
| **0.1** | Follows data closely, some wiggliness | Low | Moderate-High | When local patterns are important |
| **0.01** | Extreme wiggliness, overfitting | Very Low | Very High | Only with very dense, noise-free data |

### The Art of Tau Selection

Choosing tau involves balancing:

```
Optimal τ = argmin [Bias²(τ) + Variance(τ)]

Where:
  Bias²(τ) increases with τ (smoother = more biased)
  Variance(τ) decreases with τ (smoother = less variance)
```

**Practical methods:**
- **Cross-validation:** Choose τ that minimizes prediction error on held-out data
- **Visual inspection:** Plot fits for multiple τ values (as this implementation does)
- **Rule of thumb:** τ proportional to average nearest-neighbor distance
- **Generalized Cross-Validation (GCV):** Analytical approximation to leave-one-out error

---

## 📊 Interpreting the Visualization

### The Four-Panel Plot

```
┌──────────────────────┐  ┌──────────────────────┐
│    tau = 10.0        │  │    tau = 1.0         │
│                      │  │                      │
│  ────────────────    │  │     ╭──────╮        │
│  (nearly straight)   │  │    ╱        ╲       │
│                      │  │   ╱          ╲      │
│  Misses the dips     │  │  Captures shape     │
└──────────────────────┘  └──────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐
│    tau = 0.1         │  │    tau = 0.01        │
│                      │  │                      │
│   ╭╮    ╭╮          │  │ ╭╮╭╮  ╭╮╭╮        │
│  ╱  ╲  ╱  ╲        │  │ ╲╱  ╲╲╱  ╲╱      │
│ ╱    ╲╱    ╲       │  │  ╲    ╲╲    ╲     │
│ Follows data well   │  │  Chasing noise      │
└──────────────────────┘  └──────────────────────┘
```

### What to Look For

**Blue dots:** The training data — 1000 points with noise. Notice the increased density near the dips (around X = ±1), which is an artifact of how the function compresses values.

**Red line:** The LWR prediction across the domain. This is what the model would predict for any X value.

**Key observations:**
1. **tau = 10.0:** The red line is almost straight — LWR behaves like global linear regression when tau is large
2. **tau = 1.0:** The red line captures the two dips while maintaining smoothness — this is close to optimal for this data
3. **tau = 0.1:** The red line wiggles to follow local variations — starting to overfit
4. **tau = 0.01:** The red line zigzags dramatically — overfitting to noise, high variance

---

## 📊 Comparison with Other Regression Methods

### LWR vs Global Linear Regression

| Aspect | Linear Regression | LWR |
|--------|-------------------|-----|
| **Model** | y = β₀ + β₁x | Local linear per query point |
| **Parameters** | 2 (global) | 2 × n_query (effective) |
| **Non-linearity** | Cannot capture | Naturally captures |
| **Training** | O(nd² + d³) once | O(1) — just store data |
| **Prediction** | O(d) | O(nd²) per query |
| **Interpretability** | Simple global coefficients | Hard to summarize |

### LWR vs Polynomial Regression

| Aspect | Polynomial Regression | LWR |
|--------|----------------------|-----|
| **Flexibility** | Global polynomial of degree p | Local linear, globally flexible |
| **Extrapolation** | Defined by polynomial | Poor (no training data far away) |
| **Boundary Behavior** | Can oscillate wildly (Runge's phenomenon) | Well-behaved (local linear) |
| **Parameter Selection** | Choose polynomial degree | Choose bandwidth tau |

### LWR vs Kernel Ridge Regression

| Aspect | Kernel Ridge Regression | LWR |
|--------|------------------------|-----|
| **Optimization** | Global convex optimization | Local optimization per query |
| **Prediction Speed** | O(n) after training | O(n) per query |
| **Regularization** | Explicit (ridge parameter) | Implicit (through tau) |
| **Sparsity** | Dense (all points contribute) | Sparse (only nearby points matter) |

### LWR vs LOESS

LOESS (LOcally Estimated Scatterplot Smoothing) is a popular variant of LWR:

| Aspect | Basic LWR | LOESS |
|--------|-----------|-------|
| **Weighting** | All points, Gaussian kernel | K nearest neighbors, tricube kernel |
| **Robustness** | Sensitive to outliers | Iterative reweighting for robustness |
| **Computation** | O(n) weights per query | O(n) to find K neighbors |
| **Implementation** | Simple | More sophisticated |

---

## 🚀 Limitations & Extensions

### Current Limitations

1. **Computational Cost:** Each prediction requires solving a weighted linear regression on the entire training set — O(n) per query point.

2. **Curse of Dimensionality:** In high dimensions, all points become approximately equidistant, making "local" neighborhoods meaningless.

3. **Poor Extrapolation:** LWR cannot predict beyond the range of training data because no training points exist nearby to form a local model.

4. **Tau Selection:** No automatic method for choosing optimal tau — requires cross-validation or manual tuning.

5. **Boundary Bias:** At the edges of the data range, the kernel becomes asymmetric, potentially biasing predictions.

6. **Memory Requirements:** All training data must be stored — problematic for very large datasets.

### Potential Extensions

#### Automatic Tau Selection via Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut

def cv_tau(X, Y, tau_values):
    loo = LeaveOneOut()
    errors = []
    
    for tau in tau_values:
        cv_error = 0
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            y_pred = local_regression(X_test[0], X_train, Y_train, tau)
            cv_error += (Y_test[0] - y_pred) ** 2
        
        errors.append(cv_error / len(X))
    
    best_tau = tau_values[np.argmin(errors)]
    return best_tau, errors
```

#### Robust LWR (Handling Outliers)

Add iterative reweighting based on residuals:

```python
def robust_lwr(x0, X, Y, tau, iterations=3):
    weights = radial_kernel(x0, X, tau)
    
    for _ in range(iterations):
        # Fit model with current weights
        beta = weighted_least_squares(X, Y, weights)
        residuals = Y - X @ beta
        
        # Downweight points with large residuals
        bisquare_weights = (1 - (residuals / (6 * np.median(np.abs(residuals))))**2)**2
        bisquare_weights[np.abs(residuals) >= 6 * np.median(np.abs(residuals))] = 0
        
        weights = weights * bisquare_weights
    
    return x0 @ beta
```

#### k-Nearest Neighbors LWR

Only use K nearest neighbors instead of all points:

```python
def knn_lwr(x0, X, Y, tau, k=50):
    # Find K nearest neighbors
    distances = np.sum((X - x0) ** 2, axis=1)
    knn_idx = np.argsort(distances)[:k]
    
    X_knn = X[knn_idx]
    Y_knn = Y[knn_idx]
    
    # Apply LWR only on K neighbors
    return local_regression(x0, X_knn, Y_knn, tau)
```

#### Multivariate Extension

Extend to multiple features:

```python
def local_regression_multivariate(x0, X, Y, tau):
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
    
    # Multivariate Gaussian kernel
    weights = np.exp(-np.sum((X - x0)**2, axis=1) / (-2 * tau * tau))
    
    xw = X.T * weights
    beta = np.linalg.pinv(xw @ X) @ xw @ Y
    
    return x0 @ beta
```

#### Confidence Intervals

Estimate prediction uncertainty:

```python
def lwr_with_confidence(x0, X, Y, tau):
    weights = radial_kernel(x0, X, tau)
    
    # Point estimate
    y_pred = local_regression(x0, X, Y, tau)
    
    # Residual variance (weighted)
    residuals = Y - local_regression_vectorized(X, X, Y, tau)
    sigma2 = np.average(residuals**2, weights=weights)
    
    # Leverage at x0
    X_design = np.c_[np.ones(len(X)), X]
    H = X_design @ np.linalg.pinv(X_design.T * weights @ X_design) @ (X_design.T * weights)
    leverage = np.diag(H)
    
    # Standard error
    se = np.sqrt(sigma2 * (1 + leverage.mean()))
    
    return y_pred, y_pred - 2*se, y_pred + 2*se
```

#### Online/Incremental LWR

Update predictions as new data arrives:

```python
class IncrementalLWR:
    def __init__(self, tau):
        self.tau = tau
        self.X = []
        self.Y = []
    
    def add_point(self, x, y):
        self.X.append(x)
        self.Y.append(y)
    
    def predict(self, x0):
        X = np.array(self.X)
        Y = np.array(self.Y)
        return local_regression(x0, X, Y, self.tau)
```

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| All predictions are the same | tau is too large | Reduce tau |
| Predictions oscillate wildly | tau is too small | Increase tau |
| `LinAlgError: Singular matrix` | Too few points with non-zero weight | Increase tau or add regularization |
| Predictions are NaN | Division by zero in weights | Check for tau ≤ 0 |
| Very slow execution | Too many query points | Reduce domain resolution |
| Flat predictions at boundaries | Boundary bias | Consider local quadratic instead of linear |



*"All models are wrong, but some are useful — and locally weighted models are useful precisely because they don't pretend to be globally correct."*
