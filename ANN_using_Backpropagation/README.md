

# Artificial Neural Network from Scratch

### A NumPy Implementation for Deep Learning Fundamentals

This project demonstrates the core mechanics of an Artificial Neural Network (ANN) implemented entirely from scratch using **NumPy**. By stripping away high-level framework abstractions (like PyTorch or TensorFlow), this repository exposes the raw mathematical foundations — forward propagation, backpropagation, and gradient descent — powering modern deep learning.


---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architectural Overview](#-architectural-overview)
- [Quick Start](#-quick-start)
- [Theoretical Foundations](#-theoretical-foundations)
- [Mathematical Mechanics](#-mathematical-mechanics)
- [The Training Cycle](#-the-training-cycle)
- [Implementation Details](#-implementation-details)
- [Optimization & Data Engineering](#-optimization--data-engineering)
- [Results & Visualization](#-results--visualization)
- [Scalability & Extensions](#-scalability--extensions)
- [Troubleshooting](#-troubleshooting)

---

## 📝 Overview

### Why Build a Neural Network from Scratch?

Modern deep learning frameworks (TensorFlow, PyTorch, Keras) provide convenient abstractions that hide the underlying mathematics. While these tools accelerate development, they obscure the fundamental operations that make neural networks work. This project strips away those abstractions to reveal:

- **Matrix multiplications** that transform input data through learned weights
- **Activation functions** that introduce non-linearity into the model
- **Loss calculation** that quantifies prediction error
- **Gradient computation** via the chain rule of calculus
- **Parameter updates** through gradient descent optimization

By implementing each of these components with explicit NumPy operations, you gain a visceral understanding of what happens when you call `model.fit()` in a high-level framework.

### What This Project Teaches

- Feedforward neural network architecture and tensor flow
- The sigmoid activation function and its derivative
- Mean Squared Error (MSE) loss and its gradient
- Backpropagation as recursive application of the chain rule
- Batch gradient descent optimization
- Weight initialization strategies for symmetry breaking
- Feature scaling and target normalization for stable training

---

## 🏗️ Architectural Overview

The network implements a **3-layer feedforward architecture** (input layer, one hidden layer, output layer) configured to solve a multi-variable continuous regression problem.

### Network Topology

```
Input Layer          Hidden Layer           Output Layer
(2 features)         (3 neurons)            (1 prediction)
    |                    |                       |
    x1 ──┐          ┌── h1 ──┐             ┌── y_hat
    x2 ──┼──────────┼── h2 ──┼─────────────┤
         └──────────┘   h3 ──┘             
```

### Dimensionality & Tensor Flow

The network processes an input matrix X containing 3 samples and 2 features. The table below traces how data shapes transform across the network layers:

| Layer / Parameter | Variable | Matrix Shape | Description |
| --- | --- | --- | --- |
| **Input Matrix** | X | (3, 2) | 3 training samples, 2 normalized features each |
| **Hidden Weights** | w_h | (2, 3) | Maps 2 input features to 3 hidden neurons |
| **Hidden Biases** | b_h | (1, 3) | Row vector broadcasted across all 3 samples |
| **Hidden Pre-activation** | z_h | (3, 3) | Weighted sum before activation: X·w_h + b_h |
| **Hidden Activation** | h_layer_act | (3, 3) | Nonlinear features extracted by hidden layer |
| **Output Weights** | w_out | (3, 1) | Aggregates 3 hidden signals into 1 output prediction |
| **Output Bias** | b_out | (1, 1) | Scalar bias for the final regression output |
| **Output Pre-activation** | z_out | (3, 1) | Weighted sum: h_layer_act·w_out + b_out |
| **Final Prediction** | ŷ (y_hat) | (3, 1) | Continuous prediction vector matching target y |
| **Target Values** | y | (3, 1) | Ground truth values for supervised learning |

### Shape Propagation Trace

```
X:        (3, 2)  @  w_h:    (2, 3)  =  z_h:      (3, 3)
z_h:      (3, 3)  →  sigmoid  →  h_layer_act:  (3, 3)
h_layer_act: (3, 3)  @  w_out:  (3, 1)  =  z_out:    (3, 1)
z_out:    (3, 1)  →  sigmoid  →  ŷ:        (3, 1)
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- NumPy 1.19 or higher
- Matplotlib (optional, for visualization)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ann-from-scratch.git
cd ann-from-scratch

# Install dependencies
pip install numpy matplotlib

# Run the neural network
python ann.py
```

Or open the Jupyter Notebook for interactive exploration:

```bash
jupyter notebook ann.ipynb
```

### Expected Output

The script will train for a specified number of epochs and output:

- Initial loss value
- Loss at intermediate checkpoints (every 1000 epochs)
- Final loss after training completion
- Final predictions compared to target values
- Optional: loss curve visualization

---

## 🧠 Theoretical Foundations

### 1. Mathematical Abstraction of a Neuron

Inspired by biological neural pathways, an artificial neuron models dendritic inputs as a weighted sum (z), adds an axonal bias (b), and evaluates firing potential via a nonlinear activation function (f):

```
z = Σ(x_i · w_i) + b
output = f(z)
```

Where:
- x_i are input signals (features)
- w_i are synaptic weights
- b is the bias term (firing threshold)
- f is the activation function

### 2. The Perceptron & Linear Separability

A single-layer perceptron can only resolve linearly separable tasks. As famously demonstrated by the **XOR problem**, complex decision boundaries require non-linear compositions. Consider these logical operations:

| Operation | Linearly Separable? | Single Perceptron? |
|-----------|---------------------|---------------------|
| AND       | Yes                 | ✓ Solvable          |
| OR        | Yes                 | ✓ Solvable          |
| NOT       | Yes                 | ✓ Solvable          |
| XOR       | No                  | ✗ Unsolvable        |

Multi-layer architectures solve the XOR limitation by composing multiple linear separators into highly intricate, continuous decision boundaries. The hidden layer transforms the input space into a representation where classes become linearly separable.

### 3. Universal Approximation Theorem

> **Theorem (Cybenko, 1989; Hornik, 1991):** A feedforward network containing a single hidden layer with a finite number of neurons can approximate any continuous function on compact subsets of ℝⁿ, provided it utilizes a non-linear, bounded, and monotonically increasing activation function.

**Key implications:**
- A single hidden layer is theoretically sufficient for any continuous function approximation
- The theorem guarantees existence but does not specify the required number of neurons
- Practical networks often use multiple hidden layers (deep networks) for representational efficiency
- Non-linearity is essential — without it, multiple layers collapse into a single linear transformation

### 4. The Learning Problem

Neural network training is an optimization problem:

```
Given: Training data (X, y) and network architecture f(X; θ)
Find: Parameters θ = {w_h, b_h, w_out, b_out} that minimize loss L(ŷ, y)

where ŷ = f(X; θ) is the network's prediction
```

This is solved iteratively through gradient-based optimization.

---

## ⚡ Mathematical Mechanics

### Activation Functions

#### Sigmoid Activation

The network uses the **Sigmoid Activation Function**, which bounds any real value into a smooth probability curve between (0, 1):

**Function:**
```
f(x) = 1 / (1 + e^(-x))
```

**Properties:**
- Domain: (-∞, +∞)
- Range: (0, 1)
- f(0) = 0.5
- Monotonically increasing
- Smooth and differentiable everywhere

**Derivative (Sigmoid Prime):**
```
f'(x) = f(x) · (1 - f(x))
```

This elegant property — the derivative expressed in terms of the function itself — makes sigmoid computationally efficient for backpropagation.

#### ⚠️ The Vanishing Gradient Problem

When input values are extremely large or small:

```
x → +∞:  f(x) → 1,  f'(x) → 0
x → -∞:  f(x) → 0,  f'(x) → 0
x = 0:   f(x) = 0.5, f'(x) = 0.25 (maximum)
```

During backpropagation, gradients are multiplied through the chain rule. If any gradient approaches zero, it dampens the entire gradient chain, stalling parameter updates in earlier layers. This is why modern deep networks often prefer **ReLU** activation: f(x) = max(0, x) with derivative f'(x) = 1 for x > 0, maintaining gradient flow.

#### Sigmoid Activation Table

| Input (x) | f(x) | f'(x) | Gradient Strength |
|-----------|------|-------|-------------------|
| -5.0      | 0.007 | 0.006 | Near zero — vanishing |
| -2.0      | 0.119 | 0.105 | Weak |
| -1.0      | 0.269 | 0.197 | Moderate |
| 0.0       | 0.500 | 0.250 | Maximum |
| 1.0       | 0.731 | 0.197 | Moderate |
| 2.0       | 0.881 | 0.105 | Weak |
| 5.0       | 0.993 | 0.006 | Near zero — vanishing |

### Loss Function: Mean Squared Error (MSE)

For regression problems, MSE quantifies the average squared difference between predictions and targets:

**Function:**
```
MSE = (1/n) · Σ(y_i - ŷ_i)²
```

**Derivative:**
```
∂MSE/∂ŷ = -(2/n) · (y - ŷ)
```

**Properties:**
- Always non-negative (MSE ≥ 0)
- Heavily penalizes large errors (quadratic scaling)
- Differentiable everywhere
- Convex for linear models, non-convex for neural networks

---

## 🔁 The Training Cycle

```
[Normalized Input] → [Forward Pass] → [MSE Loss Calculation]
        ▲                                      │
        │                                      ▼
        └──────── [Weight Update] ←── [Gradient Computation]
                          (Backpropagation)
```

### Forward Propagation

Data flows linearly through consecutive matrix transformations. Each layer performs two operations: a linear transformation followed by a non-linear activation.

**Step 1: Hidden Layer Pre-activation**
```
z_h = X · w_h + b_h
```
Matrix multiplication aggregates input signals. Bias is broadcast across all samples.

**Step 2: Hidden Layer Activation**
```
h_layer_act = sigmoid(z_h)
```
Non-linear transformation extracts higher-level features.

**Step 3: Output Layer Pre-activation**
```
z_out = h_layer_act · w_out + b_out
```
Hidden features are aggregated into output prediction.

**Step 4: Output Layer Activation**
```
ŷ = sigmoid(z_out)
```
Final prediction squashed to (0, 1) range.

### Backpropagation & Parameter Updates

Using the calculus **Chain Rule**, errors are systematically calculated from the final output layer back to the initial inputs. The gradient of the loss with respect to each parameter tells us how to adjust that parameter to reduce the error.

**Step 1: Output Error Gradient**

The gradient of MSE loss with respect to output pre-activation:
```
d_output = -(y - ŷ) ⊙ sigmoid'(ŷ)
         = (y - ŷ) ⊙ ŷ ⊙ (1 - ŷ)   [for MSE + sigmoid combination]
```
⊙ denotes element-wise (Hadamard) multiplication.

**Step 2: Output Layer Parameter Gradients**
```
∂L/∂w_out = h_layer_actᵀ · d_output
∂L/∂b_out = sum(d_output, axis=0, keepdims=True)
```

**Step 3: Hidden Error Distribution**

Errors are distributed back through the output weights:
```
E_H = d_output · w_outᵀ
```
This tells us how much each hidden neuron contributed to the output error.

**Step 4: Hidden Error Gradient**
```
d_hidden = E_H ⊙ sigmoid'(h_layer_act)
         = E_H ⊙ h_layer_act ⊙ (1 - h_layer_act)
```

**Step 5: Hidden Layer Parameter Gradients**
```
∂L/∂w_h = Xᵀ · d_hidden
∂L/∂b_h = sum(d_hidden, axis=0, keepdims=True)
```

**Step 6: Parameter Updates (Gradient Descent)**

Weights and biases are shifted in the direction of the negative gradient, scaled by the learning rate (η):
```
w_out_new = w_out_old - η · ∂L/∂w_out
b_out_new = b_out_old - η · ∂L/∂b_out
w_h_new   = w_h_old   - η · ∂L/∂w_h
b_h_new   = b_h_old   - η · ∂L/∂b_h
```

### The Chain Rule Visualization

The gradient flows backward through the computation graph:

```
L = MSE(ŷ, y)
    ↑
ŷ = sigmoid(z_out)
    ↑
z_out = h_layer_act · w_out + b_out
    ↑
h_layer_act = sigmoid(z_h)
    ↑
z_h = X · w_h + b_h

Chain rule:
∂L/∂w_h = ∂L/∂ŷ · ∂ŷ/∂z_out · ∂z_out/∂h_layer_act · ∂h_layer_act/∂z_h · ∂z_h/∂w_h
```

---

## 💻 Implementation Details

### Complete Algorithm Pseudocode

```
Algorithm: Three-Layer Neural Network Training

Input: Training data X (3×2), targets y (3×1)
Parameters: learning_rate η = 0.1, epochs = 5000

1. Normalize input: X = X / max(X)
2. Normalize targets: y = y / 100
3. Initialize weights randomly: w_h ~ Uniform(0,1), w_out ~ Uniform(0,1)
4. Initialize biases to zero: b_h = zeros(1,3), b_out = zeros(1,1)

5. For epoch in 1 to epochs:
   a. // Forward Pass
      z_h = X · w_h + b_h
      h_layer_act = sigmoid(z_h)
      z_out = h_layer_act · w_out + b_out
      ŷ = sigmoid(z_out)
   
   b. // Loss Calculation
      error = y - ŷ
      loss = mean(error²)
   
   c. // Backward Pass (Backpropagation)
      d_output = error * sigmoid'(ŷ)
      grad_w_out = h_layer_actᵀ · d_output
      grad_b_out = sum(d_output)
      
      E_H = d_output · w_outᵀ
      d_hidden = E_H * sigmoid'(h_layer_act)
      grad_w_h = Xᵀ · d_hidden
      grad_b_h = sum(d_hidden)
   
   d. // Parameter Update
      w_out = w_out + η · grad_w_out
      b_out = b_out + η · grad_b_out
      w_h = w_h + η · grad_w_h
      b_h = b_h + η · grad_b_h
   
   e. // Logging
      If epoch % 1000 == 0: print loss

6. Denormalize predictions: ŷ_final = ŷ * 100
7. Return trained parameters and final predictions
```

### Python Implementation Structure

```
ann.py
├── sigmoid(x)              # Activation function
├── sigmoid_derivative(x)   # Derivative for backpropagation
├── mse(y_true, y_pred)     # Loss function
├── initialize_parameters() # Weight initialization
├── forward_pass()          # Forward propagation
├── backward_pass()         # Backpropagation
├── update_parameters()     # Gradient descent step
├── train()                 # Main training loop
└── main()                  # Entry point
```

---

## 📊 Optimization & Data Engineering

### Batch Gradient Descent

This implementation uses **Batch Gradient Descent**, which computes gradients using all training samples simultaneously:

**Advantages:**
- Deterministic updates — same data produces same gradient
- Smooth, monotonic convergence curves
- Stable gradient estimates (no sampling noise)

**Disadvantages:**
- Computationally expensive for large datasets
- Memory requirements scale with dataset size
- No stochastic exploration of the loss landscape

**Comparison with variants:**

| Optimizer | Gradient Source | Noise | Convergence Speed | Memory Usage |
|-----------|----------------|-------|-------------------|--------------|
| Batch GD | All samples | None | Slow per epoch | High |
| Stochastic GD | Single sample | High | Fast updates | Low |
| Mini-batch GD | Subset of samples | Moderate | Balanced | Moderate |

### Symmetry Breaking: Weight Initialization

Weights are initialized randomly using a **Uniform Distribution** to ensure different neurons extract distinct patterns:

```
w_h ~ Uniform(0, 1)
w_out ~ Uniform(0, 1)
```

**Why random initialization matters:**
If all weights were initialized to the same value (e.g., all zeros), each neuron in a layer would receive identical gradients and learn identical features — wasting the representational capacity of having multiple neurons. Random initialization breaks this symmetry, allowing each neuron to specialize in detecting different patterns.

**Why biases can be zero-initialized:**
Bias terms don't cause symmetry problems because they're added after the weighted sum. Even with zero initialization, the random weights ensure different neurons receive different total inputs.

### Feature Scaling

Input attributes are scaled relative to their column-wise maximums:

```
X_normalized = X / X.max(axis=0)
```

This ensures all features are in the range [0, 1], providing:

- **Isotropic loss landscape:** Equal learning rates affect all weight dimensions similarly
- **Numerical stability:** Prevents extremely large activations that saturate sigmoid
- **Faster convergence:** Gradient descent converges more efficiently on well-scaled features

### Target Normalization

Target values (y ∈ [86, 92]) are compressed to [0.86, 0.92] by dividing by 100:

```
y_normalized = y / 100
```

**Why this matters:**
The sigmoid output function produces values in (0, 1). If targets were in [86, 92], the network could never reach them — the best it could do is saturate near 1.0, and gradients would vanish. Normalizing targets to the sigmoid's output range ensures:

- The network can actually achieve the target values
- Gradients remain non-zero throughout training
- The loss function properly measures prediction quality

---

## 📈 Results & Visualization

### Expected Training Progress

With proper implementation, training should produce a monotonically decreasing loss curve:

```
Epoch 0:     Loss = 0.1523
Epoch 1000:  Loss = 0.0218
Epoch 2000:  Loss = 0.0087
Epoch 3000:  Loss = 0.0034
Epoch 4000:  Loss = 0.0015
Epoch 5000:  Loss = 0.0008
```

### Interpreting the Results

- **Decreasing loss** indicates the network is learning
- **Plateauing loss** suggests convergence or a local minimum
- **Oscillating loss** may indicate too high a learning rate
- **Immediate convergence to NaN** indicates exploding gradients or numerical issues

### Adding Visualization

To visualize training progress, add after the training loop:

```python
import matplotlib.pyplot as plt

# Collect loss values during training
loss_history = []

# Inside training loop:
loss_history.append(loss)

# After training:
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Training Loss Over Time')
plt.yscale('log')  # Log scale often reveals patterns
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🚀 Scalability & Extensions

While this 3-layer architecture serves as an excellent proof-of-concept, practical enterprise networks extend these fundamentals with modern techniques:

### Activation Function Upgrades

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
f'(x) = 0 if x < 0, 1 if x > 0
```
- Mitigates vanishing gradients for positive inputs
- Computationally simpler than sigmoid
- Can suffer from "dying ReLU" where neurons output zero permanently

**Leaky ReLU:**
```
f(x) = max(0.01x, x)
```
- Small gradient for negative inputs prevents dying neurons

### Regularization Techniques

**L2 Weight Decay:**
```
Loss_total = MSE + λ · Σ(w²)
```
Penalizes large weights, encouraging simpler models that generalize better.

**Dropout:**
During training, randomly deactivate a fraction of neurons each forward pass. This prevents co-adaptation and acts as ensemble learning within a single network.

### Advanced Optimizers

**Adam (Adaptive Moment Estimation):**
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t        # Momentum (first moment)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²       # Velocity (second moment)
θ_t = θ_{t-1} - η · m_t / (√v_t + ε)  # Parameter update
```
Combines momentum and adaptive learning rates for faster, more stable convergence.

**RMSprop:**
```
v_t = β·v_{t-1} + (1-β)·g_t²
θ_t = θ_{t-1} - η · g_t / √(v_t + ε)
```
Adapts learning rates per-parameter based on recent gradient magnitudes.

### Deep Architectures

Adding more hidden layers enables hierarchical feature learning:

```
Input → Hidden₁ (ReLU) → Hidden₂ (ReLU) → Hidden₃ (ReLU) → Output
```

Deep networks can learn more abstract representations with fewer total neurons than wide, shallow networks.

### Classification Extension

Convert the regression network to binary classification:

```python
# Replace MSE with Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + 
                    (1 - y_true) * np.log(1 - y_pred + 1e-8))

# The output sigmoid naturally produces class probabilities
# Threshold at 0.5 for binary decisions
predictions = (y_hat > 0.5).astype(int)
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Loss doesn't decrease | Learning rate too low | Increase η (try 0.5 or 1.0) |
| Loss oscillates wildly | Learning rate too high | Decrease η (try 0.01 or 0.001) |
| Loss immediately goes to NaN | Exploding gradients | Check weight initialization scale, normalize inputs |
| All predictions are similar | Symmetry not broken | Verify random weight initialization |
| Predictions stuck near 0.5 | Vanishing gradients | Check target normalization, consider ReLU |
| Shape mismatch errors | Matrix dimension bug | Trace shapes using the Dimensionality table above |

### Debugging Checklist

1. **Print shapes at each layer:** Verify matrix multiplication compatibility
2. **Check value ranges:** Sigmoid inputs should not be extreme (±10+ causes saturation)
3. **Verify gradient magnitudes:** Gradients near zero indicate vanishing gradient problem
4. **Test with a single sample:** Overfit to one example to verify backpropagation works
5. **Numerical gradient checking:** Compare analytical gradients to finite difference approximations

---


