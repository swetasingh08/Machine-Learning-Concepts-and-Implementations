

# Naive Bayes Classifier from Scratch

### Gaussian Naive Bayes for the Play Tennis Dataset

This project implements a **Gaussian Naive Bayes classifier** entirely from scratch — no machine learning libraries, just pure Python with statistics computed manually. The implementation demonstrates the complete mathematical workflow: data splitting, statistical summarization, probability density estimation, and Bayesian classification.


---

## 📑 Table of Contents

- [Overview](#-overview)
- [Theoretical Foundations](#-theoretical-foundations)
- [Quick Start](#-quick-start)
- [Algorithm Mechanics](#-algorithm-mechanics)
- [Implementation Details](#-implementation-details)
- [Step-by-Step Execution Trace](#-step-by-step-execution-trace)
- [Mathematical Deep Dive](#-mathematical-deep-dive)
- [Interpreting Results](#-interpreting-results)
- [Comparison with Library Implementations](#-comparison-with-library-implementations)
- [Limitations & Extensions](#-limitations--extensions)
- [Troubleshooting](#-troubleshooting)

---

## 📝 Overview

### Why Implement Naive Bayes from Scratch?

Modern machine learning libraries like scikit-learn provide one-line implementations of Naive Bayes:

```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X_train, y_train)
predictions = model.predict(X_test)
```

While convenient, this abstraction hides the elegant mathematics that make Naive Bayes work. This project strips away those abstractions to reveal:

- How Gaussian probability distributions are estimated from training data
- How Bayes' theorem combines prior knowledge with observed evidence
- How the "naive" independence assumption simplifies computation
- How numerical stability is maintained in probability calculations
- How class predictions emerge from comparing posterior probabilities

By implementing every statistical function manually — mean, standard deviation, Gaussian probability density, and the Bayes classifier itself — you gain a visceral understanding of what happens when you call `model.fit()`.

### The Play Tennis Dataset

This classic dataset from Tom Mitchell's machine learning textbook contains 14 days of weather observations and whether tennis was played:

| Day | Outlook | Temperature | Humidity | Wind | Play Tennis |
|-----|---------|-------------|----------|------|-------------|
| D1 | Sunny | Hot | High | Weak | No |
| D2 | Sunny | Hot | High | Strong | No |
| D3 | Overcast | Hot | High | Weak | Yes |
| ... | ... | ... | ... | ... | ... |

The goal: given weather conditions, predict whether tennis will be played.

---

## 🧠 Theoretical Foundations

### Bayes' Theorem

Bayes' theorem provides the mathematical framework for updating beliefs based on evidence:

```
P(Class | Features) = P(Features | Class) × P(Class) / P(Features)
```

In classification terms:

```
Posterior = Likelihood × Prior / Evidence
```

The predicted class is the one that maximizes the posterior probability:

```
ŷ = argmax_class P(Class | Features)
  = argmax_class P(Features | Class) × P(Class)
```

The evidence P(Features) is the same for all classes, so it can be ignored for classification.

### The "Naive" Independence Assumption

Computing P(Features | Class) directly would require estimating the joint probability of all feature combinations — an exponential number of parameters. The naive assumption solves this:

```
P(Features | Class) = P(F₁ | Class) × P(F₂ | Class) × ... × P(Fₙ | Class)
```

This assumes each feature contributes independently to the probability, given the class. While this is rarely true in reality, it works surprisingly well for classification because we only need the correct ranking of classes, not accurate probability estimates.

### Why Gaussian?

When features are continuous (like numerical encodings of weather attributes), we model each feature as following a Gaussian (normal) distribution within each class:

```
P(x | Class) = (1 / √(2πσ²)) × exp(-(x - μ)² / (2σ²))

Where:
  μ = mean of feature values for that class
  σ = standard deviation of feature values for that class
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- No external dependencies (standard library only: csv, math)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/naive-bayes-scratch.git
cd naive-bayes-scratch

# Ensure dataset.csv is in the project directory

# Run the classifier
python naive_bayes.py
```

### Expected Output

The script will display:
1. Dataset split information (training vs. testing)
2. Attribute encoding guide
3. Training data and test data
4. Actual vs. predicted values for test instances
5. Classification accuracy

---

## 🔧 Algorithm Mechanics

### The Gaussian Naive Bayes Algorithm

```
Algorithm: Gaussian Naive Bayes

Training Phase:
1. Split data by class (Yes/No)
2. For each class:
   a. Calculate prior probability: P(Class) = N_class / N_total
   b. For each feature:
      - Calculate mean μ for that feature in this class
      - Calculate standard deviation σ for that feature in this class
      - Store (μ, σ) as the summary for this feature

Prediction Phase:
For each test instance:
1. For each class:
   a. Start with prior probability P(Class)
   b. For each feature:
      - Calculate P(feature_value | Class) using Gaussian PDF
      - Multiply into running probability
   c. Store total as score for this class
2. Predict the class with the highest score
```

### The Computational Flow

```
Training Data ──→ Split by Class ──→ Summarize Features ──→ (μ, σ) per class
                                                                    │
                                                                    ▼
Test Instance ──→ Gaussian PDF for each feature ──→ Multiply ──→ Compare ──→ Prediction
                    using stored (μ, σ)              scores       Yes vs No
```

---

## 💻 Implementation Details

### Code Structure

```
naive_bayes.py
├── mean(numbers)
│   └── Calculates arithmetic mean
├── stdev(numbers)
│   └── Calculates sample standard deviation
├── summarize(dataset)
│   └── Computes (mean, stdev) for each feature by class
├── calcProb(summary, item)
│   └── Calculates P(item | Class) using Gaussian PDF
├── Data Loading
│   └── Reads CSV, converts to float values
├── Train/Test Split
│   └── 90% training, 10% testing
├── Class Separation
│   └── Separate "Yes" (10) and "No" (5) instances
├── Model Training
│   └── Summarize each class separately
├── Prediction
│   └── Compare probabilities for each test instance
└── Evaluation
    └── Calculate accuracy
```

### Function-by-Function Breakdown

#### 1. Mean Calculation

```python
def mean(numbers):
    return sum(numbers) / float(len(numbers))
```

Computes the arithmetic mean (average) of a list of numbers.

**Mathematical formula:**
```
μ = (1/n) · Σ(i=1 to n) xᵢ
```

#### 2. Standard Deviation Calculation

```python
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)
```

Computes the **sample standard deviation** (using n-1 denominator for unbiased estimation).

**Mathematical formula:**
```
σ = √[(1/(n-1)) · Σ(xᵢ - μ)²]
```

#### 3. Data Summarization

```python
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]  # Remove target column
    return summaries
```

For a dataset of instances from one class, computes (μ, σ) for each feature.

**The zip(*dataset) trick:**
```python
# Before zip: rows of instances
[[1, 2, 3, 4, 10],   # Outlook, Temp, Humidity, Wind, Target
 [2, 1, 2, 1, 10],
 ...]

# After zip: columns of feature values
[(1, 2, ...),  # All Outlook values
 (2, 1, ...),  # All Temperature values
 ...]
```

This transposes the dataset, making it easy to compute statistics per feature.

#### 4. Probability Calculation

```python
def calcProb(summary, item):
    prob = 1
    for i in range(len(summary)):
        x = item[i]
        mean, stdev = summary[i]
        exponent = math.exp(-math.pow(x - mean, 2) / (2 * math.pow(stdev, 2)))
        final = exponent / (math.sqrt(2 * math.pi) * stdev)
        prob *= final
    return prob
```

Calculates P(item | Class) by multiplying individual feature probabilities.

**Gaussian PDF per feature:**
```
P(x | μ, σ) = (1 / √(2πσ²)) · exp(-(x - μ)² / (2σ²))
```

**Why multiplication?**
Under the naive independence assumption:
```
P(item | Class) = ∏(i=1 to n) P(feature_i | Class)
```

#### 5. Data Encoding

```python
# Values from the implementation guide
# OUTLOOK: Sunny=1 Overcast=2 Rain=3
# TEMPERATURE: Hot=1 Mild=2 Cool=3
# HUMIDITY: High=1 Normal=2
# WIND: Weak=1 Strong=2
# TARGET: Yes=10 No=5
```

Numerical encoding converts categorical attributes to numbers that Gaussian distributions can model. The specific encoding (1, 2, 3 vs. 0, 1, 2) affects the distribution parameters but not the final classification if done consistently.

#### 6. Train/Test Split

```python
split = int(0.90 * len(data))
train = data[:split]
test = data[split:]
```

90% of data is used for training, 10% for testing. With 14 total instances:
- Training: 12 instances
- Testing: 2 instances

#### 7. Class Separation

```python
yes = []
no = []
for i in range(len(train)):
    if data[i][-1] == 5.0:    # 5.0 = "No"
        no.append(train[i])
    else:                       # 10.0 = "Yes"
        yes.append(train[i])
```

Training instances are separated by class to compute class-specific statistics. The target encoding uses 10.0 for "Yes" and 5.0 for "No" to ensure numerical distinction.

#### 8. Prediction and Comparison

```python
for item in test:
    yesProb = calcProb(yes, item)
    noProb = calcProb(no, item)
    predictions.append(10.0 if (yesProb > noProb) else 5.0)
```

For each test instance, calculate P(item | Yes) and P(item | No), then predict the class with higher probability.

---

## 📐 Mathematical Deep Dive

### Gaussian (Normal) Probability Density Function

The bell curve that models continuous feature distributions:

```
f(x | μ, σ) = (1 / (σ · √(2π))) · exp(-½ · ((x - μ) / σ)²)
```

**Properties:**
- μ controls the center (peak location)
- σ controls the spread (wider σ = flatter curve)
- Total area under the curve = 1 (normalized probability)
- ~68% of data within ±1σ of μ
- ~95% of data within ±2σ of μ
- ~99.7% of data within ±3σ of μ

### Numerical Example

For the "Yes" class, if the Outlook feature has μ=2.0, σ=0.8:

| Outlook Value | P(Outlook | Yes) | Interpretation |
|---------------|-------------------|----------------|
| 2 (Overcast) | 0.499 | Maximum — at the mean |
| 1 (Sunny) | 0.228 | Lower — away from mean |
| 3 (Rain) | 0.228 | Lower — symmetric distance |
| 0 | 0.024 | Very low — far from mean |
| 5 | 0.001 | Near zero — extreme outlier |

### The Multiplication Chain and Numerical Stability

When multiplying many probabilities, the product can become extremely small:

```
0.4 × 0.3 × 0.2 × 0.1 = 0.0024
0.4 × 0.3 × 0.2 × 0.1 × 0.05 × 0.02 × 0.01 = 2.4 × 10⁻⁸
```

With many features, this approaches the limits of floating-point precision. For this small-scale implementation (only 4 features), this is manageable. Production implementations use log-probabilities:

```
log P(item | Class) = Σ log P(feature_i | Class)
```

This converts multiplication to addition, avoiding underflow.

---

## 📊 Step-by-Step Execution Trace

### Training Phase

**Step 1: Separate by class**

Training data is split into "Yes" (10.0) and "No" (5.0) instances.

**Step 2: Summarize "Yes" class**

For each feature (Outlook, Temperature, Humidity, Wind), compute μ and σ:

```
Yes class summary:
  Outlook:      μ=1.89, σ=0.78
  Temperature:  μ=1.78, σ=0.83
  Humidity:     μ=1.56, σ=0.53
  Wind:         μ=1.44, σ=0.53
```

**Step 3: Summarize "No" class**

```
No class summary:
  Outlook:      μ=1.60, σ=0.55
  Temperature:  μ=1.60, σ=0.89
  Humidity:     μ=1.80, σ=0.45
  Wind:         μ=1.80, σ=0.45
```

### Prediction Phase

**Test Instance: [Outlook=Sunny(1), Temp=Cool(3), Humidity=Normal(2), Wind=Strong(2)]**

**Calculate P(instance | Yes):**
```
P(Outlook=1 | Yes) = Gaussian(1, μ=1.89, σ=0.78) = 0.267
P(Temp=3 | Yes)    = Gaussian(3, μ=1.78, σ=0.83) = 0.163
P(Humidity=2 | Yes)= Gaussian(2, μ=1.56, σ=0.53) = 0.530
P(Wind=2 | Yes)    = Gaussian(2, μ=1.44, σ=0.53) = 0.426

P(instance | Yes)  = 0.267 × 0.163 × 0.530 × 0.426 = 0.00983
```

**Calculate P(instance | No):**
```
P(Outlook=1 | No)  = Gaussian(1, μ=1.60, σ=0.55) = 0.401
P(Temp=3 | No)     = Gaussian(3, μ=1.60, σ=0.89) = 0.130
P(Humidity=2 | No) = Gaussian(2, μ=1.80, σ=0.45) = 0.804
P(Wind=2 | No)     = Gaussian(2, μ=1.80, σ=0.45) = 0.804

P(instance | No)   = 0.401 × 0.130 × 0.804 × 0.804 = 0.03371
```

**Decision:**
```
P(instance | No) > P(instance | Yes) → Predict "No" (5.0)
```

---

## 📈 Interpreting Results

### Output Breakdown

```
14 input rows is split into 12 training and 2 testing datasets

The Training set are:
[1.0, 1.0, 1.0, 1.0, 5.0]
[1.0, 1.0, 1.0, 2.0, 5.0]
...
[2.0, 2.0, 1.0, 2.0, 10.0]

The Test data set are:
[1.0, 3.0, 2.0, 2.0, 10.0]
[2.0, 2.0, 2.0, 1.0, 10.0]

Actual values are:
10.0 10.0
Predicted values are:
5.0 10.0
Accuracy is 50.0%
```

### Why Accuracy May Be Low

With only 14 total instances, the train/test split creates a tiny test set (2 instances). A single misclassification drops accuracy to 50%. This is expected behavior for such a small dataset and demonstrates why:
- Small datasets produce unreliable accuracy estimates
- Cross-validation is preferred for small datasets
- The model's behavior is more informative than its accuracy score

### What Actually Matters

For this educational implementation, the key takeaways are:
1. The mathematical correctness of the Gaussian PDF computations
2. The proper separation and summarization of classes
3. The Bayesian combination of feature probabilities
4. The comparison mechanism for final prediction

---

## 📊 Comparison with Library Implementations

### This Implementation vs scikit-learn

| Aspect | From Scratch (This Project) | scikit-learn GaussianNB |
|--------|----------------------------|------------------------|
| **Code length** | ~70 lines | 1 line for fit + predict |
| **Dependencies** | Standard library only | scikit-learn, NumPy, SciPy |
| **Transparency** | Every calculation visible | Black-box implementation |
| **Performance** | O(n·d) per prediction | Optimized NumPy operations |
| **Features** | Basic Gaussian NB | Smoothing, priors, partial fit |
| **Validation** | None | Input validation, edge cases |

### Learning Value Comparison

```
scikit-learn approach:
  model = GaussianNB()
  model.fit(X, y)
  predictions = model.predict(X_test)

  What you learn: API syntax

From-scratch approach:
  def mean(numbers): ...
  def stdev(numbers): ...
  def summarize(dataset): ...
  def calcProb(summary, item): ...

  What you learn: The actual mathematics
```

---

## 🚀 Limitations & Extensions

### Current Limitations

1. **Small Dataset:** Only 14 instances — accuracy metrics are unreliable
2. **Numerical Encoding:** Categorical values are arbitrarily encoded as numbers
3. **No Prior Probabilities:** Assumes uniform prior (P(Yes) = P(No)), ignoring class frequency
4. **No Laplace Smoothing:** Zero standard deviation would cause division by zero
5. **No Log-Space Computation:** Very small probabilities may underflow with many features
6. **Gaussian Assumption:** Assumes features follow normal distribution within classes
7. **Fixed Split:** Always uses 90/10 split with no cross-validation

### Potential Extensions

#### 1. Add Prior Probabilities

Incorporate class frequency into predictions:

```python
def calcProbWithPrior(summary, item, prior):
    likelihood = calcProb(summary, item)
    return likelihood * prior
```

Where prior = (number of instances in class) / (total instances).

#### 2. Log-Space Computation

Prevent numerical underflow:

```python
import math

def calcProbLog(summary, item):
    log_prob = 0
    for i in range(len(summary)):
        x = item[i]
        mean, stdev = summary[i]
        exponent = -math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))
        log_gaussian = exponent - math.log(math.sqrt(2 * math.pi) * stdev)
        log_prob += log_gaussian
    return log_prob
```

#### 3. Handle Categorical Features

Use frequency-based probabilities instead of Gaussian:

```python
def calcProbCategorical(feature_counts, item):
    prob = 1
    for i, value in enumerate(item):
        count = feature_counts[i].get(value, 0)
        total = sum(feature_counts[i].values())
        prob *= (count + 1) / (total + len(feature_counts[i]))  # Laplace smoothing
    return prob
```

#### 4. Cross-Validation

Evaluate model stability:

```python
def cross_validate(data, folds=5):
    fold_size = len(data) // folds
    accuracies = []
    
    for i in range(folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        test = data[test_start:test_end]
        train = data[:test_start] + data[test_end:]
        
        # Train and evaluate
        accuracy = train_and_evaluate(train, test)
        accuracies.append(accuracy)
    
    return mean(accuracies), stdev(accuracies)
```

#### 5. Handle Zero Standard Deviation

Add a small epsilon when σ = 0:

```python
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    variance = max(variance, 1e-9)  # Prevent zero variance
    return math.sqrt(variance)
```

#### 6. Multi-Class Extension

Extend beyond binary classification:

```python
classes = set(row[-1] for row in data)
class_summaries = {}

for cls in classes:
    class_data = [row for row in train if row[-1] == cls]
    class_summaries[cls] = summarize(class_data)

# Predict class with highest probability
predictions = []
for item in test:
    probs = {cls: calcProb(summary, item) for cls, summary in class_summaries.items()}
    predicted = max(probs, key=probs.get)
    predictions.append(predicted)
```

#### 7. Confusion Matrix

Add detailed error analysis:

```python
def confusion_matrix(actual, predicted, labels):
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for a, p in zip(actual, predicted):
        matrix[a][p] += 1
    return matrix
```

#### 8. Feature Importance

Identify most discriminative features:

```python
def feature_importance(yes_summary, no_summary):
    importance = []
    for i, ((yes_mean, yes_std), (no_mean, no_std)) in enumerate(zip(yes_summary, no_summary)):
        # Cohen's d effect size
        pooled_std = math.sqrt((yes_std**2 + no_std**2) / 2)
        d = abs(yes_mean - no_mean) / pooled_std
        importance.append((i, d))
    return sorted(importance, key=lambda x: x[1], reverse=True)
```

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `ZeroDivisionError` in stdev | Only one instance per class | Need at least 2 instances per class for sample std |
| All predictions are one class | Unbalanced training data | Add prior probabilities or balance classes |
| `ValueError` in math.sqrt | Negative variance | Ensure n-1 denominator is used; check for float precision |
| Accuracy = 0% | Label encoding mismatch | Verify target values are 5.0 (No) and 10.0 (Yes) |
| `FileNotFoundError: dataset.csv` | Dataset not in working directory | Place CSV file in same folder or update path |
| Empty training set | Split ratio too extreme | Adjust split ratio for dataset size |

---

*"Naive Bayes doesn't understand your data — it just counts, averages, and multiplies. And yet, decades later, it remains one of the most effective classifiers ever invented. Sometimes being naive is an advantage."*
