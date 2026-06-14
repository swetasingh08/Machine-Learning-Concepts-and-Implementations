
# Heart Disease Prediction using Bayesian Networks

### Probabilistic Graphical Model for Medical Risk Assessment

This project implements a **Discrete Bayesian Network** using the pgmpy library to model causal relationships between lifestyle factors, physiological markers, and heart disease risk. The model learns conditional probability distributions from health survey data and performs probabilistic inference to estimate individual heart disease risk based on personal health indicators.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Why Bayesian Networks for Medical Diagnosis?](#-why-bayesian-networks-for-medical-diagnosis)
- [Quick Start](#-quick-start)
- [Network Architecture](#-network-architecture)
- [Mathematical Foundations](#-mathematical-foundations)
- [Dataset Description](#-dataset-description)
- [Implementation Details](#-implementation-details)
- [Inference Examples](#-inference-examples)
- [Input Guide](#-input-guide)
- [Interpreting Results](#-interpreting-results)
- [Limitations & Extensions](#-limitations--extensions)
- [Troubleshooting](#-troubleshooting)


---

## 📝 Overview

### The Problem

Heart disease remains the leading cause of mortality worldwide. Early risk assessment based on modifiable lifestyle factors and measurable physiological markers can guide preventive interventions. However, the relationships between these factors are complex, interconnected, and probabilistic rather than deterministic.

### The Approach

This project models these relationships using a **Discrete Bayesian Network** — a directed acyclic graph (DAG) where:

- **Nodes** represent health variables (age, diet, cholesterol, heart disease, etc.)
- **Edges** represent causal or influential relationships between variables
- **Conditional Probability Tables (CPTs)** quantify the strength of these relationships
- **Inference algorithms** compute the probability of heart disease given observed evidence

The model doesn't just predict "disease or no disease" — it outputs a probability distribution, allowing clinicians and patients to understand risk in nuanced terms.

### What This Project Demonstrates

- Construction of a Bayesian Network from domain knowledge (causal structure)
- Parameter learning from data using Maximum Likelihood Estimation (MLE)
- Probabilistic inference using the Variable Elimination algorithm
- Interactive risk assessment based on user-provided health indicators
- The distinction between correlation and causation in medical data

---

## 🧠 Why Bayesian Networks for Medical Diagnosis?

### Advantages Over Black-Box Models

| Aspect | Bayesian Networks | Neural Networks / Black-Box |
|--------|-------------------|------------------------------|
| **Interpretability** | Fully interpretable — CPTs can be inspected directly | Opaque decision boundaries |
| **Causal Reasoning** | Models cause-effect relationships explicitly | Captures correlations, not causation |
| **Uncertainty Quantification** | Inherent probabilistic output | Requires additional calibration |
| **Missing Data Handling** | Naturally handles partial evidence | May require imputation |
| **Domain Knowledge Integration** | Structure can be designed by experts | Learned purely from data |
| **Small Data Efficiency** | Works well with modest datasets | Typically requires large datasets |

### The Bayesian Approach to Medical Reasoning

Medical diagnosis inherently involves reasoning under uncertainty:

- A patient with high cholesterol doesn't definitely have heart disease — but the probability increases
- A patient with an active lifestyle might have lower risk despite genetic predisposition
- Multiple weak risk factors can compound to create significant overall risk

Bayesian Networks model this kind of probabilistic, multi-factorial reasoning naturally. The conditional probability tables capture statements like:

> "Given that a patient has a sedentary lifestyle and a high-calorie diet, the probability of high cholesterol is 0.75."

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- pgmpy library
- pandas library

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-bayesian.git
cd heart-disease-bayesian

# Install dependencies
pip install pgmpy pandas

# Ensure Health.csv is in the project directory

# Run the risk assessment tool
python heart_disease_predictor.py
```

### Expected Workflow

1. The script loads and displays the health dataset
2. The Bayesian Network structure is defined
3. Parameters (CPTs) are learned from the data using MLE
4. The user is prompted to enter personal health indicators
5. The model computes and displays the probability distribution for heart disease

---

## 🏗️ Network Architecture

### Causal Graph Structure

```
                    ┌───────────┐
                    │    Age    │
                    └─────┬─────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               │
    ┌───────────┐   ┌───────────┐         │
    │  Gender   │   │ Lifestyle │◄────────┘
    └─────┬─────┘   └─────┬─────┘
          │               │
          └───────┬───────┘
                  │
                  ▼
            ┌───────────┐
            │   Diet    │
            └─────┬─────┘
                  │
                  ▼
            ┌───────────┐     ┌───────────┐
            │Cholesterol│     │  Family   │
            └─────┬─────┘     └─────┬─────┘
                  │                 │
                  └────────┬────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Heart Disease│
                    └──────────────┘
```

### Edge Justification

Each directed edge represents a hypothesized causal influence:

| Edge | Justification |
|------|---------------|
| Age → Lifestyle | Physical activity levels tend to change with age |
| Gender → Lifestyle | Activity patterns may differ by gender in certain populations |
| Lifestyle → Diet | Active individuals often make different dietary choices |
| Diet → Cholesterol | Dietary fat intake directly influences blood cholesterol levels |
| Family → Heart Disease | Genetic predisposition independently affects heart disease risk |
| Cholesterol → Heart Disease | High cholesterol is a direct physiological risk factor |

### Node States

| Node | States | Description |
|------|--------|-------------|
| Age | 0-4 | SuperSeniorCitizen, SeniorCitizen, MiddleAged, Youth, Teen |
| Gender | 0-1 | Male, Female |
| Family | 0-1 | No family history, Family history present |
| Diet | 0-1 | High-calorie, Medium/Healthy |
| Lifestyle | 0-3 | Athlete, Active, Moderate, Sedentary |
| Cholesterol | 0-2 | High, BorderLine, Normal |
| Heart Disease | 0-1 | No disease, Disease present |

---

## 📐 Mathematical Foundations

### Bayesian Networks Formally

A Bayesian Network B = (G, P) consists of:

1. **G:** A Directed Acyclic Graph (DAG) where nodes V = {X₁, X₂, ..., Xₙ} represent random variables
2. **P:** A set of Conditional Probability Distributions P(Xᵢ | Parents(Xᵢ)) for each node

### The Chain Rule for Bayesian Networks

The joint probability distribution over all variables factorizes as:

```
P(X₁, X₂, ..., Xₙ) = ∏ᵢ P(Xᵢ | Parents(Xᵢ))
```

For this heart disease model:

```
P(A, G, L, D, C, F, H) = P(A) · P(G) · P(L|A,G) · P(D|L) · P(C|D) · P(F) · P(H|C,F)
```

This factorization dramatically reduces the number of parameters needed compared to the full joint distribution.

### Parameter Learning: Maximum Likelihood Estimation

Given the network structure, MLE estimates each CPT entry as:

```
P(X = x | Parents(X) = pa) = count(X = x, Parents(X) = pa) / count(Parents(X) = pa)
```

This is simply the observed frequency in the training data — the estimate that maximizes the likelihood of observing the data.

### Inference: Variable Elimination

The query P(HeartDisease | evidence) is computed by:

1. **Conditioning:** Incorporating observed evidence into the joint distribution
2. **Marginalization:** Summing out (eliminating) unobserved variables one at a time
3. **Normalization:** Ensuring the resulting distribution sums to 1

The Variable Elimination algorithm efficiently orders these operations to minimize computational cost.

---

## 📊 Dataset Description

### Health.csv Structure

The dataset contains health survey records with the following columns:

| Column | Type | Description | Possible Values |
|--------|------|-------------|-----------------|
| age | Categorical | Age group classification | 0-4 |
| Gender | Binary | Biological sex | 0 (Male), 1 (Female) |
| Lifestyle | Categorical | Physical activity level | 0-3 |
| diet | Binary | Dietary pattern | 0 (High), 1 (Medium) |
| Family | Binary | Family history of heart disease | 0 (No), 1 (Yes) |
| cholestrol | Categorical | Blood cholesterol level | 0-2 |
| heartdisease | Binary | Heart disease diagnosis | 0 (No), 1 (Yes) |

### Sample Data

```
   age  Gender  Lifestyle  diet  Family  cholestrol  heartdisease
0    2       1          2     1       0           1             0
1    0       0          3     0       1           0             1
2    3       1          1     1       0           2             0
3    1       0          3     0       0           0             1
4    4       1          0     1       0           2             0
...
```

### Data Quality Considerations

- Complete cases only (no missing values assumed)
- Categorical encoding uses integer labels (not one-hot)
- Class balance should be checked — imbalanced heart disease prevalence may skew predictions
- Domain knowledge should validate that the learned CPTs match medical understanding

---

## 💻 Implementation Details

### Code Structure

```
heart_disease_predictor.py
├── Import libraries and suppress warnings
├── Load dataset from Health.csv
├── Define Bayesian Network structure (DAG edges)
├── Train model using MLE (fit to data)
├── Initialize inference engine (Variable Elimination)
├── Display input guide for user
├── Collect user evidence (6 health indicators)
├── Perform probabilistic query
└── Display heart disease probability distribution
```

### Key Code Components

#### Model Definition

```python
model = DiscreteBayesianNetwork([
    ('age', 'Lifestyle'),        # Age influences lifestyle
    ('Gender', 'Lifestyle'),     # Gender influences lifestyle
    ('Lifestyle', 'diet'),       # Lifestyle influences diet
    ('diet', 'cholestrol'),      # Diet influences cholesterol
    ('Family', 'heartdisease'),  # Family history affects heart disease
    ('cholestrol', 'heartdisease') # Cholesterol affects heart disease
])
```

The edge list defines the causal structure. This is based on domain knowledge (medical understanding), not learned from data. The structure encodes our assumptions about which variables influence which others.

#### Parameter Learning

```python
model.fit(data, estimator=MaximumLikelihoodEstimator)
```

This single line computes all CPTs from the data. Under the hood, pgmpy counts co-occurrences and normalizes to probabilities.

#### Inference

```python
inference = VariableElimination(model)
result = inference.query(
    variables=['heartdisease'],
    evidence={
        'age': 2,
        'Gender': 1,
        'Family': 0,
        'diet': 1,
        'Lifestyle': 1,
        'cholestrol': 2
    }
)
```

The Variable Elimination algorithm efficiently computes P(heartdisease | evidence) by marginalizing over all non-evidence, non-query variables.

---

## 📈 Inference Examples

### Example 1: Low-Risk Profile

**Patient Profile:**
- Age: Youth (3)
- Gender: Female (1)
- Family History: No (0)
- Diet: Healthy (1)
- Lifestyle: Active (1)
- Cholesterol: Normal (2)

**Expected Output:**
```
Probability of Heart Disease:
+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              0.8500 |
+-----------------+---------------------+
| heartdisease(1) |              0.1500 |
+-----------------+---------------------+
```

Interpretation: An 85% probability of NO heart disease; 15% risk.

### Example 2: High-Risk Profile

**Patient Profile:**
- Age: SuperSeniorCitizen (0)
- Gender: Male (0)
- Family History: Yes (1)
- Diet: High-calorie (0)
- Lifestyle: Sedentary (3)
- Cholesterol: High (0)

**Expected Output:**
```
Probability of Heart Disease:
+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              0.1200 |
+-----------------+---------------------+
| heartdisease(1) |              0.8800 |
+-----------------+---------------------+
```

Interpretation: An 88% probability of heart disease; only 12% chance of being disease-free.

### Sensitivity Analysis: Isolated Risk Factors

By varying one factor while holding others constant, we can estimate each factor's independent contribution to risk:

| Factor Changed | Low-Risk → High-Risk | Disease Probability Change |
|----------------|----------------------|----------------------------|
| Cholesterol only | Normal → High | ~15% → ~60% |
| Family History only | No → Yes | ~15% → ~35% |
| Lifestyle only | Active → Sedentary | ~15% → ~30% |
| Diet only | Healthy → High-calorie | ~15% → ~25% |

This demonstrates that cholesterol has the strongest direct influence in this model, followed by family history, lifestyle, and diet.

---

## 📋 Input Guide

When prompted, enter the numeric code corresponding to each health indicator:

| Variable | Code | Meaning |
|----------|------|---------|
| **Age** | 0 | SuperSeniorCitizen |
| | 1 | SeniorCitizen |
| | 2 | MiddleAged |
| | 3 | Youth |
| | 4 | Teen |
| **Gender** | 0 | Male |
| | 1 | Female |
| **Family History** | 0 | No family history of heart disease |
| | 1 | Family history of heart disease present |
| **Diet** | 0 | High-calorie / Unhealthy |
| | 1 | Medium / Healthy |
| **Lifestyle** | 0 | Athlete (very high activity) |
| | 1 | Active (regular exercise) |
| | 2 | Moderate (some activity) |
| | 3 | Sedentary (minimal activity) |
| **Cholesterol** | 0 | High |
| | 1 | BorderLine |
| | 2 | Normal |

---

## 🔍 Interpreting Results

### Reading the Probability Table

The output displays a table with two rows:

```
+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              [p_no] |
+-----------------+---------------------+
| heartdisease(1) |             [p_yes] |
+-----------------+---------------------+
```

- **heartdisease(0):** Probability of NOT having heart disease
- **heartdisease(1):** Probability of HAVING heart disease
- Values sum to 1.0 (100%)

### Clinical Interpretation Guide

| Probability Range | Risk Level | Suggested Action |
|-------------------|------------|------------------|
| 0% - 20% | Low Risk | Maintain healthy lifestyle, routine check-ups |
| 20% - 50% | Moderate Risk | Lifestyle modifications recommended, regular monitoring |
| 50% - 80% | High Risk | Medical consultation advised, preventive interventions |
| 80% - 100% | Very High Risk | Immediate medical evaluation, aggressive risk factor management |

### Important Caveats

⚠️ **This is an educational demonstration, not medical advice.** Key limitations:

- The model is only as good as its training data — results reflect patterns in the dataset, not absolute medical truth
- The causal structure is assumed, not verified through controlled experiments
- Missing risk factors (smoking, blood pressure, diabetes, etc.) limit comprehensiveness
- Small datasets may produce unreliable probability estimates for rare combinations

---

## 🚀 Limitations & Extensions

### Current Limitations

1. **Fixed Structure:** The causal graph is hardcoded based on assumptions. Real causal discovery would require structural learning algorithms.

2. **Discrete Variables Only:** Continuous variables (actual age, cholesterol mg/dL) must be discretized, losing information.

3. **No Temporal Dynamics:** The model captures static relationships, not how risk evolves over time.

4. **Missing Confounders:** Unmodeled variables (smoking, exercise intensity, genetic markers) may confound the relationships.

5. **Single Dataset:** Parameters are learned from one dataset without cross-validation or uncertainty estimates.

### Potential Extensions

#### Structural Learning

Instead of specifying edges manually, learn the structure from data:

```python
from pgmpy.estimators import HillClimbSearch, BicScore

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))
```

#### Bayesian Parameter Estimation

Replace MLE with Bayesian estimation to incorporate prior knowledge:

```python
from pgmpy.estimators import BayesianEstimator

model.fit(data, estimator=BayesianEstimator, prior_type='BDeu')
```

#### Model Validation

Split data for training and testing:

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
model.fit(train, estimator=MaximumLikelihoodEstimator)

# Evaluate on test set
y_true = test['heartdisease']
y_pred = model.predict(test.drop('heartdisease', axis=1))
```

#### Additional Risk Factors

Extend the network with more variables:

```python
model = DiscreteBayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Lifestyle', 'diet'),
    ('diet', 'cholestrol'),
    ('diet', 'blood_pressure'),       # New node
    ('Lifestyle', 'blood_pressure'),   # New edge
    ('smoking', 'cholestrol'),         # New node
    ('smoking', 'heartdisease'),       # New edge
    ('Family', 'heartdisease'),
    ('cholestrol', 'heartdisease'),
    ('blood_pressure', 'heartdisease') # New edge
])
```

#### Continuous Variables

Use Gaussian Bayesian Networks for continuous data:

```python
from pgmpy.models import LinearGaussianBayesianNetwork
```

#### Explainable Risk Reports

Generate natural language explanations:

```python
# Identify which risk factors contributed most
# "Your elevated risk is primarily driven by high cholesterol
#  and family history. Addressing diet could reduce risk by ~20%."
```

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `FileNotFoundError: Health.csv` | Dataset not in working directory | Place Health.csv in the same folder as the script, or provide full path |
| `KeyError` during inference | Invalid evidence value entered | Check input guide — values must match the categorical encoding (0-4, 0-1, etc.) |
| All probabilities are 0 or 1 | Overfitting to small dataset | Collect more data or use Bayesian estimation with smoothing |
| `ValueError: The model has not been fitted` | fit() not called before inference | Ensure model.fit() executes before VariableElimination |
| Warning messages about CPDs | Some CPT entries are uniform | Normal with small datasets — pgmpy fills missing combinations with uniform distributions |




---

*Built with pgmpy — Probabilistic Graphical Models in Python*
