
# ID3 Decision Tree Classifier

### Recursive Information Gain-Based Decision Tree from Scratch

This project implements the **ID3 (Iterative Dichotomiser 3) algorithm** — a classic decision tree learning algorithm that builds classification trees by recursively selecting attributes that maximize information gain, creating a human-interpretable model for decision-making.


---

## 📑 Table of Contents

- [Overview](#-overview)
- [Theoretical Foundations](#-theoretical-foundations)
- [Quick Start](#-quick-start)
- [Algorithm Mechanics](#-algorithm-mechanics)
- [Implementation Details](#-implementation-details)
- [Step-by-Step Execution Trace](#-step-by-step-execution-trace)
- [Mathematical Deep Dive](#-mathematical-deep-dive)
- [Interpreting the Decision Tree](#-interpreting-the-decision-tree)
- [Comparison with Other Algorithms](#-comparison-with-other-algorithms)
- [Limitations & Extensions](#-limitations--extensions)
- [Troubleshooting](#-troubleshooting)

---

## 📝 Overview

### The Classification Problem

Given a set of labeled training examples — each described by a set of attributes and a target class — build a model that can accurately classify new, unseen instances. The model should not only be accurate but also interpretable, allowing humans to understand and validate the reasoning behind each classification.

### The ID3 Approach

ID3 builds a **decision tree** — a tree-structured classifier where:

- **Internal nodes** test the value of a particular attribute
- **Branches** correspond to possible attribute values
- **Leaf nodes** specify the predicted class

The tree is constructed top-down, recursively, by always selecting the "best" attribute to split on at each node — where "best" means the attribute that provides the most information gain about the target classification.

### Why Decision Trees?

| Advantage | Description |
|-----------|-------------|
| **Interpretability** | Trees can be visualized and understood by non-experts |
| **No Data Preprocessing** | No normalization, scaling, or one-hot encoding required |
| **Feature Selection** | The tree structure reveals which attributes matter most |
| **Handles Mixed Types** | Works with both categorical and discretized continuous data |
| **White-Box Model** | Every decision path can be traced and explained |

---

## 🧠 Theoretical Foundations

### Information Theory Background

The ID3 algorithm is grounded in Claude Shannon's information theory. The key insight: the best attribute for splitting is the one that reduces uncertainty about the target classification the most.

### Entropy: Measuring Uncertainty

**Entropy** measures the impurity or randomness in a dataset:

```
H(S) = -Σ p(c) · log₂(p(c))
```

Where:
- S is the set of examples
- p(c) is the proportion of examples belonging to class c
- Sum is over all classes

**Properties:**
- H = 0 when all examples belong to the same class (perfect purity, no uncertainty)
- H = 1 for a balanced binary classification (maximum uncertainty)
- H > 1 for multi-class problems with uniform distribution

**Examples:**
- 100% Yes, 0% No → H = 0 (completely pure)
- 50% Yes, 50% No → H = 1.0 (maximum uncertainty for binary)
- 33% each of three classes → H = 1.585 (maximum uncertainty for 3 classes)
- 90% Yes, 10% No → H = 0.469 (low uncertainty)

### Information Gain: Measuring Reduction in Uncertainty

**Information Gain** quantifies how much an attribute reduces entropy:

```
IG(S, A) = H(S) - Σ (|S_v| / |S|) · H(S_v)
```

Where:
- H(S) is the entropy of the original set
- S_v is the subset of S where attribute A has value v
- The weighted sum is the expected entropy after splitting on A

**Interpretation:**
- IG = 0: The attribute provides no information about the class
- IG = 1 (max for binary): The attribute perfectly separates the classes
- Higher IG indicates a more useful attribute for classification

### The ID3 Algorithm Philosophy

ID3 follows a **greedy, recursive, divide-and-conquer** strategy:

1. Select the attribute with highest information gain
2. Split the dataset on that attribute
3. For each resulting subset, recursively build a subtree
4. Stop when all examples in a subset belong to the same class (or no attributes remain)

The algorithm prefers:
- Attributes with high information gain (near the root)
- Shorter trees (implicitly, through pure subsets terminating early)
- The attribute that creates the most homogeneous groups

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- pandas library
- NumPy library

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/id3-decision-tree.git
cd id3-decision-tree

# Install dependencies
pip install pandas numpy

# Ensure playtennis.csv is in the project directory

# Run the decision tree builder
python decision_tree.py
```

### Expected Workflow

1. The script loads the Play Tennis dataset
2. Displays the complete training data
3. Builds a decision tree using the ID3 algorithm
4. Displays the resulting tree structure
5. Classifies a test instance using the built tree
6. Outputs the final prediction

---

## 🔧 Algorithm Mechanics

### Core Functions

The implementation consists of six key functions that work together:

```
┌─────────────────────┐
│   find_entropy()    │  ← Calculates H(S) for any dataset
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│find_entropy_attr()  │  ← Calculates weighted entropy after split
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│   find_winner()     │  ← Selects attribute with highest IG
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  get_subtable()     │  ← Creates subset for a specific attribute value
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│   buildTree()       │  ← Recursively constructs the decision tree
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│   classify()        │  ← Traverses tree to classify new instances
└─────────────────────┘
```

### 1. Entropy Calculation

```python
def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    
    return entropy
```

**What it does:** Computes H(S) — the overall impurity of the dataset with respect to the target class.

**Mathematical operation:**
```
H(S) = -Σ(p_i · log₂(p_i))
```

### 2. Attribute-Specific Entropy

```python
def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    entropy2 = 0
    
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[(df[attribute] == variable) & 
                          (df[Class] == target_variable)])
            den = len(df[df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        
        fraction2 = den / len(df)
        entropy2 += fraction2 * entropy
    
    return abs(entropy2)
```

**What it does:** Computes the weighted average entropy after splitting on a specific attribute.

**Mathematical operation:**
```
H(S|A) = Σ(|S_v|/|S|) · H(S_v)
```

**Epsilon handling:** Small epsilon values prevent log(0) and division by zero errors for empty subsets.

### 3. Attribute Selection (Winner)

```python
def find_winner(df):
    IG = []
    for key in df.keys()[:-1]:
        ig = find_entropy(df) - find_entropy_attribute(df, key)
        IG.append(ig)
    return df.keys()[:-1][np.argmax(IG)]
```

**What it does:** Calculates information gain for each attribute and returns the one with maximum IG.

**Mathematical operation:**
```
Winner = argmax_A [H(S) - H(S|A)]
      = argmax_A IG(S, A)
```

### 4. Subtable Creation

```python
def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)
```

**What it does:** Filters the dataset to rows where the specified node has the given value, creating the subset for the next recursive step.

### 5. Recursive Tree Construction

```python
def buildTree(df, tree=None):
    Class = df.keys()[-1]
    node = find_winner(df)
    attValue = np.unique(df[node])
    
    if tree is None:
        tree = {}
        tree[node] = {}
    
    for value in attValue:
        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable[Class], return_counts=True)
        
        if len(counts) == 1:
            tree[node][value] = clValue[0]    # Pure subset → leaf
        else:
            tree[node][value] = buildTree(subtable)  # Impure → recurse
    
    return tree
```

**Base case:** When all examples in a subset belong to the same class (counts length = 1), create a leaf node.

**Recursive case:** When multiple classes remain, recursively build a subtree using only the remaining examples.

### 6. Classification

```python
def classify(test, tree, default=None):
    attribute = next(iter(tree))
    
    if test[attribute] in tree[attribute]:
        result = tree[attribute][test[attribute]]
        
        if isinstance(result, dict):
            return classify(test, result)  # Continue traversing
        else:
            return result  # Reached leaf
    
    return default  # Unknown attribute value
```

**What it does:** Traverses the tree from root to leaf based on the test instance's attribute values. Prints the decision path for transparency.

---

## 📊 Step-by-Step Execution Trace

Using the classic Play Tennis dataset:

### Dataset

| Day | Outlook | Temperature | Humidity | Wind | PlayTennis |
|-----|---------|-------------|----------|------|------------|
| D1 | Sunny | Hot | High | Weak | No |
| D2 | Sunny | Hot | High | Strong | No |
| D3 | Overcast | Hot | High | Weak | Yes |
| D4 | Rain | Mild | High | Weak | Yes |
| D5 | Rain | Cool | Normal | Weak | Yes |
| D6 | Rain | Cool | Normal | Strong | No |
| D7 | Overcast | Cool | Normal | Strong | Yes |
| D8 | Sunny | Mild | High | Weak | No |
| D9 | Sunny | Cool | Normal | Weak | Yes |
| D10 | Rain | Mild | Normal | Weak | Yes |
| D11 | Sunny | Mild | Normal | Strong | Yes |
| D12 | Overcast | Mild | High | Strong | Yes |
| D13 | Overcast | Hot | Normal | Weak | Yes |
| D14 | Rain | Mild | High | Strong | No |

### Round 1: Root Node Selection

**Step 1: Calculate overall entropy H(S)**
```
Yes: 9/14, No: 5/14
H(S) = -(9/14)·log₂(9/14) - (5/14)·log₂(5/14)
     = 0.940
```

**Step 2: Calculate information gain for each attribute**

*Outlook:*
```
Sunny:     [2 Yes, 3 No] → H = 0.971
Overcast:  [4 Yes, 0 No] → H = 0.000
Rain:      [3 Yes, 2 No] → H = 0.971

Weighted Avg = (5/14)·0.971 + (4/14)·0.000 + (5/14)·0.971
             = 0.694

IG(Outlook) = 0.940 - 0.694 = 0.246
```

*Temperature:*
```
Hot:   [2 Yes, 2 No] → H = 1.000
Mild:  [4 Yes, 2 No] → H = 0.918
Cool:  [3 Yes, 1 No] → H = 0.811

Weighted Avg = (4/14)·1.000 + (6/14)·0.918 + (4/14)·0.811
             = 0.911

IG(Temperature) = 0.940 - 0.911 = 0.029
```

*Humidity:*
```
High:    [3 Yes, 4 No] → H = 0.985
Normal:  [6 Yes, 1 No] → H = 0.592

Weighted Avg = (7/14)·0.985 + (7/14)·0.592
             = 0.788

IG(Humidity) = 0.940 - 0.788 = 0.152
```

*Wind:*
```
Weak:    [6 Yes, 2 No] → H = 0.811
Strong:  [3 Yes, 3 No] → H = 1.000

Weighted Avg = (8/14)·0.811 + (6/14)·1.000
             = 0.892

IG(Wind) = 0.940 - 0.892 = 0.048
```

**Step 3: Select winner**
```
IG(Outlook) = 0.246  ← HIGHEST
IG(Humidity) = 0.152
IG(Wind) = 0.048
IG(Temperature) = 0.029

Winner: Outlook
```

**Result:** Outlook becomes the root node.

### Round 2: Sunny Branch

The Sunny subset has mixed classes (2 Yes, 3 No), so we recurse.

Remaining attributes: Temperature, Humidity, Wind

```
H(Sunny) = -(2/5)·log₂(2/5) - (3/5)·log₂(3/5) = 0.971

IG(Humidity|Sunny) = 0.971 - 0.000 = 0.971  ← HIGHEST (perfect split!)
IG(Wind|Sunny) = 0.971 - 0.951 = 0.020
IG(Temperature|Sunny) = 0.971 - 0.400 = 0.571

Winner: Humidity
```

Humidity provides a perfect split for the Sunny branch:
- High → No (pure)
- Normal → Yes (pure)

### Round 3: Rain Branch

The Rain subset has mixed classes (3 Yes, 2 No).

```
H(Rain) = -(3/5)·log₂(3/5) - (2/5)·log₂(2/5) = 0.971

IG(Wind|Rain) = 0.971 - 0.000 = 0.971  ← HIGHEST (perfect split!)
IG(Humidity|Rain) = 0.971 - 0.951 = 0.020
IG(Temperature|Rain) = 0.971 - 0.551 = 0.420

Winner: Wind
```

Wind provides a perfect split for the Rain branch:
- Weak → Yes (pure)
- Strong → No (pure)

### Round 4: Overcast Branch

The Overcast subset is already pure (4 Yes, 0 No) → Leaf node: Yes

### Final Tree Structure

```
Outlook
├── Sunny
│   └── Humidity
│       ├── High → No
│       └── Normal → Yes
├── Overcast → Yes
└── Rain
    └── Wind
        ├── Weak → Yes
        └── Strong → No
```

### Test Classification

**Test Instance:** {Outlook: Sunny, Temperature: Hot, Humidity: High, Wind: Weak}

```
1. Check Outlook → Sunny → go to Humidity subtree
2. Check Humidity → High → go to leaf
3. Result: No
```

**Interpretation:** On a sunny day with high humidity, tennis is not played — matching the pattern from D1 and D2.

---

## 📐 Mathematical Deep Dive

### Why log₂?

The choice of logarithm base determines the unit of entropy:
- **log₂:** Entropy measured in bits (binary digits)
- **log_e (ln):** Entropy measured in nats (natural units)
- **log_10:** Entropy measured in dits (decimal digits)

Base 2 is conventional in machine learning because it relates to binary encoding: entropy of 1 bit means one yes/no question is needed on average to determine the class.

### The Greedy Nature of ID3

ID3 makes locally optimal choices at each node without backtracking:

```
ID3 does NOT guarantee the globally optimal (smallest) decision tree.
```

This is because:
1. The best root attribute might not lead to the smallest overall tree
2. A slightly worse first split might enable much better subsequent splits
3. Finding the optimal tree is NP-complete; ID3 provides an efficient heuristic

In practice, the greedy approach works well for most datasets.

### Why Information Gain Prefers Many-Valued Attributes

Information gain has an inherent bias toward attributes with many distinct values. Consider an attribute like "Day" with 14 unique values — it would achieve near-perfect information gain on training data but have zero generalization ability.

This is addressed in the **C4.5 algorithm** (ID3's successor) by using **Gain Ratio**:

```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

SplitInfo(S, A) = -Σ(|S_v|/|S|) · log₂(|S_v|/|S|)
```

SplitInfo penalizes attributes with many values by measuring the entropy of the attribute itself.

### The Epsilon Trick

```python
eps = np.finfo(float).eps
fraction = num / (den + eps)
entropy += -fraction * log(fraction + eps)
```

Epsilon (approximately 2.22 × 10⁻¹⁶) prevents:
- **Division by zero:** When a subset has no examples for a particular class
- **log(0):** When a fraction is zero (log(0) is undefined, approaches -∞)
- **Numerical stability:** Keeps calculations within floating-point precision limits

---

## 🌲 Interpreting the Decision Tree

### Reading the Tree Structure

The output tree uses nested Python dictionaries:

```python
{
    'Outlook': {
        'Sunny': {
            'Humidity': {
                'High': 'No',
                'Normal': 'Yes'
            }
        },
        'Overcast': 'Yes',
        'Rain': {
            'Wind': {
                'Weak': 'Yes',
                'Strong': 'No'
            }
        }
    }
}
```

### Converting to Rules

Each path from root to leaf represents a classification rule:

```
R1: IF Outlook=Sunny AND Humidity=High         THEN PlayTennis=No
R2: IF Outlook=Sunny AND Humidity=Normal       THEN PlayTennis=Yes
R3: IF Outlook=Overcast                        THEN PlayTennis=Yes
R4: IF Outlook=Rain AND Wind=Weak              THEN PlayTennis=Yes
R5: IF Outlook=Rain AND Wind=Strong            THEN PlayTennis=No
```

These rules are:
- **Mutually exclusive:** Each instance matches exactly one rule
- **Exhaustive:** All possible attribute combinations are covered
- **Human-readable:** Directly interpretable by domain experts

### Which Attributes Matter Most?

The tree structure reveals feature importance:

1. **Outlook** (Root): Most important predictor
2. **Humidity** (Sunny branch): Important when Outlook is Sunny
3. **Wind** (Rain branch): Important when Outlook is Rain
4. **Temperature**: Not used — provides minimal information gain after Outlook split

---

## 📊 Comparison with Other Algorithms

| Algorithm | Approach | Output | Handles Continuous | Handles Missing | Pruning |
|-----------|----------|--------|-------------------|-----------------|---------|
| **ID3** | Information Gain | Decision tree | No (requires discretization) | No | No |
| **C4.5** | Gain Ratio | Decision tree | Yes (binary splits) | Yes | Yes (post-pruning) |
| **CART** | Gini Impurity | Binary decision tree | Yes | Yes (surrogate splits) | Yes (cost-complexity) |
| **Random Forest** | Ensemble of trees | Multiple trees | Yes | Yes (imputation) | Per-tree |
| **XGBoost** | Gradient boosting | Ensemble | Yes | Yes (sparse-aware) | Regularization |

### ID3 vs Find-S vs Candidate Elimination

| Aspect | Find-S | Candidate Elimination | ID3 |
|--------|--------|----------------------|-----|
| **Output** | Single specific hypothesis | Version space boundaries | Decision tree |
| **Bias** | Maximally specific | Version space representation | Preference for shorter trees + high IG |
| **Noise Handling** | None | None | Some (statistical) |
| **Incremental** | Yes | Yes | No (needs all data) |
| **Interpretability** | Moderate | Moderate | High |

---

## 🚀 Limitations & Extensions

### Current Limitations

1. **Categorical Attributes Only:** Continuous values must be discretized before use.

2. **No Pruning:** The tree may overfit, capturing noise rather than signal. Every leaf is pure on training data.

3. **Greedy Construction:** No backtracking means the algorithm can get stuck in locally optimal but globally suboptimal structures.

4. **IG Bias:** Information gain favors attributes with many distinct values, which may not be the most predictive.

5. **No Missing Value Handling:** Incomplete instances require preprocessing (imputation or removal).

6. **Axis-Parallel Splits Only:** Decision boundaries are perpendicular to attribute axes, limiting expressiveness for diagonal concepts.

### Potential Extensions

#### Gain Ratio (C4.5 Style)

Replace information gain with gain ratio to penalize many-valued attributes:

```python
def gain_ratio(df, attribute):
    ig = information_gain(df, attribute)
    split_info = split_information(df, attribute)
    
    if split_info == 0:
        return 0
    
    return ig / split_info
```

#### Reduced Error Pruning

Post-process the tree to remove branches that don't improve validation accuracy:

```python
def prune(tree, validation_data):
    for node in tree:
        # Try replacing subtree with majority class leaf
        # If validation accuracy improves or stays same, prune
        pass
```

#### Handling Continuous Attributes

Binary split on threshold values:

```python
def best_split_continuous(df, attribute):
    values = sorted(df[attribute].unique())
    best_ig = 0
    best_threshold = None
    
    for i in range(len(values) - 1):
        threshold = (values[i] + values[i+1]) / 2
        ig = information_gain_binary(df, attribute, threshold)
        
        if ig > best_ig:
            best_ig = ig
            best_threshold = threshold
    
    return best_threshold, best_ig
```

#### Random Forest Ensemble

Build multiple trees on bootstrapped samples with random feature subsets:

```python
def random_forest(df, n_trees=100, max_features='sqrt'):
    forest = []
    
    for _ in range(n_trees):
        sample = df.sample(n=len(df), replace=True)
        features = random.sample(df.columns[:-1], k=int(np.sqrt(len(df.columns)-1)))
        tree = buildTree(sample[features + [df.columns[-1]]])
        forest.append(tree)
    
    return forest
```

#### Visualization with Graphviz

Generate visual tree diagrams:

```python
from graphviz import Digraph

def visualize_tree(tree, dot=None, parent=None, edge_label=None):
    if dot is None:
        dot = Digraph()
    
    for node, branches in tree.items():
        # Create node
        # Add edges to children
        pass
    
    return dot
```

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `FileNotFoundError: playtennis.csv` | Dataset not in working directory | Place CSV in same folder or update file path |
| `KeyError` during classification | Test attribute value not seen in training | Add default parameter or handle unknown values |
| Maximum recursion depth exceeded | Circular or extremely deep tree | Check for duplicate attributes or data issues |
| All IGs are zero | All attributes already used or target is pure | Expected behavior — check base case logic |
| Tree has only one node | One attribute dominates completely | Verify dataset has sufficient variability |
| `ValueError: max() arg is an empty sequence` | Empty dataframe passed to find_winner | Check data loading and preprocessing |

