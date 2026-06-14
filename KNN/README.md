

# K-Nearest Neighbors (KNN) Classifier

### Instance-Based Learning on the Iris Flower Dataset

This project implements a **K-Nearest Neighbors (KNN) classifier** using scikit-learn to classify iris flower species based on sepal and petal measurements. KNN is a foundational machine learning algorithm that classifies new instances by finding the most similar training examples and taking a majority vote.


---

## 📑 Table of Contents

- [Overview](#-overview)
- [Theoretical Foundations](#-theoretical-foundations)
- [Quick Start](#-quick-start)
- [Algorithm Mechanics](#-algorithm-mechanics)
- [Implementation Details](#-implementation-details)
- [The Iris Dataset](#-the-iris-dataset)
- [Mathematical Deep Dive](#-mathematical-deep-dive)
- [Interpreting Results](#-interpreting-results)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Comparison with Other Classifiers](#-comparison-with-other-classifiers)
- [Limitations & Extensions](#-limitations--extensions)
- [Troubleshooting](#-troubleshooting)


---

## 📝 Overview

### The Classification Problem

Given measurements of iris flowers (sepal length, sepal width, petal length, petal width), classify each flower into one of three species: **Setosa**, **Versicolor**, or **Virginica**. This is a classic multi-class classification problem with four numerical features and three balanced classes.

### The KNN Approach

K-Nearest Neighbors is fundamentally different from algorithms like decision trees or neural networks. It belongs to the family of **instance-based (lazy) learners**:

- **No training phase:** KNN doesn't build an explicit model. It simply stores the training data.
- **Classification at query time:** When asked to classify a new instance, KNN searches the stored training data for the K most similar (nearest) examples.
- **Majority voting:** The predicted class is the most common class among those K neighbors.

This makes KNN intuitive and easy to understand — classification is based directly on similarity to known examples, much like how humans often reason by analogy.

### What This Implementation Demonstrates

- Loading and preprocessing the classic Iris dataset
- Label encoding for multi-class classification
- Train-test splitting for proper model evaluation
- KNN classification with configurable hyperparameters (K=5, Euclidean distance)
- Model evaluation using accuracy metrics
- Detailed output showing correct and incorrect predictions

---

## 🧠 Theoretical Foundations

### The Instance-Based Learning Paradigm

Unlike **eager learners** (decision trees, neural networks, SVMs) that build a model during training and discard the training data, instance-based learners:

1. **Store** all training examples in memory
2. **Defer** all computation until classification time
3. **Compare** new instances directly to stored examples
4. **Classify** based on local similarity

This approach has profound implications:

| Property | Eager Learners (Decision Tree, NN) | Lazy Learners (KNN) |
|----------|-------------------------------------|---------------------|
| **Training Time** | Can be slow (model building) | Instant (just store data) |
| **Prediction Time** | Fast (model already built) | Can be slow (compare to all data) |
| **Memory Usage** | Model parameters (compact) | All training data (potentially large) |
| **Adaptability** | Retrain to add new data | Naturally handles new data |
| **Interpretability** | Varies by model | High (show nearest neighbors) |

### The KNN Assumption

KNN makes one fundamental assumption:

> **Instances that are close in the feature space are likely to belong to the same class.**

This is the **smoothness assumption** — the classification function varies smoothly across the feature space, so local neighborhoods are predominantly homogeneous.

When this assumption holds (and it often does for well-separated classes), KNN performs remarkably well despite its simplicity.

### The Curse of Dimensionality

KNN's effectiveness degrades in high-dimensional spaces due to the **curse of dimensionality**:

- In high dimensions, all points tend to be approximately equidistant from each other
- The concept of "nearest" neighbors becomes meaningless
- The volume of the space grows exponentially, requiring exponentially more data to maintain density

**Practical guideline:** KNN works best with fewer than 10-20 features. For higher-dimensional data, dimensionality reduction (PCA, feature selection) should precede KNN.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- scikit-learn library
- NumPy library
- CSV module (included in Python standard library)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/knn-iris-classifier.git
cd knn-iris-classifier

# Install dependencies
pip install scikit-learn numpy

# Ensure iris.csv is in the project directory

# Run the classifier
python knn_iris.py
```

### Expected Output

The script will display:
1. Model accuracy (percentage of correct predictions)
2. List of correct predictions with actual and predicted classes
3. List of incorrect predictions (if any) with actual and predicted classes

---

## 🔧 Algorithm Mechanics

### The KNN Classification Algorithm

```
Algorithm: K-Nearest Neighbors Classification

Input:  Training data D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
        Query instance x_q
        Number of neighbors K
        Distance metric d(·, ·)

Output: Predicted class ŷ for x_q

1. For each training instance (xᵢ, yᵢ) in D:
      Compute distance d(x_q, xᵢ)

2. Sort training instances by distance (ascending)

3. Select the K nearest instances (smallest distances)

4. Count class frequencies among K neighbors:
      votes[c] = count of neighbors with class c

5. Return the class with the most votes:
      ŷ = argmax_c votes[c]
```

### Distance Metrics

The choice of distance metric fundamentally affects which instances are considered "nearest":

#### Euclidean Distance (p=2)

The straight-line distance between two points:

```
d(x, y) = √(Σ(xᵢ - yᵢ)²)
```

**Characteristics:**
- Most commonly used distance metric
- Assumes all features are equally important and on the same scale
- Sensitive to feature scaling — requires normalization
- Corresponds to L2 norm

#### Manhattan Distance (p=1)

The sum of absolute differences (city block distance):

```
d(x, y) = Σ|xᵢ - yᵢ|
```

**Characteristics:**
- Less sensitive to outliers than Euclidean
- Appropriate for grid-like feature spaces
- Corresponds to L1 norm

#### Minkowski Distance

A generalization of Euclidean and Manhattan:

```
d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```

- p=1: Manhattan distance
- p=2: Euclidean distance
- p→∞: Chebyshev distance (maximum difference in any dimension)

### Choosing K

The value of K — the number of neighbors to consult — critically affects model behavior:

| K Value | Decision Boundary | Bias | Variance | Noise Sensitivity |
|---------|-------------------|------|----------|-------------------|
| **K=1** | Highly complex, jagged | Very low | Very high | Extremely sensitive |
| **K=3** | Moderately complex | Low | Moderate | Somewhat sensitive |
| **K=5** | Smooth (used here) | Moderate | Moderate | Moderately robust |
| **K=10** | Very smooth | Higher | Low | Robust |
| **K=N** | Always predicts majority class | Maximum | Zero | Maximum robustness |

**Guidelines for choosing K:**
- K should be odd for binary classification (avoids ties)
- K = √n (where n is training set size) is a common heuristic
- Cross-validation should determine the optimal K
- Larger K reduces overfitting but may underfit

---

## 💻 Implementation Details

### Code Structure

```
knn_iris.py
├── Import libraries (csv, numpy, sklearn)
├── Define class mapping (label encoding)
│   ├── setosa → 0
│   ├── versicolor → 1
│   └── virginica → 2
├── Define reverse mapping (for display)
├── Load dataset from iris.csv
│   ├── Skip header row
│   ├── Extract features (first 4 columns)
│   └── Extract and encode labels (last column)
├── Convert to numpy arrays (float features, int labels)
├── Split data: 75% training, 25% testing
├── Create KNN classifier (K=5, Euclidean distance)
├── Train the classifier (fit)
├── Make predictions on test set
├── Calculate and display accuracy
└── Display correct and incorrect predictions
```

### Key Implementation Details

#### 1. Label Encoding

```python
class_dict = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
reverse_class_dict = {v: k for k, v in class_dict.items()}
```

Machine learning algorithms require numerical labels. The mapping converts categorical species names to integers. The reverse mapping enables human-readable output.

#### 2. Feature-Label Separation

```python
for line in dataset:
    X.append(line[:-1])           # First 4 columns: features
    y.append(class_dict[line[-1]]) # Last column: label (encoded)
```

The Iris dataset has four feature columns (sepal length, sepal width, petal length, petal width) and one target column (species).

#### 3. Type Conversion

```python
X = np.array(X).astype(float)  # Features must be numeric
y = np.array(y).astype(int)    # Labels must be integer-encoded
```

KNN requires features to be numeric for distance computation. Labels are integer-encoded for scikit-learn compatibility.

#### 4. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)
```

**Default split:** 75% training (112 samples), 25% testing (38 samples)

**random_state=0** ensures reproducible splits across runs. Without this, each run would produce different train/test divisions and potentially different accuracy results.

#### 5. KNN Configuration

```python
classifier = KNeighborsClassifier(
    n_neighbors=5,       # K = 5 neighbors
    metric='minkowski',  # Minkowski distance family
    p=2                  # p=2 → Euclidean distance
)
```

This creates a KNN classifier that:
- Consults the 5 nearest neighbors
- Uses standard Euclidean distance
- Weighs all neighbors equally (uniform weights by default)

#### 6. Prediction and Evaluation

```python
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

`predict()` classifies each test instance by majority vote among its K nearest training neighbors. `accuracy_score()` computes the fraction of correct predictions.

---

## 🌸 The Iris Dataset

### Historical Significance

The Iris dataset is one of the most famous datasets in machine learning, introduced by British statistician and biologist **Ronald Fisher** in his 1936 paper "The Use of Multiple Measurements in Taxonomic Problems." It has become the "Hello World" of classification algorithms.

### Dataset Structure

| Attribute | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| Sepal Length | Length of the sepal | cm | 4.3 - 7.9 |
| Sepal Width | Width of the sepal | cm | 2.0 - 4.4 |
| Petal Length | Length of the petal | cm | 1.0 - 6.9 |
| Petal Width | Width of the petal | cm | 0.1 - 2.5 |
| Species | Iris species | categorical | setosa, versicolor, virginica |

### Species Characteristics

| Species | Petal Size | Sepal Size | Distinguishability |
|---------|------------|------------|--------------------|
| **Setosa** | Small petals | Smaller sepals | Linearly separable from others |
| **Versicolor** | Medium petals | Medium sepals | Overlaps somewhat with Virginica |
| **Virginica** | Large petals | Larger sepals | Overlaps somewhat with Versicolor |

### Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total samples | 150 |
| Samples per class | 50 each (perfectly balanced) |
| Features | 4 (all continuous) |
| Classes | 3 (multi-class) |
| Missing values | None |
| Class balance | 33.3% each |

### Why Iris Works Well for KNN

1. **Low dimensionality:** Only 4 features — well within KNN's comfort zone
2. **Well-separated classes:** One species is linearly separable from the others
3. **Numerical features:** Distance metrics are directly applicable
4. **Balanced classes:** Majority voting isn't biased toward any class

---

## 📐 Mathematical Deep Dive

### Euclidean Distance Computation

For two iris flowers with feature vectors:

```
x = [5.1, 3.5, 1.4, 0.2]  (Setosa)
y = [7.0, 3.2, 4.7, 1.4]  (Versicolor)

d(x, y) = √[(5.1-7.0)² + (3.5-3.2)² + (1.4-4.7)² + (0.2-1.4)²]
        = √[(-1.9)² + (0.3)² + (-3.3)² + (-1.2)²]
        = √[3.61 + 0.09 + 10.89 + 1.44]
        = √[16.03]
        = 4.00
```

This distance quantifies overall dissimilarity between the two flowers.

### Why Feature Scaling Matters

Consider two features with different scales:

```
Feature A (sepal length): range [4.3, 7.9], variance ≈ 0.69
Feature B (petal length): range [1.0, 6.9], variance ≈ 3.12
```

Without scaling, petal length dominates the distance calculation because it has larger absolute differences. To prevent this:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardization transforms each feature to mean=0, variance=1, ensuring all features contribute equally to distance.

### Weighted KNN

Standard KNN gives equal weight to all K neighbors. Weighted KNN gives closer neighbors more influence:

```
ŷ = argmax_c Σ wᵢ · I(yᵢ = c)
```

Where weights are typically:

```
wᵢ = 1 / d(x_q, xᵢ)           # Inverse distance
wᵢ = 1 / (d(x_q, xᵢ) + ε)     # Smoothed inverse distance
wᵢ = exp(-d(x_q, xᵢ)² / σ²)  # Gaussian kernel
```

Implementation in scikit-learn:

```python
KNeighborsClassifier(n_neighbors=5, weights='distance')
```

---

## 📊 Interpreting Results

### Accuracy Interpretation

A typical KNN (K=5) on Iris achieves **95-98% accuracy**. This means:

- Out of 38 test samples, approximately 36-37 are correctly classified
- Only 1-2 misclassifications typically occur
- Misclassifications almost always involve Versicolor vs. Virginica (the overlapping classes)

### Confusion Matrix Analysis

A confusion matrix reveals where errors occur:

```
                 Predicted
              Set  Ver  Vir
Actual  Set   13    0    0
        Ver    0   15    1
        Vir    0    1    8
```

**Interpretation:**
- All Setosa samples are perfectly classified (linearly separable)
- Versicolor and Virginica have occasional confusion (overlapping distributions)
- No instances are confused with Setosa (well-separated)

### Why Misclassifications Happen

When KNN makes errors, it's typically because:

1. **Class overlap:** The test instance lies in a region where training examples of different classes intermingle
2. **Insufficient K:** Too few neighbors consulted, making the decision sensitive to noise
3. **Feature ambiguity:** The four measurements don't perfectly separate Versicolor from Virginica in all cases

### Detailed Output Format

```
Accuracy:  0.9736842105263158

Correct Predictions:
Input: [5.1 3.5 1.4 0.2] | Actual: setosa | Predicted: setosa
Input: [7.0 3.2 4.7 1.4] | Actual: versicolor | Predicted: versicolor
...

Wrong Predictions:
Input: [6.0 3.0 4.8 1.8] | Actual: virginica | Predicted: versicolor
```

---

## 🎛️ Hyperparameter Tuning

### Finding Optimal K

The choice of K significantly impacts performance:

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")
```

### Distance Metric Selection

Different metrics suit different data distributions:

```python
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
results = {}

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    scores = cross_val_score(knn, X, y, cv=5)
    results[metric] = scores.mean()
```

### Weight Configuration

Compare uniform vs. distance-weighted voting:

```python
# Uniform weights: all K neighbors equal
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Distance weights: closer neighbors have more influence
knn_distance = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

---

## 📊 Comparison with Other Classifiers

### KNN vs. Other Algorithms on Iris

| Algorithm | Typical Accuracy | Training Time | Prediction Time | Interpretability |
|-----------|-----------------|---------------|-----------------|------------------|
| **KNN (K=5)** | 95-97% | Instant | Slow for large data | High (show neighbors) |
| **Decision Tree** | 94-96% | Fast | Very fast | High (visual tree) |
| **Logistic Regression** | 95-97% | Fast | Very fast | Moderate (coefficients) |
| **SVM** | 96-98% | Moderate | Fast | Low |
| **Random Forest** | 95-97% | Moderate | Fast | Low-moderate |
| **Naive Bayes** | 95-96% | Very fast | Very fast | Moderate |

### When to Choose KNN

**KNN is ideal when:**
- The decision boundary is irregular or complex
- Training data is not too large (prediction time scales with dataset size)
- You need to show examples to justify predictions
- The number of features is small (< 20)
- You want a baseline model before trying more complex approaches

**KNN is not ideal when:**
- Training data is very large (slow predictions)
- Features are high-dimensional (curse of dimensionality)
- Real-time predictions are needed on resource-constrained systems
- Feature importance ranking is required
- The data has many irrelevant features

---

## 🚀 Limitations & Extensions

### Current Limitations

1. **Computational Cost at Prediction Time:** KNN must compare the query instance to every training example. For large datasets (millions of examples), this is prohibitively slow.

2. **Memory Requirements:** All training data must be stored in memory. A dataset with 1 million 100-dimensional vectors requires significant RAM.

3. **Curse of Dimensionality:** In high-dimensional spaces, distance metrics become less meaningful as all points become approximately equidistant.

4. **Feature Scaling Sensitivity:** Features with larger ranges dominate the distance calculation unless properly normalized.

5. **Imbalanced Data:** When classes have very different numbers of examples, majority voting is biased toward the majority class.

6. **No Feature Importance:** KNN provides no inherent mechanism for determining which features are most predictive.

### Potential Extensions

#### Feature Scaling

Add standardization for improved performance:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

#### Cross-Validation

Replace single train-test split with cross-validation:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

knn = KNeighborsClassifier(n_neighbors=5)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn, X, y, cv=cv)

print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

#### k-d Tree for Efficient Search

Use spatial data structures for faster neighbor search:

```python
# 'auto' automatically chooses best algorithm
KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')

# Options: 'ball_tree', 'kd_tree', 'brute'
```

#### Confusion Matrix Visualization

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=['setosa', 'versicolor', 'virginica'])
disp.plot()
plt.show()
```

#### Decision Boundary Visualization

Plot the decision boundaries for the first two features:

```python
from matplotlib.colors import ListedColormap

# Use only first two features for 2D visualization
X_2d = X[:, :2]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_2d, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train2, y_train2)

# Create mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict on mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['red', 'green', 'blue']))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=ListedColormap(['red', 'green', 'blue']))
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
```

#### Comparison with Other Classifiers

Benchmark multiple algorithms on the same data:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'SVM': SVC(kernel='rbf', random_state=0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0)
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
```

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `FileNotFoundError: iris.csv` | Dataset not in working directory | Place iris.csv in the same folder or update file path |
| `ValueError: could not convert string to float` | Header row not skipped | Verify `dataset = dataset[1:]` removes the header |
| `KeyError: 'Iris-setosa'` or similar | Label format mismatch | Check CSV for exact species names; update class_dict |
| Low accuracy (<90%) | Features not scaled | Add StandardScaler preprocessing |
| All predictions are one class | Severe class imbalance | Check class distribution; use stratified splitting |
| `ValueError: n_neighbors > n_samples` | K larger than training set | Reduce K or increase training data |

