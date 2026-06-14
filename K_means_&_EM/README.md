

# Clustering Algorithms Comparison: K-Means vs EM (Gaussian Mixture Model)

### Unsupervised Learning on the Iris Flower Dataset

This project implements and compares two fundamental clustering algorithms — **K-Means** and **Expectation-Maximization (EM) with Gaussian Mixture Models (GMM)** — on the classic Iris dataset. By applying unsupervised learning to labeled data, the project evaluates how well each algorithm discovers natural groupings without prior knowledge of class labels.


---

## 📑 Table of Contents

- [Overview](#-overview)
- [Theoretical Foundations](#-theoretical-foundations)
- [Quick Start](#-quick-start)
- [Algorithm Deep Dives](#-algorithm-deep-dives)
- [Implementation Details](#-implementation-details)
- [Visualization Guide](#-visualization-guide)
- [Mathematical Comparison](#-mathematical-comparison)
- [Interpreting Results](#-interpreting-results)
- [When to Use Which Algorithm](#-when-to-use-which-algorithm)
- [Limitations & Extensions](#-limitations--extensions)
- [Troubleshooting](#-troubleshooting)

---

## 📝 Overview

### The Clustering Problem

Clustering is the task of grouping a set of objects such that objects in the same group (cluster) are more similar to each other than to those in other groups. Unlike classification, clustering is **unsupervised** — the algorithm discovers structure without labeled training data.

**Real-world applications:**
- Customer segmentation for targeted marketing
- Document clustering for topic discovery
- Image segmentation in computer vision
- Anomaly detection in network security
- Gene expression analysis in bioinformatics

### Why Compare K-Means and EM/GMM?

These two algorithms represent fundamentally different approaches to clustering:

| Aspect | K-Means | EM (Gaussian Mixture) |
|--------|---------|----------------------|
| **Cluster Shape** | Spherical only | Elliptical (any covariance) |
| **Cluster Assignment** | Hard (each point belongs to exactly one cluster) | Soft (probabilistic membership) |
| **Underlying Model** | Distance-based (no probability model) | Probability-based (generative model) |
| **Boundary Type** | Linear (Voronoi diagram) | Non-linear (quadratic) |
| **Output** | Cluster centers + hard labels | Means, covariances, weights + probabilities |

By applying both to the same Iris dataset, this project demonstrates:
1. How cluster shape assumptions affect results
2. The difference between hard and soft clustering
3. When probabilistic membership provides better insight
4. How well unsupervised methods recover known class structure

---

## 🧠 Theoretical Foundations

### What is Clustering?

Formally, given a set of data points X = {x₁, x₂, ..., xₙ}, clustering partitions X into K groups (clusters) C = {C₁, C₂, ..., C_K} such that:

```
Cᵢ ∩ Cⱼ = ∅  for i ≠ j    (disjoint for hard clustering)
∪ Cᵢ = X                   (covers all data)
```

The objective is to maximize intra-cluster similarity and minimize inter-cluster similarity.

### The Iris Dataset as a Clustering Benchmark

The Iris dataset is particularly interesting for clustering because:

1. **One class is well-separated** (Setosa forms a distinct cluster)
2. **Two classes overlap** (Versicolor and Virginica share a boundary region)
3. **Ground truth exists** (we can evaluate clustering against known labels)
4. **Low dimensionality** (visualizable in 2D)

This makes it an ideal test case: algorithms should easily find Setosa but may struggle with the Versicolor-Virginica boundary.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- scikit-learn library
- pandas library
- NumPy library
- Matplotlib library

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/clustering-comparison.git
cd clustering-comparison

# Install dependencies
pip install scikit-learn pandas numpy matplotlib

# Ensure Iris.csv is in the project directory

# Run the comparison
python clustering_comparison.py
```

### Expected Output

The script will display:
1. A three-panel visualization comparing Real labels, K-Means clusters, and GMM clusters
2. Accuracy scores for both algorithms
3. Confusion matrices showing where each algorithm succeeds and fails

---

## 🔧 Algorithm Deep Dives

### K-Means Clustering

#### Intuition

K-Means partitions data into K clusters by minimizing the within-cluster sum of squared distances (inertia). Think of it as placing K "centroids" in the data space and assigning each point to the nearest centroid, then iteratively repositioning centroids to the center of their assigned points.

#### Algorithm Steps

```
1. Initialize: Randomly place K centroids in the feature space
2. Assignment Step: Assign each point to the nearest centroid
3. Update Step: Move each centroid to the mean of its assigned points
4. Repeat steps 2-3 until convergence (centroids stop moving)
```

#### Mathematical Objective

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**:

```
WCSS = Σ(i=1 to K) Σ(x ∈ Cᵢ) ||x - μᵢ||²

Where:
  μᵢ = (1/|Cᵢ|) Σ(x ∈ Cᵢ) x    (centroid of cluster i)
  ||x - μᵢ||² = Euclidean distance squared
```

#### Key Characteristics

- **Convergence:** Guaranteed to converge to a local minimum (not necessarily global)
- **Sensitivity to Initialization:** Different random starts produce different results
- **Spherical Clusters:** Assumes clusters are roughly spherical and equal-sized
- **Hard Assignment:** Each point belongs to exactly one cluster
- **Scale Sensitivity:** Features must be on comparable scales

#### The Random State Parameter

```python
KMeans(n_clusters=3, random_state=0)
```

`random_state=0` fixes the centroid initialization, ensuring reproducible results. Without it, different runs might produce different clusterings due to different initial centroid positions.

---

### Expectation-Maximization (EM) with Gaussian Mixture Models

#### Intuition

Instead of hard-assigning points to clusters, GMM takes a probabilistic approach: it assumes the data was generated from a mixture of K Gaussian distributions. Each cluster is a Gaussian "bump" in the data space, and each point has a probability of belonging to each cluster.

#### The Generative Model

GMM assumes data is generated by the following process:

```
For each data point x:
    1. Choose a cluster k with probability πₖ (mixing weight)
    2. Sample x from Gaussian distribution N(μₖ, Σₖ)
```

The model parameters are:
- **πₖ:** Mixing weights (how large each cluster is), Σπₖ = 1
- **μₖ:** Mean vector for each Gaussian component
- **Σₖ:** Covariance matrix for each Gaussian component

#### The EM Algorithm

EM iteratively refines parameter estimates through two steps:

**E-Step (Expectation):** Calculate the probability that each point belongs to each cluster (responsibilities):

```
γ(zₙₖ) = πₖ · N(xₙ | μₖ, Σₖ) / Σ(j=1 to K) πⱼ · N(xₙ | μⱼ, Σⱼ)

Where γ(zₙₖ) = probability that point n belongs to cluster k
```

**M-Step (Maximization):** Update parameters to maximize the expected complete-data log-likelihood:

```
μₖ_new = Σₙ γ(zₙₖ) · xₙ / Σₙ γ(zₙₖ)
Σₖ_new = Σₙ γ(zₙₖ) · (xₙ - μₖ)(xₙ - μₖ)ᵀ / Σₙ γ(zₙₖ)
πₖ_new = Σₙ γ(zₙₖ) / N
```

#### Covariance Types

GMM can use different covariance constraints:

| Covariance Type | Description | Parameters per Component |
|-----------------|-------------|--------------------------|
| **full** | Each component has its own general covariance matrix | d(d+1)/2 |
| **tied** | All components share the same covariance matrix | d(d+1)/2 |
| **diag** | Each component has its own diagonal covariance matrix | d |
| **spherical** | Each component has a single variance | 1 |

This implementation uses the default `full` covariance, allowing elliptical clusters with arbitrary orientation.

#### Key Characteristics

- **Soft Assignment:** Points have probabilistic membership in multiple clusters
- **Elliptical Clusters:** Can capture correlations between features
- **Generative Model:** Can generate new samples from learned distribution
- **Likelihood-Based:** Maximizes the probability of observing the data
- **More Parameters:** K(d + d(d+1)/2 + 1) parameters vs. Kd for K-Means

---

## 💻 Implementation Details

### Code Structure

```
clustering_comparison.py
├── Import libraries (sklearn, pandas, numpy, matplotlib)
├── Load dataset with column names
├── Extract features (X) and encode labels (y)
├── Create figure with three subplots
│   ├── Subplot 1: Ground Truth (Real Labels)
│   ├── Subplot 2: K-Means Clustering Results
│   └── Subplot 3: GMM Clustering Results
├── Apply K-Means (K=3)
│   ├── Fit model
│   ├── Predict clusters
│   ├── Display accuracy score
│   └── Display confusion matrix
├── Apply EM/GMM (n_components=3)
│   ├── Fit model
│   ├── Predict clusters
│   ├── Display accuracy score
│   └── Display confusion matrix
└── Render visualization
```

### Key Implementation Decisions

#### 1. Label Encoding

```python
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = [label[c] for c in dataset.iloc[:, -1]]
```

Ground truth labels are encoded numerically for comparison with cluster assignments. Both algorithms output numeric cluster labels (0, 1, 2).

#### 2. Feature Selection for Visualization

```python
plt.scatter(X.Petal_Length, X.Petal_Width, ...)
```

Petal length and petal width are chosen for 2D visualization because:
- They provide the best class separation among all feature pairs
- Setosa is clearly separated on these dimensions
- The Versicolor-Virginica overlap is visible but manageable

#### 3. Model Configuration

```python
# K-Means
model = KMeans(n_clusters=3, random_state=0).fit(X)

# GMM
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
```

Both are configured with:
- **3 clusters/components** — matching the known number of Iris species
- **random_state=0** — for reproducible results
- **All 4 features** used for fitting (Petal_Length, Petal_Width, Sepal_Length, Sepal_Width)

#### 4. Performance Evaluation

```python
# Accuracy
metrics.accuracy_score(y, model.labels_)

# Confusion Matrix
metrics.confusion_matrix(y, model.labels_)
```

Since cluster labels (0, 1, 2) may not correspond to class labels in the same order, accuracy scores may appear low even when clustering is good — the algorithm might label Setosa as cluster 1 instead of cluster 0. Scikit-learn's `accuracy_score` handles this by finding the optimal label permutation.

---

## 📊 Visualization Guide

### The Three-Panel Plot

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     REAL        │  │     K-MEANS     │  │      GMM        │
│                 │  │                 │  │                 │
│  ●●●   ▲▲▲     │  │  ●●●   ▲▲▲     │  │  ●●●   ▲▲▲     │
│  ●●●   ▲▲▲     │  │  ●●●   ▲▲▲     │  │  ●●■   ▲▲▲     │
│  ●     ▲ ▲     │  │  ●     ▲ ▲     │  │  ●     ▲ ■     │
│                 │  │                 │  │                 │
│  Setosa cluster │  │  K-Means finds  │  │  GMM captures   │
│  is well-sep.   │  │  similar groups │  │  soft boundaries │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

#### Panel 1: Ground Truth (Real Labels)
- **Red:** Iris-setosa
- **Lime:** Iris-versicolor
- **Black:** Iris-virginica

Shows the actual species labels. Setosa forms a tight, well-separated cluster in the bottom-left. Versicolor and Virginica form overlapping clusters — there's no clean boundary between them.

#### Panel 2: K-Means Results
Shows how K-Means partitioned the data. Look for:
- Does K-Means correctly identify the Setosa cluster?
- How does it handle the overlapping region between Versicolor and Virginica?
- Are the boundaries linear (as K-Means produces)?

#### Panel 3: GMM Results
Shows how the Gaussian Mixture Model clustered the data. Compare with K-Means:
- Are the boundaries more flexible (curved)?
- Does GMM better handle the overlapping region?
- Which points are classified differently between K-Means and GMM?

---

## 📐 Mathematical Comparison

### Objective Functions

| Algorithm | Objective | Formula |
|-----------|-----------|---------|
| **K-Means** | Minimize within-cluster variance | WCSS = Σ Σ \|\|x - μᵢ\|\|² |
| **EM/GMM** | Maximize log-likelihood | log P(X\|θ) = Σ log(Σ πₖ N(x\|μₖ, Σₖ)) |

### Decision Boundaries

**K-Means** produces linear boundaries (Voronoi tessellation):

The boundary between clusters i and j is the perpendicular bisector of the line connecting their centroids. This creates convex, polygonal regions.

**GMM** produces quadratic boundaries:

The boundary between components i and j satisfies:
```
πᵢ N(x|μᵢ, Σᵢ) = πⱼ N(x|μⱼ, Σⱼ)
```
This is a quadratic equation in x, producing curved boundaries that can better follow natural data contours.

### Cluster Shape Comparison

```
K-Means assumes:              GMM allows:
    
    ○  ○                        ○  ○
  ○  ○  ○  ○                   ○ ○  ○
○  ○  ●  ○  ○                ○ ○  ●  ○ ○
  ○  ○  ○  ○                 ○  ○○○  ○
    ○  ○                      ○  ○○○  ○

Spherical, equal size       Elliptical, any orientation
```

### Computational Complexity

| Algorithm | Time per Iteration | Space | Convergence Rate |
|-----------|-------------------|-------|------------------|
| **K-Means** | O(n·K·d) | O(n·d + K·d) | Typically fast (10-100 iterations) |
| **EM/GMM** | O(n·K·d²) for full covariance | O(K·d²) | Slower (may need more iterations) |

Where n = number of samples, K = number of clusters, d = number of features.

---

## 📈 Interpreting Results

### Understanding Accuracy Scores

The accuracy score compares cluster assignments to true labels:

```
Accuracy = Number of correctly assigned points / Total points
```

**Expected results on Iris:**
- K-Means: ~88-90% accuracy
- GMM: ~90-97% accuracy (typically higher)

GMM often outperforms K-Means on Iris because:
1. The Versicolor-Virginica overlap region benefits from soft boundaries
2. The full covariance matrix captures the elliptical shape of clusters
3. Probabilistic assignment handles borderline points better

### Reading the Confusion Matrix

A typical confusion matrix for K-Means on Iris:

```
                 Predicted Cluster
                 0    1    2
Actual   Setosa   50    0    0
         Versi     0   48    2
         Virgi     0   14   36
```

**Interpretation:**
- Setosa: Perfectly clustered (50/50) — well-separated
- Versicolor: Mostly correct (48/50) with 2 confused as Virginica
- Virginica: Moderate accuracy (36/50) with 14 confused as Versicolor
- The confusion is asymmetric — Virginica is more often misclustered than Versicolor

### Why Label Permutation Matters

Cluster labels (0, 1, 2) are arbitrary — the algorithm might call Setosa "cluster 1" and Virginica "cluster 0." The accuracy score automatically finds the best permutation, but confusion matrices should be read with this in mind.

---

## 🎯 When to Use Which Algorithm

### Choose K-Means When:

- Clusters are expected to be roughly spherical and equal-sized
- Computational efficiency is important (large datasets)
- Interpretability is desired (centroids are meaningful)
- A simple, well-understood baseline is needed
- Hard cluster assignments are acceptable

### Choose EM/GMM When:

- Clusters may have elliptical shapes or different sizes
- Soft (probabilistic) cluster membership is valuable
- The data is believed to come from a mixture of distributions
- Uncertainty quantification matters (probability of belonging)
- Overlapping clusters are expected

### Decision Flowchart

```
Are clusters likely spherical and equal-sized?
    ├── YES → Is speed critical?
    │         ├── YES → K-Means
    │         └── NO  → Both will work; compare results
    └── NO  → GMM (elliptical clusters)
    
Do you need probability of cluster membership?
    ├── YES → GMM (provides responsibilities)
    └── NO  → K-Means is sufficient
    
Is interpretability important?
    ├── YES → K-Means (centroids are easily explained)
    └── NO  → GMM (model parameters are less intuitive)
```

---

## 🚀 Limitations & Extensions

### Current Limitations

1. **K Must Be Specified:** Both algorithms require the number of clusters to be known in advance — unrealistic for true unsupervised scenarios.

2. **Local Optima:** Both algorithms converge to local optima. K-Means is particularly sensitive to initialization.

3. **Feature Scaling:** K-Means is sensitive to feature scales. GMM with full covariance handles scaling somewhat better but still benefits from preprocessing.

4. **Spherical Assumption (K-Means):** K-Means cannot find non-spherical clusters, limiting its applicability to certain data distributions.

5. **Outlier Sensitivity:** Both algorithms are affected by outliers. K-Means centroids are pulled toward outliers; GMM can create small components to accommodate them.

### Potential Extensions

#### Determining Optimal K

Use the Elbow Method (K-Means) or BIC/AIC (GMM):

```python
# Elbow Method for K-Means
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
```

```python
# BIC for GMM
n_components_range = range(1, 11)
models = [GaussianMixture(n, random_state=0).fit(X) for n in n_components_range]

bics = [m.bic(X) for m in models]
aics = [m.aic(X) for m in models]

plt.plot(n_components_range, bics, label='BIC')
plt.plot(n_components_range, aics, label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.legend()
plt.show()
```

#### Feature Scaling

Add preprocessing for improved results:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use X_scaled instead of X
model = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
```

#### Multiple Initializations (K-Means)

Reduce sensitivity to initialization:

```python
model = KMeans(n_clusters=3, n_init=20, random_state=0).fit(X)
```

`n_init=20` runs the algorithm 20 times with different centroid seeds and keeps the best result.

#### Different Covariance Types (GMM)

Compare different covariance constraints:

```python
covariance_types = ['full', 'tied', 'diag', 'spherical']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for cov_type, ax in zip(covariance_types, axes.ravel()):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=0)
    gmm.fit(X)
    y_pred = gmm.predict(X)
    ax.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_pred])
    ax.set_title(f'GMM ({cov_type})')
plt.show()
```

#### Silhouette Score Evaluation

Evaluate clustering quality without ground truth:

```python
from sklearn.metrics import silhouette_score

# K-Means
km_labels = KMeans(n_clusters=3, random_state=0).fit_predict(X)
km_silhouette = silhouette_score(X, km_labels)

# GMM
gmm_labels = GaussianMixture(n_components=3, random_state=0).fit_predict(X)
gmm_silhouette = silhouette_score(X, gmm_labels)

print(f"K-Means Silhouette Score: {km_silhouette:.3f}")
print(f"GMM Silhouette Score: {gmm_silhouette:.3f}")
```

#### Cluster Visualization in Higher Dimensions

Use PCA to visualize all 4 features:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot clusters in PCA space
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colormap[model.labels_])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Clusters Visualized in PCA Space')
plt.show()
```

#### Anomaly Detection with GMM

Use probability scores to identify outliers:

```python
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
log_probs = gmm.score_samples(X)

# Points with very low probability are potential anomalies
threshold = np.percentile(log_probs, 5)  # Bottom 5%
anomalies = X[log_probs < threshold]
print(f"Found {len(anomalies)} potential anomalies")
```

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `FileNotFoundError: Iris.csv` | Dataset not found | Place file in working directory or update path |
| All points assigned to one cluster | Poor initialization (K-Means) | Increase `n_init` or check feature scaling |
| GMM accuracy is very low | Label permutation issue | Check confusion matrix to understand mapping |
| `ValueError: n_clusters > n_samples` | K > number of data points | Reduce n_clusters |
| Plot doesn't show | Missing `plt.show()` | Add `plt.show()` at end of script |
| Colors don't match across plots | Different label ordering | Use predicted labels directly for coloring |


*"Clustering is not about finding the 'correct' clusters — it's about finding useful ones. The best clustering is the one that serves your purpose."*
