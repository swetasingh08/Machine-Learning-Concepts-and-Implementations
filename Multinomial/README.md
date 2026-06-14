

# Text Classification with Naive Bayes

### Multinomial Naive Bayes on the 20 Newsgroups Dataset

This project implements a **text classification pipeline** using **Multinomial Naive Bayes** with **TF-IDF vectorization** to categorize news articles into four distinct topics. The implementation demonstrates the complete NLP machine learning workflow: text preprocessing, feature extraction, model training, and comprehensive evaluation.


---

## 📑 Table of Contents

- [Overview](#-overview)
- [Theoretical Foundations](#-theoretical-foundations)
- [Quick Start](#-quick-start)
- [The 20 Newsgroups Dataset](#-the-20-newsgroups-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Algorithm Deep Dives](#-algorithm-deep-dives)
- [Implementation Details](#-implementation-details)
- [Interpreting Results](#-interpreting-results)
- [Mathematical Deep Dive](#-mathematical-deep-dive)
- [Comparison with Other Text Classifiers](#-comparison-with-other-text-classifiers)
- [Limitations & Extensions](#-limitations--extensions)
- [Troubleshooting](#-troubleshooting)

---

## 📝 Overview

### The Text Classification Problem

Text classification is the task of assigning predefined categories to free-text documents. It's one of the most fundamental and widely-used NLP tasks, with applications including:

- **Spam detection:** Classifying emails as spam or legitimate
- **Sentiment analysis:** Determining if a review is positive or negative
- **Topic labeling:** Assigning news articles to categories
- **Language identification:** Detecting the language of a document
- **Intent recognition:** Understanding what a user wants from a chatbot query

### Why Naive Bayes for Text?

Naive Bayes has been a cornerstone of text classification for decades, despite (and sometimes because of) its "naive" assumptions:

| Property | Why It Matters for Text |
|----------|------------------------|
| **Fast training** | Text datasets can be enormous (millions of documents) |
| **Fast prediction** | Real-time classification needed for many applications |
| **Handles high dimensions** | Text vocabularies can have 100,000+ unique words |
| **Probabilistic output** | Confidence scores help with threshold-based decisions |
| **Robust to irrelevant features** | Many words don't help classification; NB handles this gracefully |
| **Interpretable** | Can show which words most influenced each classification |

### What This Implementation Demonstrates

- Loading and exploring the 20 Newsgroups dataset (a standard NLP benchmark)
- Building a scikit-learn pipeline that chains TF-IDF vectorization with Multinomial Naive Bayes
- Training on a multi-class text classification problem
- Comprehensive model evaluation using accuracy, classification reports, and confusion matrices
- The complete workflow from raw text to evaluated predictions

---

## 🧠 Theoretical Foundations

### Bayes' Theorem for Classification

Naive Bayes applies Bayes' theorem with the "naive" assumption of conditional independence between features:

```
P(Cₖ | x) = P(Cₖ) · P(x | Cₖ) / P(x)

Where:
  P(Cₖ | x)  = Posterior probability of class k given document x
  P(Cₖ)      = Prior probability of class k
  P(x | Cₖ)  = Likelihood of document x given class k
  P(x)       = Evidence (normalization constant)
```

The predicted class is:

```
ŷ = argmaxₖ P(Cₖ) · P(x | Cₖ)
```

### The "Naive" Assumption

The critical simplifying assumption:

```
P(x | Cₖ) = P(w₁ | Cₖ) · P(w₂ | Cₖ) · ... · P(wₙ | Cₖ)
```

This assumes that the probability of each word appearing is independent of every other word, given the class. This is clearly false in natural language (words like "New" and "York" frequently co-occur), but it works remarkably well in practice for classification.

### Why the Naive Assumption Works

1. **Classification, not probability estimation:** We only need the correct ranking of classes, not accurate probabilities
2. **Bias cancels out:** The independence assumption biases all classes similarly
3. **Document length normalization:** Longer documents don't dominate because probabilities are multiplied
4. **Optimal in many cases:** NB is the optimal classifier when features are independent given the class

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- scikit-learn library
- pandas library

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/text-classification-nb.git
cd text-classification-nb

# Install dependencies
pip install scikit-learn pandas

# Run the classifier
python text_classifier.py
```

### Expected Output

The script will display:
1. All 20 available newsgroup categories
2. The 4 selected categories for classification
3. Model accuracy score
4. Detailed classification report (precision, recall, F1-score per class)
5. Confusion matrix showing prediction patterns

### First Run Note

The first execution downloads the 20 Newsgroups dataset (~20MB). Subsequent runs use the cached copy.

---

## 📰 The 20 Newsgroups Dataset

### Dataset Overview

The 20 Newsgroups dataset is one of the most widely-used benchmarks in text classification and machine learning. Collected by Ken Lang in 1995, it contains approximately 20,000 newsgroup documents partitioned across 20 different newsgroups.

### Key Characteristics

| Property | Value |
|----------|-------|
| **Total documents** | ~18,846 |
| **Number of classes** | 20 |
| **Documents per class** | ~942 (roughly balanced) |
| **Document length** | Varies from short messages to long articles |
| **Train/Test split** | Predefined (by date) |
| **Language** | English |
| **Source** | Usenet newsgroups (early internet discussion forums) |

### Selected Categories

This implementation focuses on 4 categories that form two thematic pairs:

| Pair | Categories | Relationship |
|------|------------|--------------|
| **Religion** | `alt.atheism`, `soc.religion.christian` | Similar vocabulary, opposite viewpoints |
| **Technology** | `comp.graphics`, `sci.med` | Different but potentially overlapping terminology |

This selection is intentional — the religion pair tests whether NB can distinguish documents with similar vocabulary but different perspectives, while the tech pair tests classification of related but distinct domains.

### Available Categories Reference

```
 1. alt.atheism                   11. rec.sport.hockey
 2. comp.graphics                 12. sci.crypt
 3. comp.os.ms-windows.misc       13. sci.electronics
 4. comp.sys.ibm.pc.hardware      14. sci.med
 5. comp.sys.mac.hardware         15. sci.space
 6. comp.windows.x                16. soc.religion.christian
 7. misc.forsale                  17. talk.politics.guns
 8. rec.autos                     18. talk.politics.mideast
 9. rec.motorcycles               19. talk.politics.misc
10. rec.sport.baseball            20. talk.religion.misc
```

---

## 🏗️ Pipeline Architecture

### The Two-Stage Pipeline

```
Raw Text → [TF-IDF Vectorizer] → Numerical Features → [Multinomial NB] → Class Prediction
```

The pipeline encapsulates both stages into a single object that can be fit and used for prediction in one step.

### Stage 1: TF-IDF Vectorization

**Purpose:** Convert raw text documents into numerical feature vectors.

**Process:**
1. **Tokenization:** Split documents into individual words
2. **Vocabulary building:** Create a dictionary of all unique words in the training corpus
3. **TF-IDF computation:** For each word in each document, calculate:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

Where:
  TF(t, d)  = count of term t in document d / total terms in d
  IDF(t)    = log(total documents / documents containing t) + 1
```

**Why TF-IDF instead of raw counts?**
- Downweights common words ("the", "a", "is") that appear in all documents
- Upweights rare but informative words that characterize specific topics
- Better captures what makes a document unique

### Stage 2: Multinomial Naive Bayes

**Purpose:** Learn the probability distribution of words for each class and classify new documents.

**Training:**
1. Calculate prior probabilities P(Cₖ) for each class
2. Calculate likelihood P(w | Cₖ) for each word given each class
3. Apply Laplace smoothing to handle unseen words

**Prediction:**
1. For a new document, compute log-posterior for each class
2. Return the class with highest log-posterior

---

## 🔧 Algorithm Deep Dives

### Multinomial Naive Bayes

The Multinomial variant is specifically designed for discrete features like word counts. It models documents as generated by repeatedly drawing words from a multinomial distribution.

#### Document Generation Model

Under the Multinomial NB model, a document of class Cₖ is generated by:

```
1. Choose document length N ~ Poisson or fixed
2. For each of the N word positions:
      Draw a word w from the multinomial distribution θₖ
```

#### Training: Parameter Estimation

**Prior probabilities:**
```
P(Cₖ) = Nₖ / N

Where:
  Nₖ = number of documents in class k
  N  = total number of documents
```

**Word likelihoods (with Laplace smoothing):**
```
P(w | Cₖ) = (count(w, Cₖ) + α) / (Σ_v count(v, Cₖ) + α|V|)

Where:
  count(w, Cₖ) = occurrences of word w in class k documents
  α             = smoothing parameter (default α=1.0)
  |V|           = vocabulary size
```

#### Prediction: Log-Space Computation

To avoid numerical underflow from multiplying many small probabilities, NB works in log-space:

```
log P(Cₖ | x) = log P(Cₖ) + Σ(i=1 to n) log P(wᵢ | Cₖ)
```

The predicted class:

```
ŷ = argmaxₖ [log P(Cₖ) + Σ(i=1 to n) log P(wᵢ | Cₖ)]
```

### Laplace (Additive) Smoothing

**The problem:** If a word appears in test documents but never appeared in training documents for a particular class, P(w | Cₖ) = 0, making the entire posterior probability zero.

**The solution:** Add a small count α to every word:

```
Without smoothing: P("xyzzy" | Cₖ) = 0/1000 = 0  → kills the class
With smoothing:    P("xyzzy" | Cₖ) = (0+1)/(1000+1×|V|) ≈ small positive value
```

α=1 is standard Laplace smoothing. Higher α values produce more uniform (less discriminative) distributions.

---

## 💻 Implementation Details

### Code Structure

```
text_classifier.py
├── Import libraries
├── Load full 20 Newsgroups dataset
├── Display all 20 available categories
├── Define 4 target categories
├── Load train and test subsets
├── Display selected categories
├── Build scikit-learn Pipeline:
│   ├── TfidfVectorizer (feature extraction)
│   └── MultinomialNB (classifier)
├── Train model on training data
├── Predict on test data
├── Calculate and display accuracy
├── Generate classification report
│   ├── Precision per class
│   ├── Recall per class
│   ├── F1-score per class
│   └── Support (number of samples) per class
├── Generate confusion matrix
└── Display formatted confusion matrix table
```

### Key Implementation Decisions

#### 1. Pipeline Construction

```python
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
```

**Why a Pipeline?**
- Encapsulates the full workflow in a single object
- Prevents data leakage (TF-IDF is fit only on training data)
- Enables easy cross-validation and grid search
- Simplifies deployment (single model object to serialize)

#### 2. Default Parameters

```python
TfidfVectorizer()     # All defaults
MultinomialNB()        # All defaults (alpha=1.0)
```

The default TF-IDF vectorizer:
- Tokenizes on word boundaries (regex: `r'(?u)\b\w\w+\b'`)
- Lowercases all text
- Removes English stop words? No (default `stop_words=None`)
- Uses L2 normalization on document vectors
- Sublinear TF scaling? No (default `sublinear_tf=False`)

The default MultinomialNB:
- Uses Laplace smoothing with α=1.0
- Fits class prior from data (`fit_prior=True`)
- Assumes all classes appear in training data

#### 3. Train/Test Split

```python
news_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
news_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
```

The 20 Newsgroups dataset provides predefined train/test splits based on posting dates — earlier posts for training, later posts for testing. This simulates real-world deployment where models are trained on historical data and applied to future data.

---

## 📊 Interpreting Results

### Accuracy Interpretation

A typical accuracy for this 4-class problem is **90-97%**. This high accuracy is expected because:
- The categories are semantically distinct (religion vs. technology)
- The religion pair, while related, uses different vocabulary (atheism vs. christianity terms)
- The technology pair is about different domains (computer graphics vs. medicine)

### Classification Report Reading Guide

```
                    precision    recall  f1-score   support

    alt.atheism         0.95      0.93      0.94       319
soc.religion.christian  0.96      0.97      0.97       398
     comp.graphics      0.92      0.90      0.91       389
          sci.med       0.94      0.96      0.95       396

         accuracy                           0.94      1502
        macro avg       0.94      0.94      0.94      1502
     weighted avg       0.94      0.94      0.94      1502
```

**Understanding Each Metric:**

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Precision** | TP / (TP + FP) | When the model predicts this class, how often is it correct? |
| **Recall** | TP / (TP + FN) | Of all actual instances of this class, how many did the model find? |
| **F1-Score** | 2 × (P×R)/(P+R) | Harmonic mean of precision and recall — best single metric |
| **Support** | Count | Number of test samples in this class |

**Macro Avg:** Simple average across classes (treats all classes equally)
**Weighted Avg:** Average weighted by class support (accounts for class imbalance)

### Confusion Matrix Reading Guide

```
                         alt.atheism  soc.religion.christian  comp.graphics  sci.med
alt.atheism                    297                        8              2        12
soc.religion.christian           5                      388              1         4
comp.graphics                    4                        2            350        33
sci.med                          3                        1             15       377
```

**Reading the matrix:**
- **Rows:** Actual (true) class
- **Columns:** Predicted class
- **Diagonal (top-left to bottom-right):** Correct predictions
- **Off-diagonal:** Misclassifications

**Example interpretation:**
- 297 documents were correctly classified as `alt.atheism`
- 8 documents that were actually `alt.atheism` were misclassified as `soc.religion.christian`
- 12 documents that were actually `alt.atheism` were misclassified as `sci.med`
- 33 documents that were actually `comp.graphics` were misclassified as `sci.med` (largest confusion)

**Key insight from confusion patterns:**
- Religion pair (atheism vs. christian) has minimal confusion — good separation
- `comp.graphics` ↔ `sci.med` has the most confusion — possibly shared technical vocabulary
- Cross-domain confusion (religion misclassified as technology) is rare

---

## 📐 Mathematical Deep Dive

### TF-IDF Formula

The TfidfVectorizer with default parameters computes:

**Term Frequency (TF):**
```
TF(t, d) = count(t, d) / |d|
```
Where count(t,d) is the raw count of term t in document d, and |d| is the total number of terms in d.

**Inverse Document Frequency (IDF):**
```
IDF(t) = log((1 + N) / (1 + df(t))) + 1
```
Where N is the total number of documents and df(t) is the number of documents containing term t.

The "+1" in numerator and denominator prevents division by zero, and the final "+1" ensures IDF ≥ 1 for all terms.

**TF-IDF:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

After computing raw TF-IDF values, the vectorizer applies L2 normalization:
```
v_normalized = v / ||v||₂
```

### Naive Bayes Log-Space Computation

For numerical stability, all computations are performed in log-space:

**Log Prior:**
```
log P(Cₖ) = log(Nₖ) - log(N)
```

**Log Likelihood:**
```
log P(w | Cₖ) = log(count(w, Cₖ) + α) - log(Σ_v count(v, Cₖ) + α|V|)
```

**Log Posterior (Classification Score):**
```
score(Cₖ) = log P(Cₖ) + Σ(w in document) log P(w | Cₖ) × TF-IDF_weight(w)
```

The class with the highest score is selected.

### Why NB Works Well on Text

Despite its simplicity, NB excels at text classification because:

1. **High-dimensional sparse data:** Text creates thousands of features, many of which are zero. NB handles sparsity naturally.

2. **Word independence is partially true:** In text, words often serve as independent clues to the topic. "Linux," "kernel," and "compile" independently suggest "computers" regardless of word order.

3. **Document length normalization:** Using logs and priors means document length doesn't bias classification.

4. **Complementary to TF-IDF:** TF-IDF pre-weights informative words, making the independence assumption more reasonable.

---

## 📊 Comparison with Other Text Classifiers

| Algorithm | Accuracy | Training Speed | Prediction Speed | Interpretability |
|-----------|----------|----------------|------------------|------------------|
| **Multinomial NB** | 90-95% | Very Fast | Very Fast | High (word probabilities) |
| **Logistic Regression** | 92-97% | Moderate | Very Fast | High (feature weights) |
| **Linear SVM** | 93-97% | Moderate | Very Fast | Moderate |
| **Random Forest** | 88-93% | Moderate | Moderate | Moderate |
| **Neural Networks** | 94-98% | Slow | Fast | Low |
| **k-NN** | 85-90% | None (lazy) | Very Slow | High (show neighbors) |

### When Naive Bayes is the Best Choice

- **Baseline model:** Quick to implement, provides a performance floor
- **Real-time requirements:** Prediction is just a few array lookups and additions
- **Interpretability needed:** Can show exactly which words drove a classification
- **Limited training data:** NB often outperforms more complex models with small datasets
- **Deployment constraints:** Model size is small (just word probability tables)

---

## 🚀 Limitations & Extensions

### Current Limitations

1. **Naive Independence Assumption:** Fails when word order matters or words are strongly correlated
2. **Zero-Frequency Problem:** Words unseen in training cause problems (mitigated by smoothing)
3. **Document Length:** Very short documents may not contain enough words for reliable classification
4. **Semantic Understanding:** NB treats "good" and "excellent" as completely different features with no understanding of synonymy
5. **Class Imbalance:** Strong class imbalance can bias predictions toward majority classes

### Potential Extensions

#### Stop Word Removal

Remove common words that provide little discriminative information:

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])
```

#### N-gram Features

Capture word order with bigrams and trigrams:

```python
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # Unigrams + Bigrams
    ('nb', MultinomialNB())
])
```

#### Hyperparameter Tuning

Optimize alpha (smoothing) and other parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__min_df': [1, 2, 3],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.1, 0.5, 1.0, 2.0]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(news_train.data, news_train.target)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)
```

#### Complement Naive Bayes

Use ComplementNB which often outperforms MultinomialNB on imbalanced data:

```python
from sklearn.naive_bayes import ComplementNB

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', ComplementNB())
])
```

#### Feature Importance Analysis

Identify the most informative words for each class:

```python
feature_names = model.named_steps['tfidf'].get_feature_names_out()
nb_model = model.named_steps['nb']

for i, category in enumerate(news_train.target_names):
    top_features_idx = np.argsort(nb_model.feature_log_prob_[i])[-10:]
    top_features = [feature_names[idx] for idx in top_features_idx]
    print(f"\n{category}:")
    print("  Top words:", ", ".join(reversed(top_features)))
```

#### Cross-Validation Evaluation

Replace single train/test evaluation with cross-validation:

```python
from sklearn.model_selection import cross_val_score

# Combine train and test data
all_data = news_train.data + news_test.data
all_targets = list(news_train.target) + list(news_test.target)

scores = cross_val_score(model, all_data, all_targets, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

#### Prediction with Confidence Scores

Get probability estimates instead of just class labels:

```python
probabilities = model.predict_proba(news_test.data)

# Show top 2 predictions with confidence for first document
doc_idx = 0
top2_idx = np.argsort(probabilities[doc_idx])[-2:][::-1]
for idx in top2_idx:
    print(f"{news_test.target_names[idx]}: {probabilities[doc_idx][idx]:.3f}")
```

#### Handling New Categories

Extend to all 20 categories:

```python
news_all = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(
    news_all.data, news_all.target, test_size=0.3, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"20-class accuracy: {metrics.accuracy_score(y_test, y_pred):.3f}")
```

#### Saving and Loading the Model

```python
import joblib

# Save
joblib.dump(model, 'text_classifier_pipeline.pkl')

# Load
loaded_model = joblib.load('text_classifier_pipeline.pkl')
predictions = loaded_model.predict(new_texts)
```

---

## 🔧 Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Download hangs on first run | Network issue fetching dataset | Download manually from sklearn source |
| All predictions are one class | Severe class imbalance or bad features | Check class distribution; adjust preprocessing |
| Very low accuracy (<50%) | Wrong categories or encoding issue | Verify categories exist in dataset |
| Memory error | Too many features | Use `max_features` parameter in TfidfVectorizer |
| `ValueError: dimension mismatch` | Different vocabularies in train/test | Use pipeline (handles this automatically) |



---

*"The Naive Bayes classifier is like a diligent student who remembers every word in the textbook but doesn't understand any of the sentences — and yet somehow still aces the exam."*
