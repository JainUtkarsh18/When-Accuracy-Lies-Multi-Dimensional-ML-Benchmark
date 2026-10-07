# ML_Model_Showdown-Performance_Stability_Efficiency

A reusable machine learning robustness benchmarking project for evaluating multiple classification algorithms across diverse datasets under clean and progressively corrupted data conditions.

The project goes beyond conventional clean-test performance by investigating how machine learning models behave when real-world data quality deteriorates through feature noise, missing values, outliers, label noise, distribution shift, and feature corruption.

The benchmark focuses on **performance, stability, degradation behaviour, and graceful failure** rather than simply identifying the model with the highest clean-test score.

---

## Kaggle Notebook

The original notebook **"When Accuracy Lies: A Multi-Dimensional ML Benchmark"** is available on [Kaggle](https://www.kaggle.com/code/utkarshjain76/when-accuracy-lies-multi-dimensional-ml-benchmark).

The notebook is designed as an optimized multi-dimensional machine learning benchmark that evaluates representative model families across multiple classification datasets and controlled stress conditions.

The benchmark investigates the question:

> **Which machine-learning models fail gracefully when the assumptions of clean training and test data are progressively violated?**

---

## Project Overview

Traditional machine learning benchmarks often ask:

> **Which model achieves the highest score on clean test data?**

However, real-world data rarely remains clean.

Features can become noisy, values can go missing, extreme observations can appear, labels can be incorrect, important feature relationships can be disrupted, and deployment distributions can differ from training data.

This project therefore investigates:

> **How stable are different machine-learning models when data quality and distribution progressively deteriorate?**

Instead of evaluating models only by their initial performance, the benchmark studies their **degradation profiles**.

The primary research question is:

> **Which model families maintain predictive performance most consistently as data quality and distribution deteriorate?**

The project does **not** attempt to identify a universal best model. Instead, it examines how model behaviour changes under different failure conditions.

---

## Research Hypotheses

The benchmark investigates several hypotheses:

### H1 — Different degradation patterns

Different model families will exhibit different degradation patterns under the same corruption.

### H2 — Clean performance does not guarantee robustness

The model with the highest clean-test performance will not necessarily experience the lowest degradation.

### H3 — Robustness is corruption-dependent

Different model families may have different robustness profiles depending on the type of corruption.

### H4 — Model rankings can change

Model rankings observed on clean data may change under severe corruption or distribution shift.

---

## Features

* Multi-dataset machine learning robustness benchmarking
* Six representative classification model families
* Multiple controlled data corruption experiments
* Leakage-safe preprocessing pipelines
* Reproducible train/test splits
* Multiple random seeds
* Weighted F1-based evaluation
* Accuracy, precision, recall, and balanced accuracy analysis
* ROC-AUC evaluation where supported
* Relative performance degradation analysis
* Stability Score calculation
* Degradation Slope analysis
* Graceful Failure Index (GFI)
* Performance degradation curves
* Stability heatmaps
* Clean performance vs robustness comparison
* Multi-dimensional final benchmark summary
* Optimized benchmark execution
* Automated CSV result exports

---

## Benchmark Workflow

```text
Datasets
    │
    ▼
Train/Test Split
    │
    ▼
Leakage-Safe Preprocessing
    │
    ▼
Train Model
    │
    ▼
Clean Performance Baseline
    │
    ▼
Controlled Data Corruption
    │
    ├── Feature Noise
    ├── Missing Data
    ├── Outliers
    ├── Label Noise
    ├── Distribution Shift
    └── Feature Corruption
    │
    ▼
Repeated Evaluation
    │
    ▼
Relative Degradation
    │
    ├── Stability Score
    ├── Degradation Slope
    └── Graceful Failure Index
    │
    ▼
Robustness Analysis
    │
    ▼
Multi-Dimensional Model Comparison
```

---

## Datasets

The benchmark uses three built-in scikit-learn classification datasets with different structural properties.

### Breast Cancer Wisconsin

A small binary classification dataset containing diagnostic measurements.

* Binary classification
* Numerical features
* Relatively small sample size

### Wine

A multiclass classification dataset based on chemical measurements of wine samples.

* Multiclass classification
* Continuous numerical features
* Different class structure from the binary dataset

### Digits

A higher-dimensional multiclass dataset containing pixel-derived representations of handwritten digits.

* Multiclass classification
* 64 pixel-based features
* Higher dimensionality
* Larger sample size

Together, these datasets provide variation in:

* Number of samples
* Number of features
* Number of classes
* Feature distributions
* Classification complexity

---

## Machine Learning Models

The benchmark compares six representative model families.

### Linear

* Logistic Regression

### Tree-Based

* Decision Tree
* Random Forest

### Margin-Based

* Support Vector Machine (SVM)

### Boosting

* Histogram Gradient Boosting

### Neural

* Multi-Layer Perceptron (MLP)

The benchmark intentionally focuses on representative model families rather than evaluating many closely related algorithms.

---

## Evaluation Metrics

The benchmark evaluates models across multiple dimensions.

### Primary Metric

**Weighted F1 Score**

Weighted F1 is used as the primary metric because it can be applied to both binary and multiclass classification while accounting for class support.

### Secondary Metrics

* Accuracy
* Precision
* Recall
* Balanced Accuracy
* ROC-AUC

ROC-AUC is calculated where the underlying model provides a compatible probability or decision score.

---

## Stress Tests

The central component of the benchmark is controlled data corruption.

Six different stress tests are implemented.

### 1. Feature Noise

Gaussian noise is added to numerical feature values.

**Question:**

> How sensitive is each model to measurement noise?

---

### 2. Missing Data

A controlled proportion of feature values is randomly removed.

**Question:**

> How does model performance change when information becomes incomplete?

---

### 3. Outliers

A fraction of observations receives large numerical perturbations.

**Question:**

> Which models are sensitive to extreme observations?

---

### 4. Label Noise

A proportion of training labels is randomly replaced with incorrect labels.

**Question:**

> How sensitive is learning to incorrect supervision?

Unlike the other stress tests, label noise modifies the training data and therefore requires model retraining.

---

### 5. Distribution Shift

Feature values are systematically shifted away from the original distribution.

**Question:**

> How well does the model generalize when deployment conditions differ from the training distribution?

---

### 6. Feature Corruption

Selected feature columns are shuffled across observations.

**Question:**

> How dependent is a model on stable feature relationships?

---

## Runtime Modes

The notebook supports two execution modes.

### Fast Mode

Fast Mode is designed for normal Kaggle execution and development.

It uses:

* 2 random seeds
* Fewer Random Forest trees
* Fewer MLP iterations
* Reduced corruption levels

```python
FAST_MODE = True
```

### Full Mode

Full Mode provides a deeper benchmark using:

* 5 random seeds
* More Random Forest trees
* More MLP iterations
* Denser corruption levels

```python
FAST_MODE = False
```

Full Mode is recommended for a final research-oriented benchmark.

---

## Optimized Benchmark Engine

A major feature of this notebook is its optimized benchmark design.

A naive implementation would repeatedly train every model for every corruption level:

```text
Dataset
   ×
Seed
   ×
Stress Test
   ×
Corruption Level
   ×
Model
   ×
Training
```

This creates unnecessary computational overhead for test-time corruption.

The optimized implementation instead performs:

```text
Dataset
   ×
Seed
   ×
Model
   ×
One Clean Fit
```

The fitted model is then reused for test-time corruption experiments.

```text
Clean Model
    │
    ├── Clean Test Data
    ├── Feature Noise
    ├── Missing Data
    ├── Outliers
    ├── Distribution Shift
    └── Feature Corruption
```

Only **label noise** requires retraining because it modifies the training labels.

This significantly reduces unnecessary model fitting while preserving the intended stress-testing methodology.

---

## Relative Performance and Degradation

Raw performance scores do not fully describe robustness.

The notebook calculates relative degradation as:

$$
D =
\frac{S_{clean}-S_{corrupted}}
{S_{clean}}
$$

For example, if a model has:

```text
Clean F1       = 0.95
Corrupted F1   = 0.85
```

then:

$$
D =
\frac{0.95-0.85}{0.95}
\approx 0.105
$$

The model therefore experiences approximately **10.5% relative degradation**.

The notebook also calculates normalized relative performance:

$$
R_i =
\frac{Score_i}{Score_{clean}}
$$

A value close to:

```text
1.0
```

indicates that the model retains most of its clean-test performance.

---

## Stability Score

The **Stability Score** represents the mean normalized performance retained across corruption levels.

Conceptually:

```text
Stability Score
       │
       ▼
How much clean performance
does the model retain?
```

A score close to **1.0** indicates that the model maintains a large proportion of its baseline performance.

The metric allows models with different clean-test scores to be compared based on their degradation behaviour.

---

## Degradation Slope

The notebook fits a simple regression line to each degradation curve:

$$
Performance =
\beta_0 +
\beta_1(Corruption)
$$

The coefficient:

$$
\beta_1
$$

is used as the **Degradation Slope**.

Interpretation:

```text
Near zero
    ↓
Relatively stable

Moderately negative
    ↓
Gradual degradation

Strongly negative
    ↓
Rapid degradation
```

The slope is intended as a descriptive summary rather than a claim that degradation is perfectly linear.

---

## Graceful Failure Index

The notebook introduces a **Graceful Failure Index (GFI)** based on the normalized area under the relative-performance curve.

Conceptually:

```text
Higher GFI
    │
    ▼
Performance retained
across the corruption range
```

A model that maintains its performance throughout the full corruption range will obtain a larger normalized area.

### Important

The GFI is a **proposed benchmark summary metric** developed for this study.

It is not presented as an established universal machine-learning metric.

---

## Benchmark Analysis

The notebook performs several complementary analyses.

### Performance Degradation Curves

The degradation curves show how model performance changes as corruption severity increases.

The starting point represents clean performance.

The shape and steepness of the curve reveal whether performance:

* Remains relatively stable
* Degrades gradually
* Drops sharply
* Collapses under severe corruption

---

### Stability Heatmap

The stability heatmap summarizes robustness across multiple corruption types.

```text
                Feature Noise
                       │
                       ▼
Model ───────────── Stability Score
 │
 ├── Missing Data
 ├── Outliers
 ├── Distribution Shift
 └── Feature Corruption
```

This makes it possible to identify models that are broadly stable versus models that are robust only to particular types of corruption.

---

### Clean Performance vs Robustness

The benchmark directly compares:

```text
X-axis → Clean F1
Y-axis → Overall Stability
```

This analysis investigates whether the strongest clean-test model is also the model that best preserves its performance under corruption.

A key principle of the benchmark is:

> **Clean predictive strength and robustness are not necessarily the same property.**

---

## Final Multi-Dimensional Summary

The final summary combines:

* Clean F1
* Clean Accuracy
* Mean Stability
* Mean Relative Degradation
* Mean GFI
* Mean Degradation Slope

The resulting table should **not** be interpreted as a universal model leaderboard.

Instead, it demonstrates that different definitions of model quality can lead to different conclusions.

---

## Output Files

The notebook exports the benchmark results as CSV files.

```text
optimized_benchmark_results.csv
optimized_stress_results.csv
optimized_stability_summary.csv
optimized_final_summary.csv
```

### `optimized_benchmark_results.csv`

Contains the detailed benchmark results for each:

```text
Dataset × Seed × Model × Corruption Type × Corruption Level
```

### `optimized_stress_results.csv`

Contains relative performance and degradation measurements for the stress experiments.

### `optimized_stability_summary.csv`

Contains aggregated:

* Stability Score
* Mean Degradation

for each model, dataset, and corruption type.

### `optimized_final_summary.csv`

Contains the final multi-dimensional model summary, including:

* Clean F1
* Clean Accuracy
* Mean Stability
* Mean Relative Degradation
* Mean GFI
* Mean Degradation Slope

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* Google Colab / Kaggle Notebooks

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your_username>/ML_Model_Showdown-Performance_Stability_Efficiency.git
```

Navigate to the project:

```bash
cd ML_Model_Showdown-Performance_Stability_Efficiency
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open the benchmark notebook.

The notebook can also be executed directly through Kaggle or Google Colab.

For a quick execution:

```python
FAST_MODE = True
```

For an extended benchmark:

```python
FAST_MODE = False
```

---

## Project Structure

```text
ML_Model_Showdown-Performance_Stability_Efficiency/
│
├── When_Accuracy_Lies_A_Multi-Dimensional_ML_Benchmark.ipynb
│
├── optimized_benchmark_results.csv
├── optimized_stress_results.csv
├── optimized_stability_summary.csv
├── optimized_final_summary.csv
│
├── requirements.txt
└── README.md
```

---

## Key Insight

The central idea of this project is:

```text
Performance ≠ Robustness
```

A model can achieve excellent performance on clean test data while experiencing significant degradation when the data distribution or quality changes.

Conversely, a model with slightly lower clean performance may preserve its predictive capability more consistently under adverse conditions.

Therefore:

```text
Clean Performance
        +
Robustness
        +
Stability
        +
Graceful Failure
        ↓
More Complete Model Evaluation
```

The benchmark shifts the focus from:

> **"Which model wins?"**

to:

> **"How does the model behave when the world stops looking like the training data?"**

---

## Research Findings to Investigate

The notebook is designed to investigate whether:

1. Clean-test performance does not always predict robustness.
2. Different corruption mechanisms produce different degradation profiles.
3. Some models remain stable under one stress test but degrade sharply under another.
4. Model rankings change as corruption severity increases.
5. Robustness varies across datasets.
6. A single clean-test metric provides an incomplete picture of model reliability.

Only findings supported by the actual benchmark results should be reported as empirical conclusions.

---

## Failure Analysis

Aggregate metrics provide an overall picture, but individual degradation patterns can reveal more detailed behaviour.

Important cases to investigate include:

* High clean F1 with steep degradation
* Moderate clean F1 with strong stability
* Robustness to missing data but weakness under distribution shift
* Strong performance under feature noise but poor performance under feature corruption
* Sharp performance loss after a particular corruption threshold

Potential explanations may involve:

* Feature scaling
* Dimensionality
* Model complexity
* Regularization
* Nonlinear relationships
* Sensitivity to particular features

These explanations should be treated as interpretations unless additional experiments establish causality.

---

## Limitations

This benchmark is intended as an **exploratory robustness study**, rather than a definitive ranking of machine-learning algorithms.

Current limitations include:

1. The benchmark uses only three relatively small tabular datasets.
2. Stress tests are controlled simulations rather than naturally occurring production failures.
3. Models use default or lightly configured hyperparameters.
4. Stability Score and GFI are proposed summary measures rather than established universal standards.
5. Training and evaluation are dependent on the execution environment.
6. The benchmark does not currently evaluate calibration.
7. Fairness is not evaluated.
8. Adversarial perturbations are not included.
9. Feature-importance stability is not evaluated.
10. A larger benchmark would benefit from repeated cross-validation and more real-world datasets.

---

## Future Improvements

### Additional Models

* XGBoost
* LightGBM
* K-Nearest Neighbors
* Naive Bayes
* Extra Trees
* CatBoost

### Additional Stress Tests

* Class imbalance
* Feature dropout
* Temporal drift
* Selected-feature covariate shift
* Adversarial perturbations
* Concept drift

### Additional Reliability Measures

* Brier Score
* Expected Calibration Error
* Prediction confidence
* Feature-importance stability
* Prediction entropy

### Stronger Statistical Design

* More random seeds
* Repeated stratified cross-validation
* Confidence intervals
* Friedman statistical test
* Post-hoc comparisons

### Future Benchmark

A natural extension of this work would be:

> **Can We Trust Feature Importance? An Explainability Stability Benchmark**

---

## Conclusion

Traditional model comparison usually asks:

> **Which model achieves the highest score on clean test data?**

This benchmark broadens that evaluation.

A reliable machine-learning model should not only perform well when data is clean. It should also behave predictably when:

* Measurements become noisy
* Values disappear
* Outliers occur
* Labels contain errors
* Feature relationships are disrupted
* Deployment distributions change

The core principle is:

> **Accuracy describes how well a model performs under a particular test condition. Robustness describes how its behaviour changes when those conditions stop being ideal.**

Therefore, a more complete model evaluation should consider both **predictive performance and degradation behaviour**.

---

## References

* Scikit-learn — Machine Learning in Python
* Scikit-learn — Classification Metrics
* Scikit-learn — Model Selection and Evaluation
* Scikit-learn — Dataset Utilities
* Scikit-learn — Pipeline and Preprocessing

The Kaggle notebook contains the detailed experimental methodology, implementation, mathematical formulation, visualizations, and benchmark results.

---

## License

This project is licensed under the MIT License.

---

I am open to collaborations, suggestions, and recommendations for extending this machine learning robustness benchmarking project.
