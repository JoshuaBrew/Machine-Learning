# Applied Machine Learning & Neural Networks Repository

A comprehensive collection of applied machine learning, statistical modeling, and deep learning projects completed across undergraduate coursework in Machine Learning and Advanced Applied Data Science (ISAT 341 & ISAT 449).

This repository demonstrates a complete learning curve in predictive modeling: beginning with first-principles mathematical algorithm design in pure NumPy, advancing through supervised classification and regression pipelines in Scikit-Learn, and culminating in multi-layer neural network architectures and multi-modal deep learning applications (Computer Vision & Audio Processing).

---

## 📌 Executive Summary

Designed to showcase end-to-end technical competencies for data science, data engineering, and machine learning roles, this repository highlights a complete progression across classical statistical modeling, custom algorithm development, and modern multi-modal AI architectures.

### Key Highlights for Hiring Teams
* **First-Principles AI Engineering:** Built custom object-oriented estimators (`Perceptron`, `AdalineGD`) using pure NumPy vectorization and batch gradient descent without reliance on high-level ML frameworks.
* **Multi-Modal Deep Learning & Feature Extraction:** Applied Scikit-Learn `MLPClassifier` architectures across Computer Vision ($28\times28$ pixel matrices) and Audio AI (extracting acoustic features like MFCCs, Chroma, and Mel Spectrograms via `librosa`).
* **Pipeline Optimization & Diagnostics:** Engineered robust data processing workflows incorporating feature scaling (`StandardScaler`), hyperparameter tuning, loss curve convergence diagnostics, and model serialization (`.pkl`) for downstream deployment.
* **Rigorous Model Evaluation:** Implemented industry-standard validation metrics—including precision-recall trade-offs, F1-scores, multi-class confusion matrices, and $Z$-statistic hypothesis testing to benchmark custom algorithms against enterprise frameworks.

---

## 🛠️ Tech Stack & Tooling

* **Languages:** Python (3.x)
* **Core ML & Deep Learning:** Scikit-Learn, SciPy, TensorFlow / Keras
* **Data Manipulation & Linear Algebra:** NumPy, Pandas
* **Signal Processing & Audio AI:** Librosa, Soundfile
* **Data Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Pickle (`.pkl`)
* **Environment:** Jupyter Notebooks, Git

---

## 📂 Repository Directory Structure

```text
Machine-Learning/
├── perceptron-adaline-ml/                          # First-Principles Algorithm Engineering (Pure NumPy)
│   ├── Activity10/                                 # Custom Python Class for Perceptron Model
│   ├── Final Exam-PartII-Adaline/                  # Custom Adaline Engine & Capstone Benchmark
│   ├── Lab 0/                                      # Baseline Scikit-Learn Perceptron Training
│   ├── Lab 7/                                      # Loss Curves & Gradient Descent Optimization
│   ├── Lab8-/                                      # McCulloch-Pitts (MCP) Logic Gates (AND/OR)
│   └── classifiers/pkl_objects/                    # Serialized Model Binary Estimators (.pkl)
│
├── statistical-machine-learning/                  # Classical Supervised Machine Learning & Evaluation
│   ├── Assignment 8/                               # Model Evaluation Metrics & Confusion Matrices
│   ├── Exercise 6/                                 # Multi-Class Logistic Regression Homework
│   ├── Lab0_JupyterNotebook/                       # Environment Setup & Notebook Workflows
│   ├── Lab1-Data Modeling/                         # Data Preprocessing & Curve Fitting
│   ├── Lab2-Multiple Variable.../                  # Linear Regression on Synthetic Data
│   ├── Lab3-Multiple Variable.../                  # Linear Regression on Real-World Data
│   └── Lecture 7/                                  # Binary & Multi-Class Logistic Regression
│
├── supervised-learning-foundations/                # Supervised Learning Architectures
│   ├── Lab10-Image Recognition/                    # Digits Classification via Scikit-Learn
│   ├── Lab4- K-Nearest Neighbors/                  # Non-parametric k-NN Classification & Tuning
│   └── Lab5-Classification/                        # Cross-Validation & Multi-Class Metrics
│
└── scikit-learn-mlp-classifier/                    # Neural Networks, Computer Vision & Audio AI
    ├── audio-classification/                       # Speech Emotion Recognition (SER) via Librosa
    └── image-classification/                       # Multi-Layer Perceptrons for Digits & Fashion MNIST
```

## 📊 Foundational Machine Learning Concepts Demonstrated

* **Mathematical Foundations:** Vectorized matrix multiplication ($W^T X + b$), gradient descent weight updates, and loss curve diagnostic monitoring.
* **Data Preprocessing & Scaling:** Feature normalization (`MinMaxScaler` vs. `StandardScaler`), array flattening, and handling class distribution imbalances to prevent algorithmic bias.
* **Model Validation:** Evaluating performance using confusion matrices, precision-recall trade-offs, F1-scores, and two-sample statistical hypothesis testing ($Z$-scores).
* **Production Readiness:** Serializing trained model estimators into binary `.pkl` files for local deployment and inference without retraining.

---

## 🏆 Capstone Showcase: Custom Adaline Engine vs. Optimized Estimators

The repository culminates in a capstone analysis (`perceptron-adaline-ml/Final Exam-PartII-Adaline/`) evaluating the trade-offs between custom-built mathematical models and production-optimized ML libraries using the MNIST Digits dataset.

* **Custom Classifier Architecture (`AdalineGD`):** Implemented an object-oriented Python class featuring native `.fit()`, `.predict()`, and `.score()` methods matching Scikit-Learn’s API design patterns. Weight updates were driven by vectorized batch gradient descent:
  $$\Delta w = \eta \cdot X^T (y - \text{output})$$
* **Preprocessing Impact:** Demonstrated how feature standardization directly impacts gradient stability, reducing test-set misclassifications from **798 down to 164**.
* **Comparative Hypothesis Testing:** Conducted statistical hypothesis testing ($Z$-statistic error-rate analysis) comparing the custom `AdalineGD` model (**95.6% accuracy**) against Scikit-Learn’s optimized Perceptron estimator (**99.8% accuracy**).

---

## 🚀 Getting Started Locally

### Prerequisites
* Python 3.8+
* Jupyter Notebook or JupyterLab

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/JoshuaBrew/Machine-Learning.git](https://github.com/JoshuaBrew/Machine-Learning.git)
   cd Machine-Learning
