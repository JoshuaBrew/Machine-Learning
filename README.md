# Machine-Learning Projects
[![GitHub Repo](https://img.shields.io/badge/GitHub-JoshuaBrew%2FMachine--Learning-blue?logo=github)](https://github.com/JoshuaBrew/Machine-Learning)

A collection of applied machine learning, exploratory data analysis, and statistical modeling projects completed as part of my undergraduate coursework in Emerging Topics in Applied Data Science (ISAT 449). 

This repository showcases end-to-end workflows using Python's core scientific stack—moving from raw data ingestion and feature normalization to model architecture design, hyperparameter tuning, and performance evaluation.

---

## 🛠️ Tech Stack & Tooling

* **Languages:** Python (3.x)
* **Machine Learning Frameworks:** SciKit-Learn, TensorFlow / Keras
* **Data Processing & Analysis:** NumPy, Pandas, SciPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment & Tools:** Jupyter Notebooks, Git

---

## 🔬 Featured Projects

### Multilayer Perceptron (MLP) Image Classification
* **Core Topics:** Artificial Neural Networks, Optimization Algorithms, Loss Curves, Algorithmic Bias
* **Overview:** 
  Developed a dense Multilayer Perceptron (MLP) classifier using `scikit-learn` to recognize handwritten digits ($8 \times 8$ pixel array representations) from the Scikit-Learn Digits dataset.
* **Key Highlights:**
  * **Preprocessing:** Normalized feature intensity matrices ($0–16 \to 0.0–1.0$) to improve optimization convergence.
  * **Model Architecture:** Designed a 3-hidden-layer deep architecture (256, 128, 32 neurons) and analyzed convergence behavior across `Adam` and `Stochastic Gradient Descent (SGD)` solvers.
  * **Evaluation & Diagnostics:** Evaluated convergence via training loss curves, precision/recall metrics, and confusion matrices to diagnose class-level misclassifications (e.g., distinguishing digit 8 variants).

---

## 📊 Core Concepts & Methodologies Demonstrated

Across this repository, the code and analysis focus on the following foundational workflows:

1. **Dataset Integrity & Class Balance:** Testing target distribution symmetry before model training to mitigate algorithmic bias and improper metric evaluation.
2. **Feature Engineering & Normalization:** Scaling continuous features and reshaping multidimensional arrays (e.g., image vectorization) for linear and non-linear estimators.
3. **Hyperparameter Optimization:** Systematically tuning learning rate schedules, convergence tolerances, maximum iterations, and solver algorithms to balance model accuracy with computational efficiency.
4. **Comprehensive Model Evaluation:** Moving beyond overall raw accuracy to analyze Precision, Recall, F1-Score distributions, class membership probability vectors (`predict_proba`), and error matrices.

---

## 🚀 Getting Started Locally

### Prerequisites
* Python 3.8+
* Jupyter Notebook or JupyterLab

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/JoshuaBrew/Machine-Learning.git](https://github.com/JoshuaBrew/Machine-Learning.git)
   cd Machine-Learning
