# Machine-Learning Projects
[![GitHub Repo](https://img.shields.io/badge/GitHub-JoshuaBrew%2FMachine--Learning-blue?logo=github)](https://github.com/JoshuaBrew/Machine-Learning)

A comprehensive collection of applied machine learning, statistical modeling, and deep learning projects completed across undergraduate coursework in Machine Learning and Advanced Applied Data Science.

This repository demonstrates a complete learning curve in predictive modeling: beginning with **first-principles mathematical algorithm design in pure NumPy**, advancing through **supervised classification and regression pipelines in Scikit-Learn**, and culminating in **multi-layer neural network architectures and deep learning applications using TensorFlow/Keras**.

---

## 🛠️ Tech Stack & Tooling

* **Languages:** Python (3.x)
* **Core ML & Deep Learning:** SciKit-Learn, TensorFlow / Keras, SciPy
* **Data Manipulation & Linear Algebra:** NumPy, Pandas
* **Signal Processing & Audio AI:** Librosa, Soundfile
* **Data Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Pickle (`.pkl`)
* **Environment:** Jupyter Notebooks, Git

---

## 🔬 Learning Progression & Directory Structure

### 1. First-Principles Algorithm Engineering (Pure NumPy)
* **`Activity10 (Pre-Project Activity)-Designing the Python Class for the Perceptron Model/`**: Object-oriented Perceptron classifier engineered from scratch using pure Python and NumPy.
* **`Lab8- MCP Perceptron for AND and OR Gates in Python Padilla/`**: McCulloch-Pitts (MCP) logic gate implementations demonstrating linear decision boundary logic for `AND` and `OR` functions.
* **`Final Exam-PartII-Adaline/`**: Custom Adaptive Linear Neuron engine implementing batch Gradient Descent to minimize Sum-of-Squares Error (SSE) loss functions directly via matrix calculus.

### 2. Classical Supervised Learning & Pipeline Design (Scikit-Learn)
* **`Lab0_JupyterNotebook/` & `Lab 0- Training a Perceptron Model using Scitkit-Learn/`**: Environment setup and baseline classification workflows using Scikit-Learn estimators.
* **`Lab1-Data Modeling and Fitting with SciKitLearn/`**: Data preprocessing, curve fitting, and baseline modeling pipelines.
* **`Lab2-Multiple Variable Linear Regression-Generated Data/` & `Lab 2 -ScitKit Learn/`**: Linear regression modeling across synthetic feature distributions.
* **`Lab3-Multiple Variable Linear Regression-Imported Data/` & `Lab 3/`**: Multi-variable continuous parameter fitting on real-world datasets.
* **`Lab4- K-Nearest Neighbors Classification with Imported Data/` & `Lab 4/`**: Non-parametric classification and hyperparameter tuning using $k$-NN.
* **`Lab5-Classification of Imported Data/`**: Model evaluation, cross-validation metrics, and multi-class classification workflows.
* **`Lecture 7 - Logistic Regression on Iris Subset.ipynb` & `Exercise 6 - Logistic Regression (Homework).ipynb`**: Parametric binary and multi-class logistic regression modeling, decision boundary plotting, and probability vector analysis (`predict_proba`).

### 3. Neural Networks, Computer Vision & Audio AI (TensorFlow / Keras / Librosa)
* **`Lab10-Image Recognition with SciKitLearn(digits dataset)-(Lecture)Sp23/`**: Handwritten digit classification using $8 \times 8$ pixel array vectorization and Multilayer Perceptron (MLP) architectures.
* **`Lab 7/` & `Assignment 8/`**: Deep learning network training, cost curve analysis, and convergence diagnostics comparing `Adam` vs. `SGD` optimizers.
* **`Speech-Emotion-Recognition.ipynb`**: Acoustic signal processing pipeline extracting Mel-Frequency Cepstral Coefficients (MFCCs), Chroma, and Mel Spectrograms to train emotion classification networks on unstructured audio data.

### 4. Capstone Showcase & Serialized Assets
* **`Project/`**: Primary repository showcase consolidating end-to-end machine learning workflows.
* **`classifiers/pkl_objects/`**: Serialized model estimators (`.pkl` binary files) for local deployment and instant inference without retraining.

---

## 📊 Foundational Machine Learning Concepts Demonstrated

* **Mathematical Foundations:** Vectorized matrix multiplication ($\mathbf{W}^T \mathbf{X} + b$), gradient descent weight updates, and loss curve diagnostic monitoring.
* **Data Preprocessing & Scaling:** Feature normalization (`MinMaxScaler` vs. `StandardScaler`), array flattening, and handling class distribution imbalances to prevent algorithmic bias.
* **Model Validation:** Evaluating performance using confusion matrices, precision-recall trade-offs, F1-scores, and two-sample statistical hypothesis testing ($\mathcal{Z}$-scores).
* **Production Readiness:** Serializing trained model estimators into binary `.pkl` files for local deployment and inference without retraining.

---

## 🏆 Capstone Showcase: Custom Adaline Engine vs. Optimized Estimators

The repository culminates in a capstone project (`Final Exam-PartII-Adaline/`) evaluating the trade-offs between custom-built mathematical models and production-optimized ML libraries using the MNIST Digits dataset.

* **Custom Classifier Architecture (`AdalineGD`):** Implemented an object-oriented Python class featuring native `.fit()`, `.predict()`, and `.score()` methods matching Scikit-Learn’s API design patterns. Weight updates were driven by vectorized batch gradient descent:

  $$\Delta w = \eta \cdot X^T (y - \text{output})$$

* **Preprocessing Impact:** Demonstrated how feature standardization directly impacts gradient stability, reducing test-set misclassifications from **798 down to 164**.
* **Comparative Hypothesis Testing:** Conducted statistical hypothesis testing ($\mathcal{Z}$-statistic error-rate analysis) comparing the custom `AdalineGD` model ($95.6\%$ accuracy) against Scikit-Learn’s optimized `Perceptron` estimator ($99.8\%$ accuracy).

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
