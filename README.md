# Machine Learning Labs

Lab work from my final-year Machine Learning module (BSc Computer Science, University of Roehampton). Each lab is a self-contained Jupyter notebook with its data, results and a short write-up of the findings.

## Labs

| Lab | Topic | Techniques | Headline result |
|---|---|---|---|
| [Lab 2](lab-02-overfitting-and-lasso/) | Overfitting and regularisation | Polynomial regression, train/test split, Lasso (L1) regression | Degree 6 polynomial generalises best; Lasso predicts daily bike rentals with R² 0.84 |
| [Lab 3](lab-03-decision-trees-random-forests/) | Tree-based classification | Decision trees, random forests, feature importance, leave-one-out cross-validation | Random forest detects forged banknotes with 99.3% accuracy |
| [Lab 4](lab-04-classifier-evaluation/) | Classifier evaluation | Accuracy, confusion matrix, precision, recall, F1/F2, ROC-AUC, permutation importance | Random forest: precision 1.00, recall 0.98, AUC 0.9999 |
| [Lab 5](lab-05-mnist-neural-network/) | Neural networks | Keras, tensors, Dense + ReLU + Softmax, one-hot encoding, confusion matrix | Recognises handwritten digits with 97% test accuracy |
| [Lab 6](https://github.com/julianna162/florida-temperature-lstm) | Time-series forecasting | Stacked LSTM, sliding windows, MinMax scaling | Forecasts Florida monthly temperature with MAE 1.24 °C (separate repo) |

## Lab summaries

### Lab 2: Overfitting and Lasso regression
Fits polynomials of degree 1 to 8 to noisy synthetic data and plots training vs test R². Training fit keeps improving with degree, but test fit peaks at degree 6 and then collapses, which shows overfitting clearly. The second half uses Lasso regression on a bike-sharing dataset. Lasso zeroes 6 of 26 features while matching ordinary linear regression's accuracy.

### Lab 3: Decision trees and random forests
Classifies banknotes as genuine or forged from four wavelet-image features. A decision tree reaches 98.2% accuracy and a random forest 99.3%, with image variance the most important feature. A second, tiny hiring dataset (13 candidates) shows why a 3-row test set is unreliable and uses leave-one-out cross-validation instead. It also visualises the learned hiring rules.

### Lab 4: Evaluating a classifier
Applies the standard classification metrics to the Lab 3 banknote models: accuracy (with a check that the classes are balanced and a dummy baseline), confusion matrices, precision, recall, F1 and recall-weighted F2, and ROC curves with AUC. Both models have perfect precision, so the remaining errors are missed forgeries. Permutation importance on the test set confirms Variance as the key feature and shows Entropy adds almost nothing.

### Lab 5: MNIST neural network
Builds a network from scratch in Keras: 28 × 28 image → Flatten (784) → Dense 64 (ReLU) → Dense 10 (Softmax). It covers tensors, scaling, one-hot labels, training curves, a confusion matrix and the most common mistakes (4 misread as 9), and visualises the 64-unit hidden layer. The learning curves show mild overfitting after about epoch 5.

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/julianna162/machine-learning-labs.git
cd machine-learning-labs
pip install -r requirements.txt
jupyter lab
```

Open any lab folder and run the notebook top to bottom (**Kernel > Restart Kernel and Run All Cells**). Each folder contains the CSV data its notebook needs. Lab 5 downloads the MNIST dataset automatically the first time it runs (about 11 MB).

## Data sources

- **Lab 2:** synthetic data generated in the notebook; UCI Bike Sharing dataset (daily, pre-encoded) supplied with the module.
- **Lab 3:** UCI Banknote Authentication dataset; small example hiring dataset supplied with the module.
- **Lab 5:** MNIST handwritten digits, loaded via `keras.datasets`.
