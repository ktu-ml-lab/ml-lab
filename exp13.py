# Exp 13: MLP (Multilayer Feed-Forward Neural Network) on Wine Quality
# Uses scikit-learn's MLPClassifier (no TensorFlow needed)
# Try different hidden layers / neurons and compare accuracy
# Plots: loss curve of one model, accuracy bar chart

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn openml (needs internet). NOTE: load_wine() is a DIFFERENT dataset!
# from sklearn.datasets import fetch_openml
# data = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='liac-arff').frame
# data['quality'] = data['class'].astype(int)
# data = data.drop('class', axis=1)

# OPTION B: from CSV file (WineQT.csv from Kaggle, has an Id column)
data = pd.read_csv("WineQT.csv")
data = data.drop('Id', axis=1)           # Id is just a row number, not a feature

# OPTION C: from .data file (original UCI file, semicolon separated, header, no Id)
# data = pd.read_csv("winequality.data", sep=';')

print(data.head())
X = data.drop('quality', axis=1).values  # 11 chemical features
y = data['quality'].values               # quality score (3..8)

# ---------------- STEP 2: PREPROCESS ----------------
print("Missing values:", data.isnull().sum().sum())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit only on train
X_test = scaler.transform(X_test)

# ---------------- STEP 3: TRAIN DIFFERENT ARCHITECTURES ----------------
architectures = {"[8]": (8,), "[32,16]": (32, 16), "[64,32,16]": (64, 32, 16)}
results = {}

for name, layers in architectures.items():
    mlp = MLPClassifier(hidden_layer_sizes=layers,    # neurons in each hidden layer
                        activation='relu',            # hidden layer activation
                        solver='adam',                # optimizer
                        max_iter=1000,                # max epochs
                        random_state=42)
    mlp.fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    results[name] = acc
    print(f"Hidden layers {name}:  Test accuracy = {acc:.4f}  Epochs = {mlp.n_iter_}")

    if name == "[32,16]":                # save loss curve of the medium model
        curve = mlp.loss_curve_

# ---------------- STEP 4: PLOTS ----------------
plt.plot(curve)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve - [32,16] model")
plt.show()

plt.bar(list(results.keys()), list(results.values()))
plt.xlabel("Hidden layers")
plt.ylabel("Test accuracy")
plt.title("MLP Architecture Comparison")
plt.show()
