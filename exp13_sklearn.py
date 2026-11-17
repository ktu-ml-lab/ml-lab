# Exp 13 (BACKUP - use if TensorFlow is NOT installed in the lab)
# Same experiment using scikit-learn's MLPClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("WineQT.csv").drop('Id', axis=1)
X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = [(8,), (32, 16), (64, 32, 16)]     # tuple = neurons in each hidden layer
results = []
for layers in architectures:
    mlp = MLPClassifier(hidden_layer_sizes=layers, activation='relu',
                        max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    results.append(acc)
    print("Hidden layers", layers, " Accuracy:", round(acc, 4))

plt.plot(mlp.loss_curve_)                          # loss curve of last model
plt.title("Loss curve (64,32,16)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

plt.bar([str(a) for a in architectures], results)
plt.ylabel("Accuracy")
plt.title("MLP Architecture Comparison")
plt.show()
