# Exp 11: Linear SVM on Iris - Setosa vs Non-Setosa (binary)
# Visualize decision boundary, margins and support vectors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn (built in)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]              # only 2 features: sepal length, sepal width (to plot in 2D)
y = iris.target                   # 0 = setosa, 1 = versicolor, 2 = virginica

# OPTION B: from CSV file (header, target column is a number)
# df = pd.read_csv("iris.csv")
# X = df.iloc[:, :2].values       # first 2 columns
# y = df['target'].values

# OPTION C: from .data file (no header, species is TEXT like 'Iris-setosa')
# df = pd.read_csv("iris.data", header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
# X = df.iloc[:, :2].values
# y = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).values

print("Shape:", X.shape)

# ---------------- STEP 2: MAKE IT BINARY ----------------
y = np.where(y == 0, 1, -1)       # Setosa = +1, others = -1

# ---------------- STEP 3: TRAIN LINEAR SVM ----------------
model = SVC(kernel='linear', C=1)
model.fit(X, y)

w = model.coef_[0]                # weights  [w1, w2]
b = model.intercept_[0]           # bias
print("w =", w, " b =", b)
print("Margin width = 2/||w|| =", 2 / np.linalg.norm(w))
print("Number of support vectors:", len(model.support_vectors_))
print("Training accuracy:", model.score(X, y))

# ---------------- STEP 4: PLOT ----------------
# boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2
# margins : w.x + b = +1 and -1
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.plot(x1, -(w[0] * x1 + b) / w[1], 'k-', label="Decision boundary")
plt.plot(x1, -(w[0] * x1 + b - 1) / w[1], 'k--', label="Margin")
plt.plot(x1, -(w[0] * x1 + b + 1) / w[1], 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='green', label="Support vectors")
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Linear SVM: Setosa vs Non-Setosa")
plt.legend()
plt.show()
