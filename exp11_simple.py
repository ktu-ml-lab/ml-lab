# Exp 11: Linear SVM on Iris - Setosa vs Non-Setosa (simple version)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# 1. Load data (sklearn)
iris = load_iris()
X = iris.data[:, :2]                      # sepal length, sepal width
y = iris.target                           # 0 = setosa, 1, 2 = others

# 1. Load data (CSV) - use instead if needed
# df = pd.read_csv("iris.csv")
# X = df.iloc[:, :2].values
# y = df['target'].values

# 2. Preprocessing
print("Shape:", X.shape)
y = np.where(y == 0, 1, -1)               # setosa = +1, others = -1

# 3. Train Linear SVM
model = SVC(kernel='linear', C=1)
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]
print("w =", w, " b =", b)
print("Margin =", 2 / np.linalg.norm(w))
print("Support vectors:", len(model.support_vectors_))
print("Accuracy:", model.score(X, y))

# 4. Plot boundary, margins, support vectors
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.plot(x1, -(w[0] * x1 + b) / w[1], 'k-')          # boundary  w.x + b = 0
plt.plot(x1, -(w[0] * x1 + b - 1) / w[1], 'k--')     # margin    w.x + b = +1
plt.plot(x1, -(w[0] * x1 + b + 1) / w[1], 'k--')     # margin    w.x + b = -1
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='green')
plt.ylim(1.5, 5)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Linear SVM")
plt.show()

# 5. Effect of C on margin
for c in [0.01, 1, 100]:
    m = SVC(kernel='linear', C=c).fit(X, y)
    print("C =", c, " Margin =", round(2 / np.linalg.norm(m.coef_[0]), 3),
          " Support vectors =", len(m.support_vectors_), " Accuracy =", round(m.score(X, y), 3))
