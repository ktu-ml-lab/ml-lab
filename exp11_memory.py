import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
X = iris.data[:, :2]
y = np.where(iris.target == 0, 1, -1)

model = SVC(kernel='linear', C=100)
model.fit(X, y)
w = model.coef_[0]
b = model.intercept_[0]
print("Margin =", 2 / np.linalg.norm(w))

plt.scatter(X[:, 0], X[:, 1], c=y)
x = np.linspace(4, 8, 100)
for d in [0, 1, -1]:
    plt.plot(x, -(w[0] * x + b - d) / w[1], 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='r')
plt.show()
