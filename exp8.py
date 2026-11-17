# Exp 8: KNN image classification on Fashion MNIST (10 classes, 28x28 images)
# Try different K values, compare accuracy and time
# Plots: sample images, K vs accuracy

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from TensorFlow / Keras (ONLY if TensorFlow is installed; needs internet the first time)
# from tensorflow.keras.datasets import fashion_mnist
# (X, y), _ = fashion_mnist.load_data()
# X = X.reshape(len(X), -1)        # flatten 28x28 -> 784

# OPTION B: from CSV file (header: label, pixel1 ... pixel784)
df = pd.read_csv("fashion-mnist_train.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

# OPTION C: from .data file (no header, first column is TEXT class name, then 784 pixels)
# df = pd.read_csv("fashion.data", header=None)
# X = df.iloc[:, 1:].values                                   # all columns except the first
# y = df[0].map({name: i for i, name in enumerate(classes)}).values   # 'Sneaker' -> 7

print("Shape:", X.shape)

# ---------------- STEP 2: PREPROCESS ----------------
X = X / 255.0                         # pixels 0-255 -> 0-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X_train[:10000], y_train[:10000]   # subset for speed
X_test, y_test = X_test[:2000], y_test[:2000]

# show 5 sample images
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(classes[y_test[i]], fontsize=8)
    plt.axis('off')
plt.show()

# ---------------- STEP 3: KNN WITH DIFFERENT K ----------------
k_values = [1, 3, 5, 7, 9]
accuracies = []
for k in k_values:
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)    # Euclidean distance by default
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k}  Accuracy={acc:.4f}  Time={time.time() - start:.2f}s")

best_k = k_values[np.argmax(accuracies)]
print("\nBest K:", best_k)
knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
print(classification_report(y_test, knn.predict(X_test), labels=range(10), target_names=classes))

# ---------------- STEP 4: PLOT ----------------
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Effect of K on KNN Accuracy")
plt.show()
