# Exp 8: KNN on Fashion MNIST (simple version)
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load data (CSV)
df = pd.read_csv("fashion-mnist_train.csv")
X = df.drop('label', axis=1)
y = df['label']

# 1. Load data (Keras) - use instead of CSV if needed
# from tensorflow.keras.datasets import fashion_mnist
# (X, y), _ = fashion_mnist.load_data()
# X = X.reshape(len(X), 784)            # flatten 28x28 -> 784

# 2. Preprocessing
print("Missing values:", pd.isnull(X).sum().sum())   # check for missing values
X = X / 255.0                            # normalize pixels 0-255 -> 0-1

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X_train[:10000], y_train[:10000]
X_test, y_test = X_test[:2000], y_test[:2000]

# 4. Try different K
k_values = [1, 3, 5, 7, 9]
acc_list = []
for k in k_values:
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    print("K =", k, " Accuracy =", acc, " Time =", round(time.time() - start, 2), "s")

# 5. Plot
plt.plot(k_values, acc_list, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.show()
