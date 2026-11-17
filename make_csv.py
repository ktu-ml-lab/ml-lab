# make_csv.py
# Run this ONCE (needs internet the first time for newsgroups / fashion mnist).
# It saves scikit-learn datasets as CSV files so every experiment
# can be run BOTH ways: (A) load from scikit-learn, (B) read a CSV file.

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_digits, fetch_20newsgroups

# ---------- GENERAL RULE: sklearn dataset -> CSV ----------
# data = load_xxx()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target
# df.to_csv("xxx.csv", index=False)

# Exp 3 - Diabetes
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv("diabetes.csv", index=False)
print("diabetes.csv", df.shape)

# Exp 11 - Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv("iris.csv", index=False)
print("iris.csv", df.shape)

# Exp 17 - Digits (64 pixel columns)
data = load_digits()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv("digits.csv", index=False)
print("digits.csv", df.shape)

# Exp 7 - 20 Newsgroups (3 categories, text + label)
cats = ['sci.space', 'rec.sport.baseball', 'comp.graphics']
data = fetch_20newsgroups(subset='all', categories=cats,
                          remove=('headers', 'footers', 'quotes'))
df = pd.DataFrame({'text': data.data, 'label': data.target})
df.to_csv("newsgroups.csv", index=False)
print("newsgroups.csv", df.shape)

# Exp 8 - Fashion MNIST (small CSV, same format as Kaggle: label, pixel1..pixel784)
try:
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), _ = fashion_mnist.load_data()
    X = X_train[:12000].reshape(12000, -1)
    df = pd.DataFrame(X, columns=[f"pixel{i}" for i in range(1, 785)])
    df.insert(0, 'label', y_train[:12000])
    df.to_csv("fashion-mnist_train.csv", index=False)
    print("fashion-mnist_train.csv", df.shape)
except ImportError:
    print("tensorflow not installed - skipped fashion mnist csv")
