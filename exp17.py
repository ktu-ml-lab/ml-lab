# Exp 17: K-Means clustering on Digits dataset (8x8 images = 64 features)
# Try K = 2..14, evaluate with Inertia (elbow) and Silhouette score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn (built in, 1797 images)
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data

# OPTION B: from CSV file (header, 64 pixel columns + target)
# df = pd.read_csv("digits.csv")
# X = df.drop('target', axis=1).values     # labels NOT used: clustering is unsupervised

# OPTION C: from .data file (original UCI optdigits, no header, last column = digit, 3823 images)
# df = pd.read_csv("digits.data", header=None)
# X = df.iloc[:, :-1].values                # drop last column (the digit label)

print("Shape:", X.shape)

# ---------------- STEP 2: SCALE ----------------
X = StandardScaler().fit_transform(X)

# ---------------- STEP 3: K-MEANS FOR DIFFERENT K ----------------
K_range = range(2, 15)
inertia, silhouette = [], []
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)                    # WCSS
    silhouette.append(silhouette_score(X, labels))
    print(f"K={k}  Inertia={km.inertia_:.2f}  Silhouette={silhouette[-1]:.4f}")

print("\nBest K by silhouette:", K_range[np.argmax(silhouette)])

# ---------------- STEP 4: PLOTS ----------------
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of clusters K")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.show()

plt.plot(K_range, silhouette, marker='o', color='green')
plt.xlabel("Number of clusters K")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs K")
plt.show()
