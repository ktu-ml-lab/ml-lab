# Exp 17: K-Means on Digits dataset (simple version)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load data (sklearn)
digits = load_digits()
X = digits.data                           # 1797 images, 64 pixels each

# 1. Load data (CSV) - use instead if needed
# df = pd.read_csv("digits.csv")
# X = df.drop('target', axis=1)

# 2. Preprocessing
print("Shape:", X.shape)
X = StandardScaler().fit_transform(X)     # mean 0, std 1

# 3. K-Means for different K
k_values = range(2, 15)
inertia_list = []
sil_list = []

for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    inertia_list.append(km.inertia_)
    sil_list.append(silhouette_score(X, labels))
    print("K =", k, " Inertia =", round(km.inertia_, 2), " Silhouette =", round(sil_list[-1], 4))

# 4. Plots
plt.plot(k_values, inertia_list, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

plt.plot(k_values, sil_list, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.show()
