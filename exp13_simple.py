# Exp 13: MLP on Wine Quality (simple version, no TensorFlow)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_csv("WineQT.csv")
X = df.drop(['quality', 'Id'], axis=1)    # 11 chemical features
y = df['quality']                         # quality 3 to 8

# 2. Preprocessing
print("Missing values:", df.isnull().sum().sum())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # mean 0, std 1
X_test = scaler.transform(X_test)

# 3. Train and test different architectures
architectures = [(8,), (32, 16), (64, 32, 16)]
acc_list = []
for layers in architectures:
    mlp = MLPClassifier(hidden_layer_sizes=layers, activation='relu', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    acc_list.append(acc)
    print("Hidden layers:", layers, " Accuracy:", round(acc, 4))

# 4. Plot
plt.bar(['(8)', '(32,16)', '(64,32,16)'], acc_list)
plt.xlabel("Hidden layers")
plt.ylabel("Accuracy")
plt.title("MLP Architecture Comparison")
plt.show()
