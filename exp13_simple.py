# Exp 13: MLP on Wine Quality (simple version)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. Load data
df = pd.read_csv("WineQT.csv")
X = df.drop(['quality', 'Id'], axis=1)    # 11 chemical features
y = df['quality']                         # quality 3 to 8

# 2. Preprocessing
print("Missing values:", df.isnull().sum().sum())
y = LabelEncoder().fit_transform(y)       # 3,4,5,6,7,8 -> 0,1,2,3,4,5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # mean 0, std 1
X_test = scaler.transform(X_test)

# 3. Build, train, test different architectures
architectures = [[8], [32, 16], [64, 32, 16]]
acc_list = []

for layers in architectures:
    model = Sequential()
    model.add(Input(shape=(11,)))                     # input layer: 11 features
    for n in layers:
        model.add(Dense(n, activation='relu'))        # hidden layers
    model.add(Dense(6, activation='softmax'))         # output: 6 classes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    acc_list.append(acc)
    print("Hidden layers:", layers, " Accuracy:", round(acc, 4))

# 4. Plot
plt.bar(['[8]', '[32,16]', '[64,32,16]'], acc_list)
plt.xlabel("Hidden layers")
plt.ylabel("Accuracy")
plt.title("MLP Architecture Comparison")
plt.show()
