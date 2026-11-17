# Exp 13: MLP (Multilayer Feed-Forward Neural Network) on Wine Quality (Keras)
# Try different hidden layers / neurons and compare accuracy
# Plots: loss curve of one model, accuracy bar chart
# OPTIONAL Keras version - needs TensorFlow. The main exp13.py needs only scikit-learn.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn openml (needs internet). NOTE: load_wine() is a DIFFERENT dataset!
# from sklearn.datasets import fetch_openml
# data = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='liac-arff').frame
# data['quality'] = data['class'].astype(int)
# data = data.drop('class', axis=1)

# OPTION B: from CSV file (WineQT.csv from Kaggle, has an Id column)
data = pd.read_csv("WineQT.csv")
data = data.drop('Id', axis=1)           # Id is just a row number, not a feature

# OPTION C: from .data file (original UCI file, semicolon separated, header, no Id)
# data = pd.read_csv("winequality.data", sep=';')

print(data.head())
X = data.drop('quality', axis=1).values  # 11 chemical features
y = data['quality'].values               # quality score (3..8)

# ---------------- STEP 2: PREPROCESS ----------------
print("Missing values:", data.isnull().sum().sum())
y = LabelEncoder().fit_transform(y)      # 3,4,5,6,7,8 -> 0,1,2,3,4,5
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit only on train
X_test = scaler.transform(X_test)

# ---------------- STEP 3: BUILD + TRAIN DIFFERENT ARCHITECTURES ----------------
architectures = {"[8]": [8], "[32,16]": [32, 16], "[64,32,16]": [64, 32, 16]}
results = {}

for name, layers in architectures.items():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))          # hidden layers
    model.add(Dense(n_classes, activation='softmax'))         # output layer

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',     # labels are 0..5 (not one-hot)
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = acc
    print(f"Hidden layers {name}:  Test accuracy = {acc:.4f}  Loss = {loss:.4f}")

    if name == "[32,16]":                # save curve of the medium model to plot
        curve = history.history

# ---------------- STEP 4: PLOTS ----------------
plt.plot(curve['loss'], label='train loss')
plt.plot(curve['val_loss'], label='validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve - [32,16] model")
plt.legend()
plt.show()

plt.bar(list(results.keys()), list(results.values()))
plt.xlabel("Hidden layers")
plt.ylabel("Test accuracy")
plt.title("MLP Architecture Comparison")
plt.show()
