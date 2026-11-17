# Exp 1: Simple Linear Regression (one variable) on California Housing
# Predict house price from average number of rooms
# Methods: (1) Gradient Descent  (2) Normal Equation
# Metrics: MSE, R2      Plots: fitted lines + cost curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn (downloads once, needs internet the first time)
# from sklearn.datasets import fetch_california_housing
# data = fetch_california_housing()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# X = df['AveRooms'].values               # avg rooms per house
# y = data.target * 100000                # sklearn gives price in units of 100,000 $

# OPTION B: from CSV file (has a header row)
df = pd.read_csv("housing.csv")
X = (df['total_rooms'] / df['households']).values    # avg rooms per house
y = df['median_house_value'].values

# OPTION C: from .data file (no header, comma separated, last column is TEXT)
# cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity']
# df = pd.read_csv("housing.data", header=None, names=cols)
# X = (df['total_rooms'] / df['households']).values    # text column ocean_proximity is not used
# y = df['median_house_value'].values

print(df.head())

# ---------------- STEP 2: PREPROCESS ----------------
mask = ~np.isnan(X) & ~np.isnan(y)          # remove missing values
X, y = X[mask], y[mask]
mask = X < 10                               # remove outliers (10+ avg rooms is unrealistic)
X, y = X[mask], y[mask]

X_mean, X_std = np.mean(X), np.std(X)
X = (X - X_mean) / X_std                   # normalize -> GD converges fast
n = len(X)

# ---------------- STEP 3: GRADIENT DESCENT ----------------
theta0, theta1 = 0, 0                      # intercept, slope
lr = 0.01                                  # learning rate (alpha)
iterations = 1000
cost_history = []

for i in range(iterations):
    y_pred = theta0 + theta1 * X
    error = y - y_pred
    d_theta0 = (-2 / n) * np.sum(error)        # dJ/d(theta0)
    d_theta1 = (-2 / n) * np.sum(error * X)    # dJ/d(theta1)
    theta0 = theta0 - lr * d_theta0
    theta1 = theta1 - lr * d_theta1
    cost_history.append(np.mean(error ** 2))   # cost = MSE

y_pred_gd = theta0 + theta1 * X

# ---------------- STEP 4: NORMAL EQUATION ----------------
# theta = (X^T X)^-1 X^T y
X_b = np.c_[np.ones(n), X]                 # add column of 1s for intercept
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
y_pred_ne = X_b @ theta

# ---------------- STEP 5: EVALUATE ----------------
print("\nGradient Descent : theta0 =", round(theta0, 4), " theta1 =", round(theta1, 4))
print("  MSE =", mean_squared_error(y, y_pred_gd), " R2 =", r2_score(y, y_pred_gd))
print("Normal Equation  : theta0 =", round(theta[0], 4), " theta1 =", round(theta[1], 4))
print("  MSE =", mean_squared_error(y, y_pred_ne), " R2 =", r2_score(y, y_pred_ne))

# ---------------- STEP 6: PLOTS ----------------
plt.scatter(X, y, s=5, alpha=0.3, label="Actual data")
plt.plot(X, y_pred_gd, color='red', label="Gradient Descent")
plt.plot(X, y_pred_ne, color='green', linestyle='--', label="Normal Equation")
plt.xlabel("Avg rooms (normalized)")
plt.ylabel("House value")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

plt.plot(cost_history, color='green')
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.show()

# ---------------- STEP 7: PREDICT NEW VALUE ----------------
new_rooms = 6
x_new = (new_rooms - X_mean) / X_std       # normalize with SAME mean/std
print("\nPredicted price for", new_rooms, "rooms:", theta0 + theta1 * x_new)
