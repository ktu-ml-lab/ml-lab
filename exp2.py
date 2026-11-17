# Exp 2: Polynomial Regression on Auto MPG
# Predict mpg from engine displacement; compare with Linear Regression
# Metrics: MSE, R2      Plot: linear line + polynomial curves

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn openml (needs internet; Auto MPG is not built into sklearn)
# from sklearn.datasets import fetch_openml
# df = fetch_openml(name='autoMpg', version=1, as_frame=True, parser='liac-arff').frame
# df = df.rename(columns={'class': 'mpg'})

# OPTION B: from CSV file (has a header row)
df = pd.read_csv("auto-mpg.csv", na_values='?')

# OPTION C: from .data file (spaces, no header, '?' = missing, car name is TEXT)
# cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
# df = pd.read_csv("auto-mpg.data", sep=r'\s+', header=None, names=cols, na_values='?')

print(df.head())
print(df.isnull().sum())

# ---------------- STEP 2: PREPROCESS ----------------
df = df.dropna(subset=['displacement', 'mpg'])   # remove rows missing the 2 columns we use
X = df[['displacement']].values   # input  (2D, double brackets)
y = df['mpg'].values              # output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- STEP 3: LINEAR REGRESSION ----------------
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)
print("\nModel\t\tMSE\tR2")
print(f"Linear\t\t{mean_squared_error(y_test, y_pred):.2f}\t{r2_score(y_test, y_pred):.4f}")

# ---------------- STEP 4: POLYNOMIAL REGRESSION (degree 2, 3, 5) ----------------
degrees = [2, 3, 5]
models = {}
for d in degrees:
    poly = PolynomialFeatures(degree=d)             # x -> [1, x, x^2, ... x^d]
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    print(f"Degree {d}\t{mean_squared_error(y_test, y_pred):.2f}\t{r2_score(y_test, y_pred):.4f}")
    models[d] = (poly, model)

# ---------------- STEP 5: PLOT ----------------
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

plt.scatter(X, y, s=15, label="Data")
plt.plot(X_range, lin.predict(X_range), color='red', label="Linear")
for d, color in zip(degrees, ['green', 'orange', 'purple']):
    poly, model = models[d]
    plt.plot(X_range, model.predict(poly.transform(X_range)), color=color, label=f"Degree {d}")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
