# Exp 3: Ridge (L2) and Lasso (L1) Regression on Diabetes dataset
# Tune alpha using cross-validation, compare with plain Linear Regression
# Metrics: MSE, R2    Plots: actual vs predicted, coefficient shrinkage

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- STEP 1: LOAD DATA (keep ONE option, comment the others) ----------------
# OPTION A: from scikit-learn (built in, already scaled)
from sklearn.datasets import load_diabetes
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# OPTION B: from CSV file (made from sklearn, already scaled)
# df = pd.read_csv("diabetes.csv")
# X = df.drop('target', axis=1)
# y = df['target']

# OPTION C: from .data file (original raw file: tab separated, header, NOT scaled)
# df = pd.read_csv("diabetes.data", sep='\t')
# X = df.drop('Y', axis=1)                 # target column is called Y here
# y = df['Y']
# X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)   # MUST scale raw values

print(X.head())

# ---------------- STEP 2: SPLIT ----------------
print("Missing values:", X.isnull().sum().sum())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- STEP 3: TRAIN 3 MODELS ----------------
lin = LinearRegression()
lin.fit(X_train, y_train)

ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)   # tries 50 alphas, 5-fold CV
ridge.fit(X_train, y_train)

lasso = LassoCV(cv=5, max_iter=10000)                  # picks alpha by 5-fold CV
lasso.fit(X_train, y_train)

# ---------------- STEP 4: EVALUATE ----------------
models = {"Linear": lin, "Ridge": ridge, "Lasso": lasso}
print("\nModel\tMSE\t\tR2")
for name, m in models.items():
    y_pred = m.predict(X_test)
    print(f"{name}\t{mean_squared_error(y_test, y_pred):.2f}\t\t{r2_score(y_test, y_pred):.4f}")

print("\nBest Ridge alpha:", ridge.alpha_)
print("Best Lasso alpha:", lasso.alpha_)

# ---------------- STEP 5: PLOTS ----------------
for name, m in models.items():
    plt.scatter(y_test, m.predict(X_test), label=name, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')   # perfect line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()

coef = pd.DataFrame({'Linear': lin.coef_, 'Ridge': ridge.coef_, 'Lasso': lasso.coef_}, index=X.columns)
print("\nCoefficients:\n", coef)
coef.plot(kind='bar')
plt.title("Coefficient Comparison (Shrinkage)")
plt.ylabel("Coefficient value")
plt.show()
