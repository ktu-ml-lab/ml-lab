# Exp 3: Linear vs Ridge vs Lasso on Diabetes dataset (simple version)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data (sklearn)
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 1. Load data (CSV) - use instead if needed
# df = pd.read_csv("diabetes.csv")
# X = df.drop('target', axis=1)
# y = df['target']

# 2. Preprocessing
print("Shape:", X.shape, " Missing:", X.isnull().sum().sum())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train 3 models
lin = LinearRegression()
ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
lasso = LassoCV(cv=5)

for name, model in [("Linear", lin), ("Ridge", ridge), ("Lasso", lasso)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, " MSE =", round(mean_squared_error(y_test, y_pred), 2),
          " R2 =", round(r2_score(y_test, y_pred), 4))

print("Best Ridge alpha:", ridge.alpha_)
print("Best Lasso alpha:", lasso.alpha_)

# 4. Plot coefficients
coef = pd.DataFrame({'Linear': lin.coef_, 'Ridge': ridge.coef_, 'Lasso': lasso.coef_},
                    index=X.columns)
print(coef)
coef.plot(kind='bar')
plt.title("Coefficients: Linear vs Ridge vs Lasso")
plt.ylabel("Coefficient value")
plt.show()
