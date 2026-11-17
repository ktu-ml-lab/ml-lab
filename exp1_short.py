import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("housing.csv")
df['rooms'] = df['total_rooms'] / df['households']    # avg rooms per house
df = df[df['rooms'] < 10]                             # remove outliers
x = df['rooms'].values
y = df['median_house_value'].values
x = (x - x.mean()) / x.std()                          # normalize

# gradient descent
a, b = 0, 0                     # a = intercept, b = slope
cost = []
for i in range(1000):
    e = y - (a + b * x)         # error
    a = a + 0.01 * 2 * e.mean()
    b = b + 0.01 * 2 * (e * x).mean()
    cost.append((e ** 2).mean())

# normal equation
X = np.c_[np.ones(len(x)), x]
t = np.linalg.inv(X.T @ X) @ X.T @ y

print("GD:", a, b, " MSE:", mean_squared_error(y, a + b * x), " R2:", r2_score(y, a + b * x))
print("NE:", t[0], t[1], " MSE:", mean_squared_error(y, X @ t), " R2:", r2_score(y, X @ t))

plt.scatter(x, y, s=2)
plt.plot(x, a + b * x, 'r', label="gradient descent")
plt.plot(x, X @ t, 'g--', label="normal equation")
plt.legend()
plt.show()

plt.plot(cost)
plt.title("cost")
plt.show()
