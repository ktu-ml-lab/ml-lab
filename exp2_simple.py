import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error

df=pd.read_csv("auto-mpg.csv")
print(df.head())
print(df.isnull().sum())

X=df[['displacement']].values      # input  (engine size)
y=df['mpg'].values                 # output (mileage)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# linear regression
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("linear regression:")
print("r2 score:",r2_score(y_test,y_pred))
print("mse:",mean_squared_error(y_test,y_pred))

# polynomial degree 2
poly2=PolynomialFeatures(degree=2)
X_train2=poly2.fit_transform(X_train)
X_test2=poly2.transform(X_test)
model2=LinearRegression()
model2.fit(X_train2,y_train)
y_pred2=model2.predict(X_test2)
print("polynomial degree 2:")
print("r2 score:",r2_score(y_test,y_pred2))
print("mse:",mean_squared_error(y_test,y_pred2))

# polynomial degree 3
poly3=PolynomialFeatures(degree=3)
X_train3=poly3.fit_transform(X_train)
X_test3=poly3.transform(X_test)
model3=LinearRegression()
model3.fit(X_train3,y_train)
y_pred3=model3.predict(X_test3)
print("polynomial degree 3:")
print("r2 score:",r2_score(y_test,y_pred3))
print("mse:",mean_squared_error(y_test,y_pred3))

# plot
X_range=np.linspace(X.min(),X.max(),100).reshape(-1,1)
plt.scatter(X,y,color='gray',label='actual data')
plt.plot(X_range,model.predict(X_range),color='blue',label='linear')
plt.plot(X_range,model2.predict(poly2.transform(X_range)),color='green',label='degree 2')
plt.plot(X_range,model3.predict(poly3.transform(X_range)),color='red',label='degree 3')
plt.xlabel("displacement")
plt.ylabel("mpg")
plt.title("linear vs polynomial regression")
plt.legend()
plt.show()
