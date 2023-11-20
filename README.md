# ml-record
## 1.Implementation of Univariate Linear Regression
```py

import numpy as np
import matplotlib.pyplot as plt

X = np.array(eval(input()))
Y = np.array(eval(input()))

X_mean=np.mean(X)
Y_mean=np.mean(Y)

num = 0
denom = 0

for i in range(len(X)):
    num += (X[i]-X_mean)*(Y[i]-Y_mean)
    denom += (X[i]-X_mean)**2

m = num/denom

b = Y_mean - m*X_mean
print (m, b)


Y_pred = m*X+b
print (Y_pred)


print("X values : ",X)
print("Y values : ",Y)
dots=[150]
plt.figure(figsize=(10, 8))
plt.scatter(X,Y,color='green',s=dots)
plt.plot(X,Y_pred,color='red',linewidth=4)
plt.xlabel("X-axis",fontweight='bold',fontsize=20)
plt.ylabel("Y-axis",fontweight='bold',fontsize=20)
plt.show()
```
