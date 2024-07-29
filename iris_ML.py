import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris)

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int32)

#training the model 

log_reg = LogisticRegression()
log_reg.fit(X,y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label = "Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label = "Not Irisi-Virginica")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()