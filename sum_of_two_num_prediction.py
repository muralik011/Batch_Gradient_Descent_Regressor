import numpy as np
from gradient_descent import BatchGradientDescentRegressor


bgd = BatchGradientDescentRegressor(1000, 0.01)
X_train = np.random.rand(10000)
X_train = X_train.reshape((-1, 2)).copy()
print(f'X_train shape: {X_train.shape}')
y_train = np.sum(X_train, axis=1)
print(f'y_train shape: {y_train.shape}')
bgd.fit(X_train, y_train)
print(f'Training MSE: {np.round(bgd.loss, 5)}')

X_test = np.random.rand(1000)
X_test = X_test.reshape((-1, 2)).copy()

print(f'X_test shape: {X_test.shape}')
y_test = np.sum(X_test, axis=1)
print(f'y_test shape: {y_test.shape}')

y_predicted = bgd.predict(X_test)
print(f'Test MSE: {np.mean((y_predicted - y_test) ** 2)}')
