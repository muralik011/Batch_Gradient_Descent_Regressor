import pandas as pd
import numpy as np
from gradient_descent import BatchGradientDescentRegressor

y_train = pd.DataFrame({'x': np.arange(-1000000, 1000000, 2)}, dtype='int64')
x_train = y_train.shift(1)
x = pd.concat([x_train, y_train], axis=1)
x = x.sample(frac=1).copy()
x = x.reset_index(drop=True).copy()
x.dropna(inplace=True)
x.columns = ['x', 'y']
x_train = x[['x']]
y_train = x[['y']]
len_x = len(x_train)
test_size = int(0.2 * len_x)

x_test = x_train.iloc[:test_size].copy()
y_test = y_train.iloc[:test_size].copy()

x_train = x_train.iloc[test_size:].copy()
y_train = y_train.iloc[test_size:].copy()

gd = BatchGradientDescentRegressor(n_epochs=550, learning_rate=0.1)

x_train_max = np.max(x_train)
x_train = x_train/x_train_max # normalization
x_train = x_train.values.reshape(-1, 1)

gd.fit(x_train, y_train.values)

y_predicted = gd.predict(x_test / x_train_max)
print(f'Test MSE: {np.mean((y_predicted - y_test) ** 2)}')
# Test MSE: 0.000021
