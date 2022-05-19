import pandas as pd
import numpy as np
from gradient_descent import BatchGradientDescentRegressor

y = pd.DataFrame({'x': np.arange(-1000000, 1000000, 2)}, dtype='int64')
x = y.shift(1)
data = pd.concat([x, y], axis=1)
data = data.sample(frac=1).copy()
data = data.reset_index(drop=True).copy()
data.dropna(inplace=True)
data.columns = ['x', 'y']
x = data[['x']].copy()
y = data[['y']].copy()
len_x = len(x)
test_size = int(0.2 * len_x)

x_test = x.iloc[:test_size].copy()
y_test = y.iloc[:test_size].copy()

x_train = x.iloc[test_size:].copy()
y_train = y.iloc[test_size:].copy()

bgd = BatchGradientDescentRegressor(n_epochs=550, learning_rate=0.1)

x_train_max = np.max(x_train)
x_train = x_train/x_train_max # normalization
x_train = x_train.values.reshape(-1, 1)

bgd.fit(x_train, y_train.values)

y_predicted = bgd.predict(x_test / x_train_max)
print(f'Test MSE: {np.mean((y_predicted - y_test) ** 2)}')
# Training MSE: 2.2950712791315194e-05
# Test MSE: 0.000021
