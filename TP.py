import pandas as pd

df = pd.read_csv('../desktop/bodyPerformance.csv')

print(df.head())

df.groupby(['age','gender'])[['height','weight','body fat_%']].mean()

import matplotlib.pyplot as plt

AS = df.groupby(['age'])['body fat_%'].mean()
AS.plot()

import numpy as np
from sklearn.linear_model import LinearRegression

test_df = pd.read_csv('../desktop/bodyPerformance.csv')

X = np.array(df['age']).reshape(-1,1)
y = df['body fat_%']
lr = LinearRegression()
lr.fit(X, y)

w = lr.coef_[0]

plt.scatter(X, y, s=1)
plt.plot(X, w*X, c='red')

test_X = np.array(test_df['age']).reshape(-1,1)
pred_y = lr.predict(test_X)

plt.scatter(test_X, pred_y)
plt.plot(test_X, w*test_X, c='red')

test_y = test_df['body fat_%']

plt.scatter(test_X, test_y, c='purple', s=1)



from sklearn.metrics import mean_squared_error
test_loss = mean_squared_error(test_y, pred_y)

train_y = lr.predict(X)
train_loss = mean_squared_error(y, train_y)

print(test_loss, train_loss)
plt.scatter(test_y, pred_y)
plt.plot([0,np.max(test_y)],[0,np.max(test_y)], color='red')
plt.show()



from sklearn.metrics import r2_score
r2 = r2_score(test_y, pred_y)
print(r2)
