import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
df = pd.read_csv('BostonHousing.csv')
df.head(n=10)
# df.drop(columns=['CAT. MEDV'],inplace=True)
# df.dropna(inplace=True)
df.isnull().sum()
df.info()
df.describe()
df.corr()['medv'].sort_values()
X = df.loc[:, df.columns != 'medv'].values #or X = df.loc[:,['lstat','ptratio','rm]]
y = df.loc[:, df.columns == 'medv'].values #or y = df.loc[:,'medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)
scaler = StandardScaler() #standardise (Z-score normalization => mean = 0 and Std = 1)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential(layers=[Dense(128, input_shape=(13, ), activation='relu', 
name='dense_1'),Dense(64, activation='relu', name='dense_2'),Dense(1, activation='linear', 
name='dense_output')])
model.compile(optimizer='adam', loss='mse',metrics=['mae'])
model.summary()
model.fit(X_train, y_train, epochs=100, validation_split=0.05, verbose = 'auto')
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
mse_nn, mae_nn= model.evaluate(X_test, y_test)
r2 = r2_score(y_test, y_pred)
print('Mean squared error on test data: ', mse_nn)
print('Mean absolute error on test data: ', mae_nn)
print('Accuracy:', r2*100)
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='green')
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()