import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt

# 1. Load training data
data_train = pd.read_csv("Google_Stock_Price_Train.csv")
train = data_train.loc[:, ["Open"]].values
print("Training Data Shape:", train.shape)

# 2. Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

# 3. Create time-series data structure
x_train = []
y_train = []
timesteps = 5
for i in range(timesteps, len(train_scaled)):
    x_train.append(train_scaled[i - timesteps:i])
    y_train.append(train_scaled[i])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape input to 3D [samples, timesteps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. Build the RNN model
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

regressor = Sequential()
regressor.add(SimpleRNN(units=100, activation="relu", return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(SimpleRNN(units=100, activation="relu", return_sequences=True))
regressor.add(SimpleRNN(units=100, activation="relu", return_sequences=True))
regressor.add(SimpleRNN(units=100, activation="relu"))
regressor.add(Dense(units=1))

# 5. Compile and train
regressor.compile(optimizer="adam", loss="mse")
regressor.fit(x_train, y_train, epochs=10, batch_size=1)

# 6. Load test data
data_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = data_test.loc[:, ["Open"]].values

# 7. Prepare test inputs
data_total = pd.concat((data_train["Open"], data_test["Open"]), axis=0)
inputs = data_total[len(data_total) - len(data_test) - timesteps:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(timesteps, len(inputs)):
    x_test.append(inputs[i - timesteps:i])
x_test = np.array(x_test)

# Reshape for RNN input
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 8. Predict
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# 9. Visualize the results
plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
