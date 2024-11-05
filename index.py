import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime

file_path = r'D:\3 semister\Exiton\Task 5(Energy Consumption Forecasting)\household_power_consumption.txt'
data = pd.read_csv(file_path, sep=';', parse_dates={'datetime': [0, 1]}, infer_datetime_format=True, na_values='?', low_memory=False)


data.dropna(inplace=True)

data['Global_active_power'] = pd.to_numeric(data['Global_active_power'])

data.set_index('datetime', inplace=True)
daily_data = data['Global_active_power'].resample('D').sum()

# Visualize the data
plt.figure(figsize=(14, 5))
plt.plot(daily_data, label='Daily Global Active Power')
plt.title('Daily Global Active Power Consumption')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 30 
X, y = create_dataset(scaled_data, time_step)


X = X.reshape(X.shape[0], X.shape[1], 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, batch_size=32, epochs=100)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)) 

plt.figure(figsize=(14, 5))


plt.subplot(2, 1, 1)
plt.plot(y_test_actual, label='Actual Consumption', color='blue')
plt.title('Actual Global Active Power Consumption')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(predictions, label='Predicted Consumption', color='red')
plt.title('Predicted Global Active Power Consumption')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()

plt.tight_layout()
plt.show()

rmse = np.sqrt(np.mean((predictions - y_test_actual) ** 2))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
