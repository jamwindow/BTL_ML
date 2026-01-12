import numpy as np
import pandas as pd
import random
import serial
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from influxdb_client import InfluxDBClient, Point, WriteOptions
from sklearn.preprocessing import MinMaxScaler
import io

# import paho.mqtt.client as mqttclient
import time
import json

from twisted.python.util import println

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

WIFI_SSID = "RD-SEAI_2.4G"
WIFI_PASSWORD = ""
temp = 30
humi = 50
light = 0

url = ""
token = ""
org = ""
bucket = ""


# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# Find the end of this pattern
		end_ix = i + n_steps
		# Check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# Gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def split_sequences_ml(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences) - n_steps):
        # Flatten: (n_steps, n_features) → (n_steps * n_features)
        X.append(sequences[i:i+n_steps].flatten())
        y.append(sequences[i+n_steps])
    return np.array(X), np.array(y)


# Read given train and test sets
train_data = pd.read_csv(r"C:\Users\Admin\Desktop\Python\PycharmProjects\PythonProject\BTL_ML\data_rounded.csv")

train_data = train_data.dropna()   # bỏ các dòng NaN

humi_seq_train = np.array(train_data['Humidity'])
temp_seq_train = np.array(train_data['Temperature'])

humi_seq_train = humi_seq_train.reshape((len(humi_seq_train), 1))
temp_seq_train = temp_seq_train.reshape((len(temp_seq_train), 1))
# Horizontally stack columns
dataset = np.hstack((temp_seq_train, humi_seq_train))

scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)


# Choose a number of time steps
n_steps = 3
# Convert into input/output
X_dl, y_dl = split_sequences(dataset_scaled, n_steps)

X_ml, y_ml = split_sequences_ml(dataset_scaled, n_steps)

# The dataset knows the number of features
n_features = X_dl.shape[2]

dl_model = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)),
    MaxPooling1D(2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(n_features)
])

dl_model.compile(optimizer='adam', loss='mse')
dl_model.fit(X_dl, y_dl, epochs=50, verbose=1)

lr_model = LinearRegression()
lr_model.fit(X_ml, y_ml)


def DL_prediction(array):
    # x_input = np.array(array).reshape((1, n_steps, n_features))
    # predicted_value = model.predict(x_input, verbose=0)
    array_scaled = scaler.transform(array)
    x_input = array_scaled.reshape((1, n_steps, n_features))

    pred_scaled = dl_model.predict(x_input, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)

    print("DL Predicted temperature:", pred[0][0])
    print("DL Predicted humidity:", pred[0][1])
    print("-" * 20)
    temp_1 = float(pred[0][0])
    humi_1 = float(pred[0][1])
    return temp_1, humi_1

def ML_predict(temps, humis):
    array = np.column_stack((temps[-n_steps:], humis[-n_steps:]))

    array_scaled = scaler.transform(array)
    x_input = array_scaled.flatten().reshape(1, -1)

    pred_scaled = lr_model.predict(x_input)
    pred = scaler.inverse_transform(pred_scaled)

    temp = float(pred[0][0])
    humi = float(pred[0][1])

    print("ML Predicted temperature:", pred[0][0])
    print("ML Predicted humidity:", pred[0][1])
    print("-" * 20)
    return temp, humi

dl_pred = dl_model.predict(X_dl, verbose=0)
ml_pred = lr_model.predict(X_ml)

print("DL MSE:", mean_squared_error(y_dl, dl_pred))
print("ML MSE:", mean_squared_error(y_ml, ml_pred))

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
write_api = client.write_api(write_options=WriteOptions(batch_size=1))

while True:
    flux_query = '''
    from(bucket: "smart_home")
      |> range(start: -1m)
      |> filter(fn: (r) => r["_measurement"] == "mqtt_consumer")
      |> filter(fn: (r) => r["_field"] == "air_room01" or r["_field"] == "gas_room01" or r["_field"] == "humidity_room01" or r["_field"] == "temperature_room01")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''

    tables = query_api.query(flux_query)
    data = []
    for table in tables:
        for record in table.records:
            data.append([record.values.get("temperature_room01"),
                         record.values.get("humidity_room01")])

    data = np.array(data)
    temps = data[:, 0].tolist()
    humis = data[:, 1].tolist()

    # array = [temps, humis]
    # array = np.array(array)
    # array = array[:, -3:]

    array = np.column_stack((temps[-3:], humis[-3:]))

    print(array)

    DL_temp_predict, DL_humi_predict = DL_prediction(array)
    ML_temp_predict, ML_humi_predict = ML_predict(temps, humis)


    # Tạo point để ghi vào InfluxDB
    point = Point("dl_prediction") \
        .tag("host", "python_dl") \
        .field("DL_temperature_predict", DL_temp_predict) \
        .field("DL_humidity_predict", DL_humi_predict) \
        .field("ML_temperature_predict", ML_temp_predict) \
        .field("ML_humidity_predict", ML_humi_predict) \
        # Ghi vào bucket
    write_api.write(bucket=bucket, org=org, record=point)
    print("Đã ghi dữ liệu dự đoán vào InfluxDB")


    time.sleep(10)