import numpy as np
import matplotlib. pyplot as plt
import matplotlib
from sklearn. preprocessing import MinMaxScaler
from keras. layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from keras.models import model_from_json

import keras
import os
import pandas as pd
import sys

np.random.seed(7)

# Graph predicted vs True Close Value – LSTM
def graph(pred, expected):
    plt.plot(expected, label='True Value')
    plt.plot(pred, label='LSTM Value')
    plt.title('Prediction by LSTM')
    plt.xlabel('Time Scale')
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.savefig("prediction.png")
    plt.show()

#Get the Dataset
df=pd.read_csv('supplemental_train.csv',na_values=['null'],index_col='timestamp',parse_dates=True,infer_datetime_format=True)
df.fillna(method='ffill', inplace=True)

os.system("clear")

pairs = {'BTCUSD': 1}

#Set Target Variable
asset_ix = df['Asset_ID'] == pairs['BTCUSD']
df = df[asset_ix]
output_var = df['Close']
#Selecting the Features
features = ['Count','Open', 'High', 'Low', 'Volume', 'VWAP', 'Target']

scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)

#Splitting to Training set and Test set
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()


#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

# load json, create model and output prediction
if sys.argv[1] == "run":
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    lstm = model_from_json(loaded_model_json)
    # load weights into new model
    lstm.load_weights("model.h5")
    print("Loaded model from disk")
    pred = lstm.predict(X_train)
    graph(pred, y_train)
    exit()

#Building the LSTM Model
lstm = keras.models.Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

#Model Training
history=lstm.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction
y_pred= lstm.predict(X_test)

#Predicted vs True Close Value – LSTM
graph(y_pred, y_test)

 
# serialize model to JSON
model_json = lstm.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
lstm.save_weights("model.h5")
print("Saved model to disk")