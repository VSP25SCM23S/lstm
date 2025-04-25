# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from flask_cors import CORS

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import statsmodels.api as sm

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Initilize flask app
app = Flask(__name__)
CORS(app)
client = storage.Client()

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i) for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 60
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
        verbose=1,
        shuffle=False
    )

    # Move BASE_IMAGE_PATH and LOCAL_IMAGE_PATH here (before using them)
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    LOCAL_IMAGE_PATH = "static/images/"

    # --- FB Prophet Forecast ---
    prophet_df = pd.DataFrame({
        'ds': [firstDay + timedelta(days=i) for i in range(len(Ys))],
        'y': scaler.inverse_transform(Ys).flatten()
    })

    prophet_model = Prophet()
    prophet_model.fit(prophet_df)

    future = prophet_model.make_future_dataframe(periods=60)
    forecast = prophet_model.predict(future)

    plt.figure(figsize=(10, 6))
    prophet_model.plot(forecast)
    plt.title('FB Prophet Forecast for ' + type)
    PROPHET_FORECAST_IMAGE_NAME = "prophet_forecast_data_" + type + "_" + repo_name + ".png"
    PROPHET_FORECAST_IMAGE_PATH = LOCAL_IMAGE_PATH + PROPHET_FORECAST_IMAGE_NAME
    plt.savefig(PROPHET_FORECAST_IMAGE_PATH)
    plt.close()

    # --- StatsModels Forecast ---
    Ys_original = scaler.inverse_transform(Ys).flatten()

    model_arima = sm.tsa.ARIMA(Ys_original, order=(5,1,0))
    model_arima_fit = model_arima.fit()

    forecast_steps = 60
    forecast_arima = model_arima_fit.forecast(steps=forecast_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(Ys_original)), Ys_original, label='History')
    plt.plot(range(len(Ys_original), len(Ys_original) + forecast_steps), forecast_arima, label='Forecast')
    plt.title('StatsModels Forecast for ' + type)
    plt.xlabel('Days')
    plt.ylabel('Issues')
    plt.legend()
    STATSMODELS_FORECAST_IMAGE_NAME = "statsmodels_forecast_data_" + type + "_" + repo_name + ".png"
    STATSMODELS_FORECAST_IMAGE_PATH = LOCAL_IMAGE_PATH + STATSMODELS_FORECAST_IMAGE_NAME
    plt.savefig(STATSMODELS_FORECAST_IMAGE_PATH)
    plt.close()

    MODEL_LOSS_IMAGE_NAME = "model_loss_" + type +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + type +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_ISSUES_DATA_IMAGE_NAME = "all_issues_data_" + type + "_"+ repo_name + ".png"
    ALL_ISSUES_DATA_URL = BASE_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME

    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + type)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)
    plt.close()

    y_pred = model.predict(X_test)

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test, marker='.', label="true")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), y_pred, 'r', label="prediction")
    axs.legend()
    axs.set_title('LSTM Generated Data For ' + type)
    axs.set_xlabel('Time Steps')
    axs.set_ylabel('Issues')
    plt.savefig(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('All Issues Data')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    plt.savefig(LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)
    plt.close()

    bucket = client.get_bucket(BUCKET_NAME)
    bucket.blob(MODEL_LOSS_IMAGE_NAME).upload_from_filename(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)
    bucket.blob(ALL_ISSUES_DATA_IMAGE_NAME).upload_from_filename(LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)
    bucket.blob(LSTM_GENERATED_IMAGE_NAME).upload_from_filename(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)
    bucket.blob(PROPHET_FORECAST_IMAGE_NAME).upload_from_filename(PROPHET_FORECAST_IMAGE_PATH)
    bucket.blob(STATSMODELS_FORECAST_IMAGE_NAME).upload_from_filename(STATSMODELS_FORECAST_IMAGE_PATH)

    json_response = {
        "model_loss_image_url": MODEL_LOSS_URL,
        "lstm_generated_image_url": LSTM_GENERATED_URL,
        "all_issues_data_image": ALL_ISSUES_DATA_URL,
        "prophet_forecast_data_image": BASE_IMAGE_PATH + PROPHET_FORECAST_IMAGE_NAME,
        "statsmodels_forecast_data_image": BASE_IMAGE_PATH + STATSMODELS_FORECAST_IMAGE_NAME
    }

    return jsonify(json_response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
