import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
import plotly.graph_objects as go

def download_data(ticker, start_date, end_date):
    """
    Download historical stock data.
    """
    try:
        # Check if cached data exists
        cache_path = f'data/{ticker}.csv'
        if os.path.exists(cache_path):
            print(f"Loading cached data for {ticker}...")
            data = pd.read_csv(cache_path, parse_dates=['Date'], index_col='Date')
        else:
            # Download data from Yahoo Finance
            print(f"Downloading data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)
            print(f"Downloaded data for {ticker}:")
            print(data.head())  # Debug: Print the first few rows of the data
            
            # Save data to cache
            data.index.name = 'Date'  # Set the index name to 'Date'
            data.reset_index(inplace=True)  # Reset index to make 'Date' a column
            os.makedirs('data', exist_ok=True)  # Create 'data' directory if it doesn't exist
            data.to_csv(cache_path, index=False)  # Save without the index
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Handle missing values
        if data['Close'].isnull().any():
            print(f"Found missing values in data for {ticker}. Filling missing values...")
            data['Close'].fillna(method='ffill', inplace=True)  # Forward fill missing values
        
        # Return the 'Close' prices as a NumPy array
        return data['Close'].values.reshape(-1, 1)
    except Exception as e:
        raise ValueError(f"Error downloading data for {ticker}: {e}")

def preprocess_data(data, lookback=60, forecast_horizon=30):
    """
    Preprocess data for LSTM model.
    """
    if len(data) < lookback:
        raise ValueError(f"Insufficient data points for lookback period. Required: {lookback}, Available: {len(data)}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data) - forecast_horizon):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i+forecast_horizon, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    """
    Build the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(ticker, start_date, end_date, forecast_horizon=30):
    """
    Train the LSTM model for a specific forecast horizon.
    """
    try:
        # Download and preprocess data
        data = download_data(ticker, start_date, end_date)
        if data is None:
            raise ValueError(f"No data available for {ticker}")
        
        # Preprocess the data
        X, y, scaler = preprocess_data(data, forecast_horizon=forecast_horizon)
        
        # Build and train the LSTM model
        model = build_lstm_model((X.shape[1], 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X, y, batch_size=32, epochs=100, validation_split=0.2, callbacks=[early_stopping])
        
        # Save the model and scaler
        os.makedirs('models', exist_ok=True)  # Create 'models' directory if it doesn't exist
        model.save(f'models/lstm_{ticker}_{forecast_horizon}.h5')
        joblib.dump(scaler, f'models/scaler_{ticker}_{forecast_horizon}.pkl')  # Save the scaler
        return model, scaler
    except Exception as e:
        print(f"Error training LSTM model for {ticker}: {e}")
        return None, None

def predict_future_prices(model, scaler, data, lookback=60, forecast_horizon=30):
    """
    Predict future prices using the trained LSTM model.
    """
    try:
        scaled_data = scaler.transform(data)
        X_test = []
        X_test.append(scaled_data[-lookback:, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        return predicted_price[0][0]
    except Exception as e:
        print(f"Error predicting future prices: {e}")
        return None
    
def generate_plot_data(ticker, start_date, end_date, model, scaler, forecast_horizon):
    """
    Generate historical and predicted price data for plotting.
    """
    # Download historical data
    data = download_data(ticker, start_date, end_date)
    dates = pd.date_range(start=start_date, end=end_date, periods=len(data))
    
    # Predict future prices
    predicted_price = predict_future_prices(model, scaler, data, forecast_horizon=forecast_horizon)
    
    # Generate future dates
    future_date = dates[-1] + pd.Timedelta(days=forecast_horizon)
    
    # Combine historical and predicted data
    historical_prices = data.flatten()
    predicted_prices = [historical_prices[-1], predicted_price]
    
    return dates, historical_prices, future_date, predicted_prices
