from lstm_model import train_lstm_model

# List of tickers to pre-train models for
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# Date range for training data
start_date = "2020-01-01"
end_date = "2023-10-01"

# Train and save models for each ticker
for ticker in tickers:
    print(f"Training model for {ticker}...")
    train_lstm_model(ticker, start_date, end_date)
    print(f"Model for {ticker} saved to models/lstm_{ticker}.h5")