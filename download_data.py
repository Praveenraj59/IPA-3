import yfinance as yf
import pandas as pd
import os
import time
import random

# Create a data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# List of tickers to download data for
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# Date range for data
start_date = "2021-05-01"
end_date = "2024-02-01"

# Number of retries for failed downloads
max_retries = 5

# Download and save data for each ticker
for ticker in tickers:
    retries = 0
    while retries < max_retries:
        try:
            print(f"Downloading data for {ticker} (Attempt {retries + 1})...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                print(f"No data found for {ticker}.")
            else:
                # Ensure the 'Date' column is present
                data.index.name = 'Date'  # Set the index name to 'Date'
                data.reset_index(inplace=True)  # Reset index to make 'Date' a column
                data.to_csv(f'data/{ticker}.csv', index=False)  # Save without the index
                print(f"Data for {ticker} saved to data/{ticker}.csv")
                break  # Exit the retry loop if successful
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            retries += 1
            if retries < max_retries:
                wait_time = random.uniform(5, 15)  # Random wait time between 5 and 15 seconds
                print(f"Retrying after {wait_time:.2f} seconds...")
                time.sleep(wait_time)  # Wait before retrying
            else:
                print(f"Failed to download data for {ticker} after {max_retries} attempts.")