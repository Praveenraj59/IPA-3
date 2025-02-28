from flask import Flask, render_template, request
import yfinance as yf
from lstm_model import train_lstm_model, predict_future_prices, generate_plot_data
from sentiment_analysis import get_sentiment_score
import numpy as np
import os
import joblib
import plotly
import json
from plotly import graph_objects as go  # Import Plotly's graph_objects as go

app = Flask(__name__)

# Risk tolerance mapping
risk_mapping = {
    'Conservative': 0.2,
    'Moderate': 0.5,
    'Aggressive': 0.8
}

def is_valid_ticker(ticker):
    """
    Check if the ticker symbol is valid.
    """
    stock = yf.Ticker(ticker)
    try:
        # Check if the ticker has valid historical data
        history = stock.history(period="1d")
        return not history.empty
    except:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/beginner', methods=['GET', 'POST'])
def beginner():
    if request.method == 'POST':
        # Get form data
        ticker = request.form['ticker']
        risk_tolerance_text = request.form['risk_tolerance']
        risk_tolerance = risk_mapping[risk_tolerance_text]
        investment_amount = float(request.form['investment_amount'])
        review_frequency = request.form['review_frequency']
        
        # Validate ticker
        if not is_valid_ticker(ticker):
            return render_template('error.html', message=f"Invalid ticker symbol: {ticker}. Please check the symbol and try again.")
        
        # Map review frequency to forecast horizon
        if review_frequency == 'monthly':
            forecast_horizon = 30  # 1 month
        elif review_frequency == 'quarterly':
            forecast_horizon = 90  # 3 months
        elif review_frequency == 'annually':
            forecast_horizon = 365  # 12 months
        
        # Define date range
        start_date = "2020-01-01"
        end_date = "2023-10-01"
        
        # Check if a pre-trained model exists
        model_path = f'models/lstm_{ticker}_{forecast_horizon}.h5'
        scaler_path = f'models/scaler_{ticker}_{forecast_horizon}.pkl'
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)  # Load the scaler
            print(f"Loaded pre-trained model and scaler for {ticker}.")
        else:
            # Train a new model
            model, scaler = train_lstm_model(ticker, start_date, end_date, forecast_horizon)
            if model is None or scaler is None:
                return render_template('error.html', message=f"Failed to train LSTM model for {ticker}. Please check the ticker symbol and try again.")
        
        # Fetch data and predict future prices
        data = yf.download(ticker, start=start_date, end=end_date)['Close'].values.reshape(-1, 1)
        predicted_price = predict_future_prices(model, scaler, data, forecast_horizon=forecast_horizon)
        
        if predicted_price is None:
            return render_template('error.html', message=f"Failed to predict future prices for {ticker}. Please try again.")
        
        # Sentiment analysis
        sentiment_score = get_sentiment_score(ticker)
        
        # Decision logic
        current_price = data[-1][0]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        if price_change > 5 and sentiment_score > 0:
            recommendation = "Buy"
        elif price_change < -5 and sentiment_score < 0:
            recommendation = "Avoid"
        else:
            recommendation = "Hold"
        
        # Insights
        sentiment_description = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        insights = f"The stock is expected to {'increase' if price_change > 0 else 'decrease'} by {abs(price_change):.2f}% in the next {forecast_horizon} days. Sentiment is {sentiment_description}."
        
        # Generate plot data
        dates, historical_prices, future_date, predicted_prices = generate_plot_data(
            ticker, start_date, end_date, model, scaler, forecast_horizon
        )

        # Create a line graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=historical_prices, mode='lines', name='Historical Prices'
        ))
        fig.add_trace(go.Scatter(
            x=[dates[-1], future_date], y=predicted_prices, mode='lines', name='Predicted Prices'
        ))
        fig.update_layout(
            title=f"{ticker} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            showlegend=True
        )

        # Convert the graph to JSON for embedding in HTML
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('result.html', 
                               ticker=ticker, 
                               predicted_price=round(predicted_price, 2), 
                               sentiment_score=round(sentiment_score, 2), 
                               recommendation=recommendation,
                               insights=insights,
                               graph_json=graph_json)
    return render_template('beginner.html')

@app.route('/investor', methods=['GET', 'POST'])
def investor():
    if request.method == 'POST':
        # Get form data
        ticker = request.form['ticker']
        risk_tolerance_text = request.form['risk_tolerance']
        risk_tolerance = risk_mapping[risk_tolerance_text]
        current_allocation = float(request.form['current_allocation'])
        investment_horizon = request.form['investment_horizon']
        rebalance_frequency = request.form['rebalance_frequency']
        
        # Validate ticker
        if not is_valid_ticker(ticker):
            return render_template('error.html', message=f"Invalid ticker symbol: {ticker}. Please check the symbol and try again.")
        
        # Map rebalance frequency to forecast horizon
        if rebalance_frequency == 'monthly':
            forecast_horizon = 30  # 1 month
        elif rebalance_frequency == 'quarterly':
            forecast_horizon = 90  # 3 months
        elif rebalance_frequency == 'annually':
            forecast_horizon = 365  # 12 months
        
        # Define date range
        start_date = "2020-01-01"
        end_date = "2023-10-01"
        
        # Check if a pre-trained model exists
        model_path = f'models/lstm_{ticker}_{forecast_horizon}.h5'
        scaler_path = f'models/scaler_{ticker}_{forecast_horizon}.pkl'
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)  # Load the scaler
            print(f"Loaded pre-trained model and scaler for {ticker}.")
        else:
            # Train a new model
            model, scaler = train_lstm_model(ticker, start_date, end_date, forecast_horizon)
            if model is None or scaler is None:
                return render_template('error.html', message=f"Failed to train LSTM model for {ticker}. Please check the ticker symbol and try again.")
        
        # Fetch data and predict future prices
        data = yf.download(ticker, start=start_date, end=end_date)['Close'].values.reshape(-1, 1)
        predicted_price = predict_future_prices(model, scaler, data, forecast_horizon=forecast_horizon)
        
        if predicted_price is None:
            return render_template('error.html', message=f"Failed to predict future prices for {ticker}. Please try again.")
        
        # Sentiment analysis
        sentiment_score = get_sentiment_score(ticker)
        
        # Decision logic
        current_price = data[-1][0]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        if price_change > 5 and sentiment_score > 0:
            recommendation = "Buy More"
        elif price_change < -5 and sentiment_score < 0:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
        
        # Insights
        sentiment_description = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        insights = f"The stock is expected to {'increase' if price_change > 0 else 'decrease'} by {abs(price_change):.2f}% in the next {forecast_horizon} days. Sentiment is {sentiment_description}."
        
        # Generate plot data
        dates, historical_prices, future_date, predicted_prices = generate_plot_data(
            ticker, start_date, end_date, model, scaler, forecast_horizon
        )

        # Create a line graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=historical_prices, mode='lines', name='Historical Prices'
        ))
        fig.add_trace(go.Scatter(
            x=[dates[-1], future_date], y=predicted_prices, mode='lines', name='Predicted Prices'
        ))
        fig.update_layout(
            title=f"{ticker} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            showlegend=True
        )

        # Convert the graph to JSON for embedding in HTML
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('result.html', 
                               ticker=ticker, 
                               predicted_price=round(predicted_price, 2), 
                               sentiment_score=round(sentiment_score, 2), 
                               recommendation=recommendation,
                               insights=insights,
                               graph_json=graph_json)
    return render_template('investor.html')

if __name__ == '__main__':
    app.run(debug=True) 