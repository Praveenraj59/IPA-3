from newsapi import NewsApiClient
from transformers import pipeline
import os

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=os.getenv("91320d29db69404193ce5597fa9e9479"))

def fetch_news(ticker):
    """
    Fetch news articles related to the stock ticker.
    """
    try:
        news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
        articles = news['articles']
        
        # Handle cases where 'title' or 'description' is None
        texts = []
        for article in articles:
            title = article.get('title', '')  # Use empty string if 'title' is None
            description = article.get('description', '')  # Use empty string if 'description' is None
            texts.append(f"{title} {description}".strip())  # Combine and strip extra spaces
        
        return texts
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(texts):
    """
    Analyze sentiment using a pre-trained NLP model.
    """
    try:
        sentiment_pipeline = pipeline("sentiment-analysis")
        sentiments = []
        for text in texts:
            result = sentiment_pipeline(text)[0]
            sentiments.append((result['label'], result['score']))
        return sentiments
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return []

def get_sentiment_score(ticker):
    """
    Get the average sentiment score for a stock ticker.
    """
    try:
        texts = fetch_news(ticker)
        if not texts:  # If no articles are found
            return 0
        
        sentiments = analyze_sentiment(texts)
        scores = [score if label == 'POSITIVE' else -score for label, score in sentiments]
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        print(f"Error calculating sentiment score for {ticker}: {e}")
        return 0