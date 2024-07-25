import logging
from typing import List
from textblob import TextBlob
from ai.openai_client import OpenAIClient

def get_openai_sentiment(symbol: str, description: str) -> float:
    logging.debug(f"Analyzing sentiment for {symbol}")
    client = OpenAIClient()
    try:
        return client.analyze_sentiment_openai(symbol, description)
    except Exception as e:
        logging.error(f"Error analyzing sentiment with OpenAI: {e}")
        return 0.0

def analyze_reddit_sentiment(symbol: str, comments: List[str], limit: int = 100) -> float:
    sentiment_scores = []
    for comment in comments[:limit]:
        if symbol.lower() in comment.lower():
            blob = TextBlob(comment)
            sentiment_scores.append(blob.sentiment.polarity)
    
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
