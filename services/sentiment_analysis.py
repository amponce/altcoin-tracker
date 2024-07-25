import logging
from textblob import TextBlob
from ai.openai_client import OpenAIClient

def analyze_sentiment_openai(symbol: str, description: str) -> float:
    logging.debug(f"Analyzing sentiment for {symbol}")
    client = OpenAIClient()
    try:
        response = client.analyze_sentiment(symbol, description)
        return response
    except Exception as e:
        logging.error(f"Error analyzing sentiment with OpenAI: {e}")
        return 0

def analyze_reddit_sentiment(symbol: str, comments: list) -> float:
    sentiment_scores = []
    for comment in comments:
        if symbol.lower() in comment.body.lower():
            blob = TextBlob(comment.body)
            sentiment_scores.append(blob.sentiment.polarity)
    
    if sentiment_scores:
        return sum(sentiment_scores) / len(sentiment_scores)
    else:
        return 0
