import logging
from typing import List, Dict
from config import OPPORTUNITY_CRITERIA
from services.sentiment_analysis import get_openai_sentiment, analyze_reddit_sentiment

def analyze_trends(current_data: List[Dict], previous_data: List[Dict]) -> List[Dict]:
    logging.debug("Analyzing trends")
    opportunities = []
    
    for current, previous in zip(current_data, previous_data):
        if current["id"] != previous["id"]:
            continue
        
        symbol = current["symbol"]
        current_price = current["quote"]["USD"]["price"]
        previous_price = previous["quote"]["USD"]["price"]
        price_change = (current_price - previous_price) / previous_price
        
        current_volume = current["quote"]["USD"]["volume_24h"]
        previous_volume = previous["quote"]["USD"]["volume_24h"]
        volume_change = (current_volume - previous_volume) / previous_volume
        
        market_cap = current["quote"]["USD"]["market_cap"]
        circulating_supply = current["circulating_supply"]
        
        description = current.get("description", "")
        sentiment_score = get_openai_sentiment(symbol, description)
        reddit_comments = current.get("reddit_comments", [])  # Assume this is added somewhere
        reddit_sentiment = analyze_reddit_sentiment(symbol, reddit_comments)
        
        if (price_change > OPPORTUNITY_CRITERIA['price_change_threshold'] and 
            volume_change > OPPORTUNITY_CRITERIA['volume_change_threshold'] and 
            sentiment_score > OPPORTUNITY_CRITERIA['sentiment_threshold']):
            opportunities.append({
                "symbol": symbol,
                "price_change": price_change,
                "volume_change": volume_change,
                "sentiment_score": sentiment_score,
                "reddit_sentiment": reddit_sentiment,
                "market_cap": market_cap,
                "circulating_supply": circulating_supply
            })
    
    logging.debug(f"Found {len(opportunities)} potential opportunities")
    return opportunities