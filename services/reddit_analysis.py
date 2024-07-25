import logging
import re
from collections import Counter
from textblob import TextBlob
from ai.openai_client import OpenAIClient
from clients.reddit_client import RedditClient

client = OpenAIClient()

def generate_summary(title, post_text, comments_text):
    """Generate a summary of a post and its comments using AI"""
    return client.generate_summary(title, post_text, comments_text)

def extract_currencies(text, cmc_data):
    """Extract currency mentions using CoinMarketCap data with stricter filtering"""
    mentioned_currencies = []
    words = re.findall(r'\b[A-Z0-9]{2,10}\b', text.upper())  # Match 2-10 character uppercase words
    
    # Common words to exclude
    common_words = set(['THE', 'A', 'AN', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'AND', 'OR', 'BUT', 'IS', 'ARE', 'WAS', 'WERE',
                        'WILL', 'BE', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'CAN', 'COULD', 'WOULD', 'SHOULD', 'MAY', 'MIGHT',
                        'MUST', 'SHALL', 'NOT', 'NO', 'YES', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN'])
    
    # Minimum market cap threshold (e.g., $10 million)
    min_market_cap = 10_000_000

    for word in words:
        if word in cmc_data and word not in common_words:
            token = cmc_data[word]
            market_cap = token['quote']['USD']['market_cap']
            if market_cap >= min_market_cap:
                mentioned_currencies.append(token['symbol'])

    return list(dict.fromkeys(mentioned_currencies))  # Remove duplicates while preserving order

def analyze_reddit(client, query, subreddit, num_posts, num_comments, cmc_data):
    """Analyze Reddit posts and comments for cryptocurrency mentions and sentiment, and generate summaries"""
    try:
        posts = client.search_posts(query=query, subreddit_name=subreddit, limit=num_posts)
        
        if not posts:
            print(f"No posts found for query '{query}' in r/{subreddit}. The subreddit might not exist or could be private.")
            return

        all_text = ""
        currency_mentions = Counter()
        summaries = []

        for post in posts:
            post_text = f"{post.title} {post.text}"
            all_text += post_text + " "
            
            currencies = extract_currencies(post_text, cmc_data)
            currency_mentions.update(currencies)
            
            comments = client.get_comments(post, max_comments=num_comments)
            comment_text = " ".join([comment.body for comment in comments])
            all_text += comment_text + " "
            
            comment_currencies = extract_currencies(comment_text, cmc_data)
            currency_mentions.update(comment_currencies)
            
            # Generate summary for this post and its comments
            summary = generate_summary(post.title, post.text, comment_text)
            summaries.append(summary)

        sentiment = TextBlob(all_text).sentiment
        return {
            "currency_mentions": currency_mentions,
            "overall_sentiment": sentiment.polarity,
            "analyzed_text": all_text,
            "summaries": summaries
        }
    except Exception as e:
        logging.error(f"Error in analyze_reddit: {str(e)}")
        return None
