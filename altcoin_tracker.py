import json
import requests
import time
from openai import OpenAI
import logging
import cmd
from requests.exceptions import RequestException
from typing import Dict, List
from ratelimit import limits, sleep_and_retry
from tabulate import tabulate
import os
from dotenv import load_dotenv
import praw
import prawcore
from textblob import TextBlob
import re
from collections import Counter
import random

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI API
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1/"
MAX_LISTINGS = 500
MAX_REQUESTS_PER_MINUTE = 30
OPPORTUNITY_CRITERIA = {
    'price_change_threshold': 0.05,
    'volume_change_threshold': 0.1,
    'sentiment_threshold': 0.6
}

# Headers for CoinMarketCap API requests
headers = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": os.getenv('CMC_API_KEY'),
}

class Post:
    def __init__(self, title, score, author, num_comments, url, id=None, text=''):
        self.title = title
        self.score = score
        self.author = author
        self.num_comments = num_comments
        self.url = url
        self.id = id
        self.text = text

class Comment:
    def __init__(self, author, score, body, depth=0, id=None):
        self.id = id
        self.author = author if author else '[deleted]'
        self.score = score
        self.body = body
        self.depth = depth
        self.children = []
        self.collapsed = False
        self.has_more_replies = False
        self.is_root = depth == 0

class RedditClient:
    def __init__(self, use_api=True):
        self.use_api = use_api
        self.reddit = self._authenticate()

    def _authenticate(self):
        try:
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            reddit.user.me()
            return reddit
        except prawcore.exceptions.ResponseException as e:
            logging.error(f"Error authenticating with Reddit API: {e}")
        except Exception as e:
            logging.error(f"An error occurred during Reddit authentication: {e}")
        return None

    def search_posts(self, query: str, subreddit_name: str = None, limit: int = 10):
        try:
            if self.reddit:
                if subreddit_name:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    posts = subreddit.search(query, limit=limit)
                else:
                    posts = self.reddit.subreddit('all').search(query, limit=limit)
                
                return [Post(post.title, post.score, post.author.name if post.author else '[deleted]', post.num_comments, post.url, post.id, post.selftext) for post in posts]
            else:
                logging.error("Reddit API not authenticated.")
                return []
        except prawcore.exceptions.RequestException as e:
            logging.error(f"Error in search_posts: {e}. HTTP Status: {e.response.status_code}")
            return []
        except Exception as e:
            logging.error(f"Error in search_posts: {e}")
            return []

    def get_posts(self, subreddit_name: str = None, sort: str = 'hot', limit: int = 10):
        try:
            if self.reddit:
                subreddit = self.reddit.subreddit(subreddit_name) if subreddit_name else self.reddit.front
                sorting = {
                    'hot': subreddit.hot,
                    'new': subreddit.new,
                    'top': subreddit.top
                }
                posts = sorting.get(sort, subreddit.hot)(limit=limit)
                return [Post(post.title, post.score, post.author.name if post.author else '[deleted]', post.num_comments, post.url, post.id, post.selftext) for post in posts]
            else:
                logging.error("Reddit API not authenticated.")
                return []
        except prawcore.exceptions.RequestException as e:
            logging.error(f"Error in get_posts: {e}. HTTP Status: {e.response.status_code}")
            return []
        except Exception as e:
            logging.error(f"Error in get_posts: {e}")
            return []

    def get_comments(self, post, max_comments=50):
        try:
            if self.reddit:
                submission = self.reddit.submission(id=post.id)
                submission.comments.replace_more(limit=0)
                comments = []
                for top_level_comment in submission.comments[:max_comments]:
                    comments.append(Comment(
                        author=top_level_comment.author.name if top_level_comment.author else '[deleted]',
                        score=top_level_comment.score,
                        body=top_level_comment.body[:500],
                        depth=top_level_comment.depth,
                        id=top_level_comment.id
                    ))
                return comments
            else:
                logging.error("Reddit API not authenticated.")
                return []
        except Exception as e:
            logging.error(f"Error in get_comments: {e}")
            return []

class RateLimiter:
    @sleep_and_retry
    @limits(calls=MAX_REQUESTS_PER_MINUTE, period=60)
    def limited_request(self, method, url, **kwargs):
        logging.debug(f"Making API request: {method} {url}")
        return requests.request(method, url, **kwargs)

rate_limiter = RateLimiter()

def get_latest_listings(limit: int = 500) -> List[Dict]:
    logging.debug(f"Fetching latest listings (limit: {limit})")
    endpoint = f"{CMC_BASE_URL}cryptocurrency/listings/latest"
    parameters = {
        "start": "1",
        "limit": str(limit),
        "convert": "USD",
        "aux": "platform"
    }
    
    try:
        response = rate_limiter.limited_request("GET", endpoint, headers=headers, params=parameters)
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, dict):
            logging.error(f"Unexpected response format. Expected dict, got {type(data)}")
            return []
        
        if "data" not in data:
            logging.error("Response does not contain 'data' key")
            return []
        
        listings = data["data"]
        if not isinstance(listings, list):
            logging.error(f"Unexpected 'data' format. Expected list, got {type(listings)}")
            return []
        
        logging.debug(f"Successfully fetched {len(listings)} listings")
        logging.debug(f"Sample listings: {listings[:5]}")  # Log first 5 listings for inspection
        return listings
    except RequestException as e:
        logging.error(f"Error fetching data from CoinMarketCap: {e}")
    except ValueError as e:
        logging.error(f"Error parsing JSON response: {e}")
    except KeyError as e:
        logging.error(f"Missing expected key in response: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in get_latest_listings: {e}")
    
    return []

def analyze_sentiment_openai(symbol: str, description: str) -> float:
    logging.debug(f"Analyzing sentiment for {symbol}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the sentiment of the given cryptocurrency description and respond with a single number between -1 (very negative) and 1 (very positive)."},
                {"role": "user", "content": f"Cryptocurrency: {symbol}\nDescription: {description}\n\nSentiment score:"}
            ],
            max_tokens=1,
            n=1,
            stop=None,
            temperature=0.5
        )
        
        response_content = response.choices[0].message.content.strip()
        logging.debug(f"Raw sentiment score response: {response_content}")
        
        try:
            sentiment_score = float(response_content)
            return max(-1, min(1, sentiment_score))
        except ValueError:
            logging.error(f"Error converting sentiment score to float: {response_content}")
            return 0
    except Exception as e:
        logging.error(f"Error analyzing sentiment with OpenAI: {e}")
        return 0

def analyze_reddit_sentiment(symbol: str) -> float:
    reddit_client = RedditClient()
    posts = reddit_client.get_posts(subreddit_name="CryptoCurrency", limit=5)
    comments = []
    for post in posts:
        comments.extend(reddit_client.get_comments(post))
    
    sentiment_scores = []
    for comment in comments:
        if symbol.lower() in comment.body.lower():
            blob = TextBlob(comment.body)
            sentiment_scores.append(blob.sentiment.polarity)
    
    if sentiment_scores:
        return sum(sentiment_scores) / len(sentiment_scores)
    else:
        return 0

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
        sentiment_score = analyze_sentiment_openai(symbol, description)
        reddit_sentiment = analyze_reddit_sentiment(symbol)
        
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

def detailed_analysis(symbol: str, data: Dict) -> Dict:
    logging.info(f"Performing detailed analysis for {symbol}")
    reddit_sentiment = analyze_reddit_sentiment(symbol)
    
    endpoint = f"{CMC_BASE_URL}cryptocurrency/info"
    parameters = {"symbol": symbol}
    try:
        response = rate_limiter.limited_request("GET", endpoint, headers=headers, params=parameters)
        response.raise_for_status()
        response_data = response.json()
        token_info = response_data.get("data", {}).get(symbol, {})
    except Exception as e:
        logging.error(f"Error fetching detailed information for {symbol}: {e}")
        token_info = {}
    
    return {
        "symbol": symbol,
        "price_change": data.get("percent_change_24h", "N/A"),
        "volume_change": data.get("volume_change_24h", "N/A"),
        "sentiment_score": data.get("sentiment_score", "N/A"),
        "reddit_sentiment": reddit_sentiment,
        "market_cap": data.get("market_cap", "N/A"),
        "circulating_supply": data.get("circulating_supply", "N/A"),
        "max_supply": token_info.get("max_supply", "N/A"),
        "total_supply": token_info.get("total_supply", "N/A"),
        "platform": token_info.get("platform", "N/A"),
        "category": token_info.get("category", "N/A"),
        "description": token_info.get("description", "N/A"),
        "date_added": token_info.get("date_added", "N/A"),
        "urls": token_info.get("urls", "N/A")
    }

class AltcoinTracker(cmd.Cmd):
    prompt = 'altcoin-tracker> '

    def __init__(self):
        super().__init__()
        self.current_data = []
        self.ethereum_tokens = []
        self.cmc_data = {}
        self.reddit_client = RedditClient()
        self.price_alerts = []
        self.trends = []
        self.reddit_analysis_results = None
        self.network = "all"
        self.chat_history = []
        self.show_welcome_menu()

    def show_welcome_menu(self):
        menu = """
        Welcome to the Enhanced CoinMarketCap Altcoin Tracker with AI Chat and Reddit Integration!
        ============================================================================================
        Type 'help' to see available commands.

        Available Commands:
        -------------------
        update   - Update the current listings and process data
        list     - List all current Ethereum-based tokens
        analyze  - Perform detailed analysis on a specific token
        reddit   - Browse Reddit posts and comments
        reddit_analyze - Analyze Reddit posts for cryptocurrency mentions and sentiment
        alerts   - Check for triggered price alerts
        trends   - Display current market trends
        news     - Fetch recent cryptocurrency news
        chat     - Start a chat session with the AI about scanned coins
        top_potential - Analyzing tokens for profit potential...
        exit     - Exit the program
        ============================================================================================
        """
        print(menu)

    def do_update(self, arg):
        """Update the current listings and process CoinMarketCap data"""
        print("Updating listings...")
        try:
            self.current_data = get_latest_listings(limit=500)
            if not self.current_data:
                print("Error: No data received from CoinMarketCap API.")
                return
            
            print(f"Fetched {len(self.current_data)} tokens.")
            self.process_cmc_data()
            self.filter_ethereum_tokens()  # Ensure Ethereum tokens are filtered after processing CMC data
            self.update_price_alerts()
            self.update_trend_analysis()
            print("Updated CoinMarketCap data, currency detection patterns, price alerts, and trend analysis")

            # Verify cmc_data contents
            logging.debug(f"CMC data after update: {list(self.cmc_data.keys())[:10]}")  # Log the first 10 token symbols

            # Check for specific tokens
            tokens_to_check = ['XAUt', 'USDe', 'BabyDoge', '0x0', 'Gomining']
            missing_tokens = self.check_token_presence(tokens_to_check)
            if missing_tokens:
                logging.error(f"Missing tokens in cmc_data: {missing_tokens}")

        except Exception as e:
            print(f"An error occurred while updating: {str(e)}")


    def check_token_presence(self, tokens):
        missing_tokens = []
        for token in tokens:
            if token.upper() not in self.cmc_data:
                missing_tokens.append(token)
        return missing_tokens

    def filter_ethereum_tokens(self):
        """Filter Ethereum-based tokens"""
        logging.debug("Filtering Ethereum-based tokens")
        self.ethereum_tokens = []
        for token in self.current_data:
            platform = token.get('platform')
            if platform and platform.get('name') == 'Ethereum':
                self.ethereum_tokens.append(token)
                logging.debug(f"Added Ethereum token: {token['symbol']}")

        # Include Ethereum itself
        eth_token = next((token for token in self.current_data if token['symbol'].upper() == 'ETH'), None)
        if eth_token:
            self.ethereum_tokens.append(eth_token)

        logging.debug(f"Filtered Ethereum tokens: {len(self.ethereum_tokens)}")
        if self.ethereum_tokens:
            logging.debug(f"Sample Ethereum tokens: {self.ethereum_tokens[:5]}")  # Log first 5 Ethereum tokens for inspection
        else:
            logging.warning("No Ethereum-based tokens found")


    def process_cmc_data(self):
        """Process CoinMarketCap data for efficient lookups"""
        self.cmc_data = {}
        for token in self.current_data:
            symbol = token['symbol'].upper()
            name = token['name'].upper()
            self.cmc_data[symbol] = token
            if name != symbol:
                self.cmc_data[name] = token
            logging.debug(f"Processed token: {symbol} ({name})")

        logging.debug(f"Processed CMC data. Total tokens: {len(self.cmc_data)}")
        logging.debug(f"Sample processed tokens: {list(self.cmc_data.keys())[:10]}")  # Log first 10 tokens for inspection


    def do_list(self, arg):
        """List all current Ethereum-based tokens"""
        if not self.ethereum_tokens:
            print("No Ethereum-based tokens found. Use 'update' to fetch the latest data.")
            return
        
        table_data = []
        for token in self.ethereum_tokens[:20]:  # Limit to top 20 for readability
            price = token['quote']['USD']['price']
            market_cap = token['quote']['USD']['market_cap']
            volume_24h = token['quote']['USD']['volume_24h']
            price_change_24h = token['quote']['USD']['percent_change_24h']
            
            table_data.append([
                token['symbol'],
                f"${price:.2f}",
                f"${market_cap:,.0f}",
                f"${volume_24h:,.0f}",
                f"{price_change_24h:.2f}%"
            ])
        
        headers = ["Symbol", "Price", "Market Cap", "24h Volume", "24h Change"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def do_analyze(self, arg):
        """Perform detailed analysis on a specific token"""
        symbol = arg.strip().upper()
        token = self.cmc_data.get(symbol)
        
        if token:
            try:
                analysis = detailed_analysis(token['symbol'], token['quote']['USD'])
                print(f"Detailed analysis for {token['symbol']}:")
                for key, value in analysis.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    elif isinstance(value, dict):
                        print(f"{key}:")
                        for subkey, subvalue in value.items():
                            print(f"  {subkey}: {subvalue}")
                    else:
                        print(f"{key}: {value}")
                
                if self.reddit_analysis_results:
                    mentions = self.reddit_analysis_results['currency_mentions'].get(symbol, 0)
                    print(f"\nReddit Analysis:")
                    print(f"Mentions: {mentions}")
                    
                    if mentions > 0:
                        relevant_text = ' '.join(re.findall(r'[^.!?]*{}[^.!?]*[.!?]'.format(symbol), self.reddit_analysis_results['analyzed_text'], re.IGNORECASE))
                        sentiment = TextBlob(relevant_text).sentiment.polarity
                        print(f"Reddit Sentiment: {sentiment:.4f}")
                    else:
                        print("Not enough mentions for sentiment analysis")
                else:
                    print("\nNo Reddit analysis available. Run 'reddit_analyze' first.")
                
                self.perform_technical_analysis(symbol)
                
                # New AI-powered comprehensive analysis
                comprehensive_analysis = self.generate_comprehensive_analysis(token, analysis, self.reddit_analysis_results)
                print("\nComprehensive AI Analysis:")
                print(comprehensive_analysis)
                    
            except Exception as e:
                print(f"An error occurred while analyzing the token: {e}")
        else:
            print(f"Token symbol '{symbol}' not found. Use 'list' to see available tokens.")

    def generate_comprehensive_analysis(self, token, analysis, reddit_analysis):
        """Generate a comprehensive analysis using AI"""
        context = f"""
        Token: {token['symbol']}
        Name: {token['name']}
        Price: ${token['quote']['USD']['price']}
        Market Cap: ${token['quote']['USD']['market_cap']}
        24h Change: {token['quote']['USD']['percent_change_24h']}%
        7d Change: {token['quote']['USD']['percent_change_7d']}%
        Volume 24h: ${token['quote']['USD']['volume_24h']}
        
        Description: {token.get('description', 'N/A')}
        
        Technical Analysis:
        RSI (14): {self.calculate_rsi(token['symbol'])}
        MACD: {self.calculate_macd(token['symbol'])}
        Bollinger Bands: {self.calculate_bollinger_bands(token['symbol'])}
        
        Reddit Mentions: {reddit_analysis['currency_mentions'].get(token['symbol'], 0) if reddit_analysis else 'N/A'}
        Reddit Sentiment: {analysis.get('reddit_sentiment', 'N/A')}
        
        Additional Info:
        {json.dumps(analysis, indent=2)}
        """
        
        prompt = f"""
        You extract surprising, insightful, and interesting information from text content, specifically focusing on cryptocurrency and finance. Your goal is to provide insights related to new altcoins, trending coins based on sentiment, and strategies for making money with cryptocurrency.

        Analyze the following information about the cryptocurrency {token['symbol']}:

        {context}

        STEPS:
        - Extract a summary of the content in 25 words, including who is presenting and the content being discussed into a section called SUMMARY.
        - Extract 20 to 50 of the most surprising, insightful, and/or interesting ideas from the input in a section called IDEAS:. If there are less than 50 then collect all of them. Make sure you extract at least 20.
        - Extract 10 to 20 of the best insights from the input and from a combination of the raw input and the IDEAS above into a section called INSIGHTS. These INSIGHTS should be fewer, more refined, more insightful, and more abstracted versions of the best ideas in the content.
        - Extract 15 to 30 of the most surprising, insightful, and/or interesting quotes from the input into a section called QUOTES:. Use the exact quote text from the input.
        - Extract 15 to 30 of the most practical and useful personal habits of the speakers, or mentioned by the speakers, in the content into a section called HABITS.
        - Extract 15 to 30 of the most surprising, insightful, and/or interesting valid facts about the greater world that were mentioned in the content into a section called FACTS:.
        - Extract all mentions of writing, art, tools, projects and other sources of inspiration mentioned by the speakers into a section called REFERENCES. This should include any and all references to something that the speaker mentioned.
        - Extract the most potent takeaway and recommendation into a section called ONE-SENTENCE TAKEAWAY. This should be a 15-word sentence that captures the most important essence of the content.
        - Extract the 15 to 30 of the most surprising, insightful, and/or interesting recommendations that can be collected from the content into a section called RECOMMENDATIONS.

        OUTPUT INSTRUCTIONS:
        - Only output Markdown.
        - Write the IDEAS bullets as exactly 15 words.
        - Write the RECOMMENDATIONS bullets as exactly 15 words.
        - Write the HABITS bullets as exactly 15 words.
        - Write the FACTS bullets as exactly 15 words.
        - Write the INSIGHTS bullets as exactly 15 words.
        - Extract at least 25 IDEAS from the content.
        - Extract at least 10 INSIGHTS from the content.
        - Extract at least 20 items for the other output sections.
        - Do not give warnings or notes; only output the requested sections.
        - You use bulleted lists for output, not numbered lists.
        - Do not repeat ideas, quotes, facts, or resources.
        - Do not start items with the same opening words.
        - Ensure you follow ALL these instructions when creating your output.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency analyst providing comprehensive insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.7,
            )
            
            analysis = response.choices[0].message.content.strip()
            return analysis
        except Exception as e:
            logging.error(f"Error generating comprehensive analysis: {str(e)}")
            return "Error generating comprehensive analysis."

    def do_reddit(self, arg):
        """Browse Reddit posts and comments"""
        subreddit = input("Enter subreddit name (default: CryptoCurrency): ") or "CryptoCurrency"
        sort = input("Sort by (hot/new/top, default: hot): ") or "hot"
        limit = int(input("Number of posts to fetch (default: 5): ") or "5")

        try:
            posts = self.reddit_client.get_posts(subreddit_name=subreddit, sort=sort, limit=limit)
            
            if not posts:
                print(f"No posts found in r/{subreddit}. The subreddit might not exist or could be private.")
                return

            for i, post in enumerate(posts, 1):
                print(f"\n{i}. {post.title} (Score: {post.score})")
                print(f"   Author: {post.author}")
                print(f"   Comments: {post.num_comments}")
                print(f"   URL: {post.url}")
            
            while True:
                choice = input("\nEnter the number of the post to view comments (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                try:
                    post_index = int(choice) - 1
                    if 0 <= post_index < len(posts):
                        selected_post = posts[post_index]
                        comments = self.reddit_client.get_comments(selected_post)
                        print(f"\nComments for: {selected_post.title}\n")
                        if not comments:
                            print("No comments found for this post.")
                        else:
                            for comment in comments:
                                print(f"{'  ' * comment.depth}[{comment.author}] (Score: {comment.score}): {comment.body[:100]}...")
                    else:
                        print("Invalid post number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'q' to quit.")
        except Exception as e:
            print(f"An error occurred while fetching Reddit data: {str(e)}")
            logging.error(f"Error in do_reddit: {str(e)}")

    def extract_currencies(self, text):
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
            if word in self.cmc_data and word not in common_words:
                token = self.cmc_data[word]
                market_cap = token['quote']['USD']['market_cap']
                if market_cap >= min_market_cap:
                    mentioned_currencies.append(token['symbol'])

        return list(dict.fromkeys(mentioned_currencies))  # Remove duplicates while preserving order


    def do_reddit_analyze(self, arg):
        """Analyze Reddit posts and comments for cryptocurrency mentions and sentiment, and generate summaries"""
        if not self.cmc_data:
            print("No CoinMarketCap data loaded. Running update...")
            self.do_update("")

        token_symbol = arg.strip().upper() if arg else None
        
        if token_symbol:
            query = f'"{token_symbol}"'  # Use quotes to search for exact matches
            subreddit = "CryptoCurrency"  # Default to r/CryptoCurrency for token-specific analysis
            print(f"Analyzing Reddit for mentions of {token_symbol}")
        else:
            query = input("Enter search query (default: cryptocurrency): ") or "cryptocurrency"
            subreddit = input("Enter subreddit name (default: CryptoCurrency): ") or "CryptoCurrency"
        
        num_posts = int(input("Number of posts to analyze (default: 10): ") or "10")
        num_comments = int(input("Number of top comments to analyze per post (default: 50): ") or "50")

        try:
            posts = self.reddit_client.search_posts(query=query, subreddit_name=subreddit, limit=num_posts)
            
            if not posts:
                print(f"No posts found for query '{query}' in r/{subreddit}. The subreddit might not exist or could be private.")
                return

            all_text = ""
            currency_mentions = Counter()
            summaries = []

            for post in posts:
                post_text = f"{post.title} {post.text}"
                all_text += post_text + " "
                
                currencies = self.extract_currencies(post_text)
                currency_mentions.update(currencies)
                
                comments = self.reddit_client.get_comments(post, max_comments=num_comments)
                comment_text = " ".join([comment.body for comment in comments])
                all_text += comment_text + " "
                
                comment_currencies = self.extract_currencies(comment_text)
                currency_mentions.update(comment_currencies)
                
                # Generate summary for this post and its comments
                summary = self.generate_summary(post.title, post.text, comment_text)
                summaries.append(summary)

            if token_symbol:
                print(f"\nAnalysis for {token_symbol}:")
                count = currency_mentions.get(token_symbol, 0)
                print(f"Mentions: {count}")
                if count > 0:
                    relevant_text = ' '.join(re.findall(r'[^.!?]*{}[^.!?]*[.!?]'.format(token_symbol), all_text, re.IGNORECASE))
                    sentiment = TextBlob(relevant_text).sentiment
                    print(f"Sentiment: Polarity {sentiment.polarity:.2f}, Subjectivity {sentiment.subjectivity:.2f}")
                else:
                    print("Not enough mentions for sentiment analysis")
            else:
                print("\nTop mentioned cryptocurrencies:")
                for currency, count in currency_mentions.most_common(10):
                    token_info = self.cmc_data.get(currency)
                    if token_info:
                        market_cap = token_info['quote']['USD']['market_cap']
                        price = token_info['quote']['USD']['price']
                        print(f"{currency}: {count} mentions (Market Cap: ${market_cap:,.0f}, Price: ${price:.2f})")
                    else:
                        print(f"{currency}: {count} mentions")

            sentiment = TextBlob(all_text).sentiment
            print(f"\nOverall sentiment:")
            print(f"Polarity: {sentiment.polarity:.2f} (-1 to 1, where -1 is very negative and 1 is very positive)")
            print(f"Subjectivity: {sentiment.subjectivity:.2f} (0 to 1, where 0 is very objective and 1 is very subjective)")

            print("\nConversation Summaries:")
            for i, summary in enumerate(summaries, 1):
                print(f"\nSummary {i}:")
                print(summary)

            self.reddit_analysis_results = {
                "currency_mentions": currency_mentions,
                "overall_sentiment": sentiment.polarity,
                "analyzed_text": all_text,
                "summaries": summaries
            }

            print("\nAnalysis complete. You can now use the 'analyze' command to get detailed information about specific currencies.")

        except Exception as e:
            print(f"An error occurred while analyzing Reddit data: {str(e)}")
            logging.error(f"Error in do_reddit_analyze: {str(e)}")


    def generate_summary(self, title, post_text, comments_text):
        """Generate a summary of a post and its comments using AI"""
        full_text = f"Title: {title}\n\nPost: {post_text}\n\nComments: {comments_text}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that summarizes Reddit discussions about cryptocurrencies. Provide a concise summary focusing on the main points, key arguments, and any notable information about cryptocurrencies mentioned."},
                    {"role": "user", "content": f"Please summarize the following Reddit post and its comments:\n\n{full_text}"}
                ],
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return "Error generating summary."

    def extract_key_topics(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        common_words = set([
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'this', 'that', 'with', 'will', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'would',
            'should', 'may', 'might', 'must', 'shall', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their',
            'his', 'her', 'its', 'our', 'your', 'my', 'about', 'if', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very'
        ])
        key_topics = [
            word for word, count in word_freq.most_common(50)
            if word not in common_words and len(word) > 3 and count > 2
        ]
        print("\nKey Topics:")
        print(", ".join(key_topics[:20]))

    def do_top_potential(self, arg):
        """Identify and display tokens with the highest profit potential"""
        try:
            num_top_tokens = int(arg) if arg else 5
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return
        
        print(f"Analyzing tokens for profit potential and displaying top {num_top_tokens} tokens...")
        
        potential_tokens = []
        for token in self.ethereum_tokens:
            score = self.calculate_potential_score(token)
            potential_tokens.append((token['symbol'], score))
        
        potential_tokens.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {num_top_tokens} tokens with highest profit potential:")
        for i, (symbol, score) in enumerate(potential_tokens[:num_top_tokens], 1):
            print(f"{i}. {symbol} (Score: {score:.2f})")
        
        print(f"\nUse 'analyze <symbol>' for detailed analysis of a specific token.")

    def calculate_potential_score(self, token):
        price_change_24h = token['quote']['USD']['percent_change_24h']
        price_change_7d = token['quote']['USD']['percent_change_7d']
        volume_change_24h = token['quote']['USD']['volume_change_24h']
        market_cap = token['quote']['USD']['market_cap']
        
        # Favor tokens with positive short-term momentum but not overbought
        price_momentum = price_change_24h * 0.7 + price_change_7d * 0.3
        
        # Favor increasing volume
        volume_factor = max(volume_change_24h, 0) / 100
        
        # Favor lower market cap tokens (more room for growth) but not too small
        market_cap_factor = 1 / (market_cap / 1e8)  # Adjusted to favor mid-cap tokens more
        
        # Consider Reddit mentions if available
        reddit_factor = 0
        if self.reddit_analysis_results:
            mentions = self.reddit_analysis_results['currency_mentions'].get(token['symbol'], 0)
            reddit_factor = min(mentions / 10, 1)  # Cap at 1 to avoid overweighting
        
        score = (price_momentum * 0.4 + volume_factor * 0.3 + market_cap_factor * 0.2 + reddit_factor * 0.1) * 100
        
        return max(score, 0)  # Ensure non-negative score

    def update_price_alerts(self):
        self.price_alerts = []
        for token in self.ethereum_tokens:
            current_price = token['quote']['USD']['price']
            alert_threshold = 0.1  # 10% movement
            self.price_alerts.append({
                'symbol': token['symbol'],
                'current_price': current_price,
                'upper_threshold': current_price * (1 + alert_threshold),
                'lower_threshold': current_price * (1 - alert_threshold)
            })

    def update_trend_analysis(self):
        self.trends = []
        for token in self.ethereum_tokens:
            price_change_24h = token['quote']['USD']['percent_change_24h']
            volume_change_24h = token['quote']['USD']['volume_change_24h']
            market_cap = token['quote']['USD']['market_cap']
            
            trend = {
                'symbol': token['symbol'],
                'price_change_24h': price_change_24h,
                'volume_change_24h': volume_change_24h,
                'market_cap': market_cap,
                'momentum': price_change_24h * volume_change_24h / 100  # Simple momentum calculation
            }
            self.trends.append(trend)

    def do_alerts(self, arg):
        """Check for any triggered price alerts"""
        try:
            triggered_alerts = []
            for alert in self.price_alerts:
                symbol = alert['symbol']
                token_data = self.cmc_data.get(symbol)

                if not token_data:
                    logging.error(f"Token data for {symbol} not found in cmc_data.")
                    continue

                quote = token_data.get('quote', {}).get('USD')
                if not quote:
                    logging.error(f"Quote data for {symbol} not found in cmc_data.")
                    continue

                current_price = quote.get('price')
                if current_price is None:
                    logging.error(f"Current price for {symbol} not found in cmc_data.")
                    continue

                if current_price >= alert['upper_threshold']:
                    triggered_alerts.append(f"{symbol} has increased by 10% or more. Current price: ${current_price:.4f}")
                elif current_price <= alert['lower_threshold']:
                    triggered_alerts.append(f"{symbol} has decreased by 10% or more. Current price: ${current_price:.4f}")

            if triggered_alerts:
                print("Triggered Alerts:")
                for alert in triggered_alerts:
                    print(alert)
            else:
                print("No price alerts triggered.")
        except Exception as e:
            logging.error(f"Error checking alerts: {e}")
            print(f"An error occurred while checking alerts: {e}")

    def do_trends(self, arg):
        """Display current market trends"""
        sorted_trends = sorted(self.trends, key=lambda x: abs(x['momentum']), reverse=True)
        print("Top 10 Tokens by Momentum:")
        headers = ["Symbol", "Price Change 24h", "Volume Change 24h", "Market Cap", "Momentum"]
        table_data = [
            [
                trend['symbol'],
                f"{trend['price_change_24h']:.2f}%",
                f"{trend['volume_change_24h']:.2f}%",
                f"${trend['market_cap']:,.0f}",
                f"{trend['momentum']:.2f}"
            ]
            for trend in sorted_trends[:10]
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def perform_technical_analysis(self, symbol):
        """Perform basic technical analysis"""
        token = self.cmc_data.get(symbol)
        if token:
            price = token['quote']['USD']['price']
            volume = token['quote']['USD']['volume_24h']
            market_cap = token['quote']['USD']['market_cap']
            
            print("\nTechnical Analysis:")
            print(f"RSI (14): {self.calculate_rsi(symbol):.2f}")
            print(f"MACD: {self.calculate_macd(symbol)}")
            print(f"Bollinger Bands: {self.calculate_bollinger_bands(symbol)}")

    def calculate_rsi(self, symbol):
        # Placeholder for RSI calculation
        return random.uniform(0, 100)

    def calculate_macd(self, symbol):
        # Placeholder for MACD calculation
        return f"12-day EMA: {random.uniform(45, 55):.2f}, 26-day EMA: {random.uniform(45, 55):.2f}, MACD Line: {random.uniform(-1, 1):.2f}, Signal Line: {random.uniform(-1, 1):.2f}"

    def calculate_bollinger_bands(self, symbol):
        # Placeholder for Bollinger Bands calculation
        middle = random.uniform(45, 55)
        return f"Upper Band: {middle + 2:.2f}, Middle Band: {middle:.2f}, Lower Band: {middle - 2:.2f}"

    def do_news(self, arg):
        """Fetch and display recent news articles related to cryptocurrencies"""
        news_articles = self.fetch_crypto_news()
        print("\nRecent Cryptocurrency News:")
        for article in news_articles[:5]:
            print(f"Title: {article['title']}")
            print(f"Source: {article['source']}")
            print(f"URL: {article['url']}")
            print(f"Published: {article['published_at']}")
            print("---")

    def fetch_crypto_news(self):
        # Placeholder for fetching news. In a real implementation, you'd use a news API.
        return [
            {"title": "Bitcoin Surges Past $40,000", "source": "CryptoNews", "url": "https://example.com/news1", "published_at": "2024-07-23 10:00:00"},
            {"title": "Ethereum 2.0 Upgrade Set for Next Month", "source": "BlockchainToday", "url": "https://example.com/news2", "published_at": "2024-07-23 09:30:00"},
            {"title": "New Cryptocurrency Regulations Proposed in EU", "source": "CryptoInsider", "url": "https://example.com/news3", "published_at": "2024-07-23 08:45:00"},
        ]

    def do_chat(self, arg):
        """Start a chat session with the AI about scanned coins"""
        print("Starting chat session. Type 'exit' to end the chat.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Ending chat session.")
                break
            
            response = self.get_ai_response(user_input)
            print(f"AI: {response}")

    def get_ai_response(self, user_input: str) -> str:
        coin_context = self.prepare_coin_context()
        reddit_context = self.prepare_reddit_context()
        
        messages = [
            {"role": "system", "content": f"You are an AI assistant specialized in cryptocurrency analysis. Use the following context about Ethereum-based tokens we've scanned: {coin_context}\n\nRecent Reddit discussions: {reddit_context}"},
            {"role": "user", "content": user_input}
        ]
        
        messages.extend(self.chat_history)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2800,
                n=1,
                stop=None,
                temperature=0.7,
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": ai_response})
            
            self.chat_history = self.chat_history[-10:]  # Keep only the last 5 exchanges
            
            return ai_response
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def prepare_coin_context(self) -> str:
        context = "Here's a summary of the top 5 Ethereum-based tokens we've scanned:\n"
        for token in self.ethereum_tokens[:5]:
            price = token['quote']['USD']['price']
            market_cap = token['quote']['USD']['market_cap']
            price_change_24h = token['quote']['USD']['percent_change_24h']
            context += f"{token['symbol']}: Price ${price:.2f}, Market Cap ${market_cap:,.0f}, 24h Change {price_change_24h:.2f}%\n"
        return context

    def prepare_reddit_context(self) -> str:
        if self.reddit_analysis_results:
            context = "Recent Reddit Analysis:\n"
            for currency, count in self.reddit_analysis_results['currency_mentions'].most_common(5):
                context += f"{currency}: {count} mentions\n"
            context += f"Overall sentiment: {self.reddit_analysis_results['overall_sentiment']:.2f}\n"
        else:
            context = "No recent Reddit analysis available.\n"
        return context

    def do_exit(self, arg):
        """Exit the program"""
        print("Thank you for using the AltcoinTracker. Goodbye!")
        return True

    def do_help(self, arg):
        """List available commands with "help" or detailed help with "help cmd"."""
        cmd.Cmd.do_help(self, arg)
        if not arg:
            print("\nAdditional Commands:")
            print("alerts  - Check for triggered price alerts")
            print("trends  - Display current market trends")
            print("news    - Fetch recent cryptocurrency news")
            print("chat    - Start a chat session with the AI about scanned coins")

def main():
    tracker = AltcoinTracker()
    tracker.do_update("")  # Initialize data on startup
    tracker.cmdloop()
    
if __name__ == "__main__":
    main()
