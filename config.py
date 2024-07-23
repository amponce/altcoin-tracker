import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# CoinMarketCap API settings
CMC_API_KEY = os.getenv('CMC_API_KEY')
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1/"

# OpenAI API settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# REDDIT API settings
REDDIT_CLIENT_ID=os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET=os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT=os.getenv('REDDIT_USER_AGENT')

# Twitter API settings (not used in this version, but prepared for future use)
BEARER_TOKEN=os.getenv('BEARER_TOKEN')
TW_CLIENT_ID=os.getenv('TW_CLIENT_ID')
TW_CLIENT_SECRET=os.getenv('TW_CLIENT_SECRET')
TW_CONSUMER_KEY = os.getenv('TW_CONSUMER_KEY')
TW_CONSUMER_SECRET = os.getenv('TW_CONSUMER_SECRET')
TW_ACCESS_TOKEN = os.getenv('TW_ACCESS_TOKEN')
TW_ACCESS_TOKEN_SECRET = os.getenv('TW_ACCESS_TOKEN_SECRET')

# App settings
CHECK_INTERVAL = 300  # 5 minutes
MAX_LISTINGS = 100
OPPORTUNITY_CRITERIA = {
    'price_change_threshold': 0.03,
    'volume_change_threshold': 0.05,
    'sentiment_threshold': 0.2
}

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 30
