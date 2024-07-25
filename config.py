import os
from dotenv import load_dotenv

load_dotenv()

CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1/"
MAX_LISTINGS = 500
MAX_REQUESTS_PER_MINUTE = 30
OPPORTUNITY_CRITERIA = {
    'price_change_threshold': 0.05,
    'volume_change_threshold': 0.1,
    'sentiment_threshold': 0.6
}

HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": os.getenv('CMC_API_KEY'),
}

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
