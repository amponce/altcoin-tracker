import logging
import requests
from requests.exceptions import RequestException
from config import CMC_BASE_URL, HEADERS, MAX_REQUESTS_PER_MINUTE
from utils.rate_limiter import RateLimiter

rate_limiter = RateLimiter()

def get_latest_listings(limit: int = 500):
    logging.debug(f"Fetching latest listings (limit: {limit})")
    endpoint = f"{CMC_BASE_URL}cryptocurrency/listings/latest"
    parameters = {
        "start": "1",
        "limit": str(limit),
        "convert": "USD",
        "aux": "platform"
    }
    
    try:
        response = rate_limiter.limited_request("GET", endpoint, headers=HEADERS, params=parameters)
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
