Here’s the updated README for your CoinMarketCap Altcoin Tracker project:

# CoinMarketCap Altcoin Tracker

This project is an automated tool that monitors CoinMarketCap for potential buying opportunities in the altcoin market. It analyzes price changes, volume changes, and uses AI-powered sentiment analysis to identify promising altcoins.

## Features

- Fetches latest cryptocurrency data from CoinMarketCap API
- Retrieves trending tokens from CoinMarketCap Community API
- Performs AI-powered sentiment analysis using OpenAI
- Analyzes trends based on price changes, volume changes, and sentiment
- Implements rate limiting to comply with API usage guidelines
- Provides terminal-based notifications of potential buying opportunities
- Browse and analyze Reddit posts for cryptocurrency mentions and sentiment
- Generate comprehensive AI-powered analysis for selected tokens
- Display current market trends and check for triggered price alerts

## Requirements

- Python 3.7+
- CoinMarketCap API key
- OpenAI API key
- Reddit API credentials

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/cmc-altcoin-tracker.git
   cd cmc-altcoin-tracker
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory with your API keys:
   ```
   CMC_API_KEY=your_coinmarketcap_api_key
   OPENAI_API_KEY=your_openai_api_key
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_reddit_user_agent
   ```

## Configuration

You can adjust the tracker's settings in the `config.py` file. This includes:

- `CHECK_INTERVAL`: Time between checks (in seconds)
- `MAX_LISTINGS`: Number of cryptocurrencies to fetch from CoinMarketCap
- `OPPORTUNITY_CRITERIA`: Thresholds for identifying opportunities
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting for API calls

## Usage

Run the tracker with the following command:

```
python altcoin_tracker.py
```

### Commands

Once the tracker is running, you can interact with it using the following commands:

- `update`: Update the current listings and process CoinMarketCap data
- `list`: List all current Ethereum-based tokens
- `analyze <symbol>`: Perform detailed analysis on a specific token
- `reddit`: Browse Reddit posts and comments
- `reddit_analyze`: Analyze Reddit posts for cryptocurrency mentions and sentiment
- `alerts`: Check for triggered price alerts
- `trends`: Display current market trends
- `news`: Fetch recent cryptocurrency news
- `chat`: Start a chat session with the AI about scanned coins
- `top_potential <number>`: Identify and display tokens with the highest profit potential
- `exit`: Exit the program

### Example Interactions

- "Analyze the top 10 tokens for profit potential"
- "Get the latest price alerts for significant changes"
- "Browse recent Reddit discussions about cryptocurrencies"
- "Fetch and display the latest cryptocurrency news"

## Output

For each potential opportunity, the tracker will display:

- Cryptocurrency symbol
- Price change (percentage)
- Volume change (percentage)
- Sentiment score (-1 to 1)
- Whether the token is currently trending

## Disclaimer

This tool is for informational purposes only. It does not constitute financial advice. Always do your own research before making any investment decisions.

## License

[MIT License](LICENSE)

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/cmc-altcoin-tracker/issues) if you want to contribute.

## Author

[Your Name]

## Acknowledgments

- CoinMarketCap for providing the cryptocurrency data API
- OpenAI for the sentiment analysis capabilities
- Reddit for providing the discussion platform and API access

---
