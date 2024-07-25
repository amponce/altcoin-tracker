# CoinMarketCap Altcoin Tracker

This project is an automated tool that monitors CoinMarketCap for potential buying opportunities in the altcoin market. It analyzes price changes, volume changes, and uses AI-powered sentiment analysis to identify promising altcoins.


![image](https://github.com/user-attachments/assets/6eb23e28-194f-4497-8835-783adf48a200)

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
   git clone https://github.com/amponce/altcoin-tracker.git
   cd altcoin-tracker
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
   REDDIT_USER_AGENT=python:com.App-12354.appname:v1.0 (by /u/username)
   ```

## Getting Reddit API Credentials

1. Go to [Reddit's App Preferences](https://www.reddit.com/prefs/apps).
2. Click on "Create App" or "Create Another App" at the bottom of the page.
3. Fill in the required fields:
   - **name**: Choose a name for your application.
   - **App type**: Select "script".
   - **description**: (optional) Provide a description for your app.
   - **about url**: (optional) Add a URL for more information about your app.
   - **redirect uri**: Set this to `http://localhost:8000` or any URL you prefer (this won't be used in this script).
   - **permissions**: (optional) Set the required permissions.
4. Click "Create app".
5. Your app will be created, and you will see `client_id` and `client_secret`. Copy these values and add them to your `.env` file along with a user agent.

## Configuration

You can adjust the tracker's settings in the `config.py` file. This includes:

- `CHECK_INTERVAL`: Time between checks (in seconds)
- `MAX_LISTINGS`: Number of cryptocurrencies to fetch from CoinMarketCap
- `OPPORTUNITY_CRITERIA`: Thresholds for identifying opportunities
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting for API calls

## Usage

Run the tracker with the following command:

```
python main.py
```

### Available Commands

- `u`: Update CoinMarketCap listings and process data
- `a`: Analyze Reddit posts and comments for cryptocurrency mentions and sentiment
- `d <symbol>`: Perform detailed analysis on a specific token
- `l`: List all current Ethereum-based tokens
- `s`: Show the current status of the tool
- `p <number>`: Display top <number> tokens with highest profit potential
- `c`: Start a chat session with the AI about scanned coins
- `al`: Check for any triggered price alerts
- `t`: Display current market trends
- `n`: Fetch and display recent crypto news
- `h`: Display detailed help information
- `q`: Quit the program

## Example Workflow

1. Use `u` to update data
2. Use `a` to analyze Reddit
3. Use `d <symbol>` to analyze specific tokens
4. Explore other commands as needed

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

## Acknowledgments

- CoinMarketCap for providing the cryptocurrency data API
- OpenAI for the sentiment analysis capabilities
- Reddit for providing the discussion platform and API access

---
