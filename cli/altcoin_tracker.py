import cmd
import logging
import re
import random
import json
from ai.openai_client import OpenAIClient
from clients.reddit_client import RedditClient
from clients.coinmarketcap_client import get_latest_listings
from services.trend_analysis import analyze_trends
from services.reddit_analysis import analyze_reddit

from services.sentiment_analysis import get_openai_sentiment, analyze_reddit_sentiment
from config import HEADERS
from models import Post
from textblob import TextBlob
from tabulate import tabulate
from colorama import init, Fore, Back, Style
from ascii_art import WELCOME_ART
from utils.progress import progress_bar
from utils.rate_limiter import RateLimiter
from config import CMC_BASE_URL

rate_limiter = RateLimiter()


class AltcoinTracker(cmd.Cmd):
    prompt = Fore.GREEN + 'altcoin-tracker> ' + Style.RESET_ALL

    def __init__(self):
        super().__init__()
        init(autoreset=True)  # Initialize colorama
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
        self.openai_client = OpenAIClient()
        self.prompt = f"{Fore.GREEN}altcoin-tracker>{Fore.RESET} "

    def do_q(self, arg):
        """Alias for exit"""
        return self.do_exit(arg)

    def do_tour(self, arg):
        """Take an interactive tour of the application"""
        print(f"{Fore.CYAN}Welcome to the AltcoinTracker tour! Let's explore the main features:{Style.RESET_ALL}")
        
        steps = [
            ("u", "First, let's update our data. This fetches the latest information from CoinMarketCap."),
            ("l", "Now, let's list the top Ethereum-based tokens."),
            ("t", "Let's check the current market trends."),
            ("b", "We can browse Reddit posts related to cryptocurrencies."),
            ("a", "Let's analyze Reddit sentiment for a popular coin like 'BTC'."),
            ("n", "Finally, let's check the latest crypto news.")
        ]
        
        for cmd, description in steps:
            input(f"\n{Fore.YELLOW}Press Enter to run the '{cmd}' command. {description}{Style.RESET_ALL}")
            getattr(self, f"do_{cmd}")(arg="BTC" if cmd == "a" else "")
        
        print(f"\n{Fore.GREEN}Tour completed! You can now explore other commands using 'help'.{Style.RESET_ALL}")

    def show_welcome_menu(self):
        print(WELCOME_ART)
        print(f"{Fore.CYAN + Style.BRIGHT}Welcome to the Enhanced CoinMarketCap Altcoin Tracker!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Type 'help' to see available commands.{Style.RESET_ALL}")


    def display_help_menu(self):
        help_items = [
            ("u", "Update CMC Data"),
            ("a", "Analyze Reddit"),
            ("d <symbol>", "Detailed Analysis"),
            ("l", "List Tokens"),
            ("s", "Status"),
            ("p", "Profit Potential"),
            ("t", "Trends"),
            ("h", "Full Help"),
            ("q", "Quit")
        ]
        help_text = " | ".join([f"{Fore.GREEN}{cmd}{Fore.RESET}:{desc}" for cmd, desc in help_items])
        print(f"\n{help_text}\n")



    def cmdloop(self, intro=None):
        print(self.intro)
        while True:
            try:
                self.display_help_menu()
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    if self.use_rawinput:
                        try:
                            line = input(self.prompt)
                        except EOFError:
                            line = 'EOF'
                    else:
                        self.stdout.write(self.prompt)
                        self.stdout.flush()
                        line = self.stdin.readline()
                        if not len(line):
                            line = 'EOF'
                        else:
                            line = line.rstrip('\r\n')
                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
                if stop:
                    break
            except KeyboardInterrupt:
                print("^C")

    def do_u(self, arg):
        """Update the current listings and process CoinMarketCap data"""
        print("Updating listings...")
        try:
            self.current_data = get_latest_listings(limit=500)
            if not self.current_data:
                print("Error: No data received from CoinMarketCap API.")
                return
            
            print(f"Fetched {len(self.current_data)} tokens.")
            for i in range(100):
                progress_bar(i+1, 100, prefix='Processing:', suffix='Complete', length=50)
                if i == 24:
                    self.process_cmc_data()
                elif i == 49:
                    self.filter_ethereum_tokens()
                elif i == 74:
                    self.update_price_alerts()
                elif i == 99:
                    self.update_trend_analysis()
            print("\nUpdated CoinMarketCap data, currency detection patterns, price alerts, and trend analysis")

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

    def update_price_alerts(self):
        """Update the price alerts based on the latest data"""
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
        """Update the trend analysis based on the latest data"""
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

    def detailed_analysis(self, symbol):
        """Perform detailed analysis on a specific token"""
        symbol = symbol.upper()
        token = self.cmc_data.get(symbol)
        
        if not token:
            print(f"Token symbol '{symbol}' not found. Use 'l' to see available tokens.")
            return None, None

        try:
            print(f"Detailed analysis for {token['symbol']}:")
            
            # Price information
            price = token['quote']['USD']['price']
            market_cap = token['quote']['USD']['market_cap']
            volume_24h = token['quote']['USD']['volume_24h']
            percent_change_24h = token['quote']['USD']['percent_change_24h']
            percent_change_7d = token['quote']['USD']['percent_change_7d']
            
            print(f"Price: ${price:.4f}")
            print(f"Market Cap: ${market_cap:,.0f}")
            print(f"24h Volume: ${volume_24h:,.0f}")
            print(f"24h Change: {percent_change_24h:.2f}%")
            print(f"7d Change: {percent_change_7d:.2f}%")
            
            # Technical Analysis
            print("\nTechnical Analysis:")
            rsi = self.calculate_rsi(symbol)
            macd = self.calculate_macd(symbol)
            bollinger_bands = self.calculate_bollinger_bands(symbol)
            print(f"RSI (14): {rsi:.2f}")
            print(f"MACD: {macd}")
            print(f"Bollinger Bands: {bollinger_bands}")
            
            # Sentiment Analysis
            openai_client = OpenAIClient()
            sentiment_score = openai_client.analyze_sentiment_openai(token['symbol'], token.get('description', ''))
            print(f"\nSentiment Score: {sentiment_score:.2f}")
            
            # Reddit Analysis
            reddit_mentions = 0
            reddit_sentiment = None
            reddit_summaries = []
            if self.reddit_analysis_results:
                reddit_mentions = self.reddit_analysis_results['currency_mentions'].get(symbol, 0)
                print(f"\nReddit Analysis:")
                print(f"Mentions: {reddit_mentions}")
                
                if reddit_mentions > 0:
                    relevant_text = ' '.join(re.findall(r'[^.!?]*{}[^.!?]*[.!?]'.format(symbol), self.reddit_analysis_results['analyzed_text'], re.IGNORECASE))
                    reddit_sentiment = TextBlob(relevant_text).sentiment.polarity
                    print(f"Reddit Sentiment: {reddit_sentiment:.4f}")
                    reddit_summaries = self.reddit_analysis_results.get('summaries', [])
                    if reddit_summaries:
                        print("\nReddit Discussion Summaries:")
                        for i, summary in enumerate(reddit_summaries, 1):
                            print(f"Summary {i}: {summary}")
                    else:
                        print("No relevant Reddit discussions found for this token.")
                else:
                    print("Not enough mentions for sentiment analysis")
            else:
                print("\nNo Reddit analysis available. Use the 'a' command to run Reddit analysis first.")
            
            # Prepare analysis dictionary for comprehensive analysis
            analysis = {
                'symbol': token['symbol'],
                'name': token['name'],
                'price': price,
                'market_cap': market_cap,
                'volume_24h': volume_24h,
                'percent_change_24h': percent_change_24h,
                'percent_change_7d': percent_change_7d,
                'description': token.get('description', 'N/A'),
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bollinger_bands,
                'sentiment_score': sentiment_score,
                'reddit_mentions': reddit_mentions,
                'reddit_sentiment': reddit_sentiment if reddit_sentiment is not None else 'N/A',
                'reddit_summaries': reddit_summaries
            }
            
            return analysis, token
            
        except Exception as e:
            print(f"An error occurred while analyzing the token: {str(e)}")
            return None, None

    def do_l(self, arg):
        """List all current Ethereum-based tokens"""
        if not self.ethereum_tokens:
            print("No Ethereum-based tokens found. Use 'u' to fetch the latest data.")
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
        
    def do_s(self, arg):
        """Show the current status of the tool"""
        print(f"{Fore.CYAN}Current Status:{Fore.RESET}")
        print(f"CoinMarketCap data: {'Loaded' if self.cmc_data else 'Not loaded'}")
        print(f"Number of tracked tokens: {len(self.cmc_data)}")
        print(f"Number of Ethereum tokens: {len(self.ethereum_tokens)}")
        print(f"Reddit analysis: {'Available' if self.reddit_analysis_results else 'Not available'}")
        print(f"Price alerts set: {len(self.price_alerts)}")
        print(f"Trend analysis: {'Available' if self.trends else 'Not available'}")
        
        print(f"\n{Fore.YELLOW}Next steps:{Fore.RESET}")
        if not self.cmc_data:
            print("- Use 'u' to update CoinMarketCap data")
        elif not self.reddit_analysis_results:
            print("- Use 'a' to run Reddit analysis (recommended)")
        else:
            print("- Use 'd <symbol>' to get detailed analysis of a specific token")
            print("- Use 'p' to see tokens with high profit potential")
            print("- Use 't' to view current market trends")
        print("- Use 'h' to see all available commands")
  
    def do_p(self, arg):
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

    def do_d(self, arg):
        """Perform detailed analysis on a specific token"""
        if not self.cmc_data:
            print("Please run 'u' to update CoinMarketCap data first.")
            return
        if not self.reddit_analysis_results:
            print("Note: Reddit analysis hasn't been run. Use 'a' to analyze Reddit data.")
        
        symbol = arg.strip().upper()
        analysis, token = self.detailed_analysis(symbol)
        
        if analysis and token:
            # Generate comprehensive analysis
            comprehensive_analysis = self.generate_comprehensive_analysis(token, analysis, self.reddit_analysis_results)
            print("\nComprehensive AI Analysis:")
            print(comprehensive_analysis)
        else:
            print(f"Could not perform analysis on token '{symbol}'.")

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
        RSI (14): {analysis.get('rsi', 'N/A')}
        MACD: {analysis.get('macd', 'N/A')}
        Bollinger Bands: {analysis.get('bollinger_bands', 'N/A')}
        
        Reddit Mentions: {reddit_analysis['currency_mentions'].get(token['symbol'], 0) if reddit_analysis else 'N/A'}
        Reddit Sentiment: {analysis.get('reddit_sentiment', 'N/A')}
        
        Additional Info:
        {json.dumps(analysis, indent=2, default=str)}
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
            response = self.openai_client.client.chat.completions.create(
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

    def do_b(self, arg):
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
            logging.error(f"Error in do_b: {str(e)}")

    def do_a(self, arg):
        """Analyze Reddit posts and comments for cryptocurrency mentions and sentiment, and generate summaries"""
        if not self.cmc_data:
            print("No CoinMarketCap data loaded. Running update...")
            self.do_u("")

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
            self.reddit_analysis_results = analyze_reddit(self.reddit_client, query, subreddit, num_posts, num_comments, self.cmc_data)

            if token_symbol:
                print(f"\nAnalysis for {token_symbol}:")
                count = self.reddit_analysis_results["currency_mentions"].get(token_symbol, 0)
                print(f"Mentions: {count}")
                if count > 0:
                    relevant_text = ' '.join(re.findall(r'[^.!?]*{}[^.!?]*[.!?]'.format(token_symbol), self.reddit_analysis_results["analyzed_text"], re.IGNORECASE))
                    sentiment = TextBlob(relevant_text).sentiment
                    print(f"Sentiment: Polarity {sentiment.polarity:.2f}, Subjectivity {sentiment.subjectivity:.2f}")
                else:
                    print("Not enough mentions for sentiment analysis")
            else:
                print("\nTop mentioned cryptocurrencies:")
                for currency, count in self.reddit_analysis_results["currency_mentions"].most_common(10):
                    token_info = self.cmc_data.get(currency)
                    if token_info:
                        market_cap = token_info['quote']['USD']['market_cap']
                        price = token_info['quote']['USD']['price']
                        print(f"{currency}: {count} mentions (Market Cap: ${market_cap:,.0f}, Price: ${price:.2f})")
                    else:
                        print(f"{currency}: {count} mentions")

            sentiment = TextBlob(self.reddit_analysis_results["analyzed_text"]).sentiment
            print(f"\nOverall sentiment:")
            print(f"Polarity: {sentiment.polarity:.2f} (-1 to 1, where -1 is very negative and 1 is very positive)")
            print(f"Subjectivity: {sentiment.subjectivity:.2f} (0 to 1, where 0 is very objective and 1 is very subjective)")

            print("\nConversation Summaries:")
            for i, summary in enumerate(self.reddit_analysis_results["summaries"], 1):
                print(f"\nSummary {i}:")
                print(summary)

            print("\nAnalysis complete. You can now use the 'analyze' command to get detailed information about specific currencies.")

        except Exception as e:
            print(f"An error occurred while analyzing Reddit data: {str(e)}")
            logging.error(f"Error in do_reddit_analyze: {str(e)}")

    def do_al(self, arg):
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

    def do_t(self, arg):
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

    def do_n(self, arg):
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

    def do_c(self, arg):
        """Start a chat session with the AI about scanned coins or just for casual conversation"""
        print("Starting chat session. Type 'exit' to end the chat.")
        print("CryptoChat: Hey there! I'm CryptoChat, your friendly cryptocurrency assistant. We can chat about crypto, or anything else you'd like. What's on your mind?")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("CryptoChat: It was great chatting with you! Take care and come back anytime.")
                print("Ending chat session.")
                break
            
            response = self.get_ai_response(user_input)
            print(f"CryptoChat: {response}")

    def get_ai_response(self, user_input: str) -> str:
        coin_context = self.prepare_coin_context()
        reddit_context = self.prepare_reddit_context()
        
        system_message = f"""You are a friendly and knowledgeable AI assistant named CryptoChat. 
        You're capable of engaging in casual conversation as well as providing detailed cryptocurrency analysis.
        You have access to the following information about Ethereum-based tokens:
        {coin_context}
        
        And recent Reddit discussions:
        {reddit_context}
        
        Maintain a natural conversational flow. If the user wants to chat casually, respond in a friendly manner.
        If they ask about cryptocurrencies, provide informed responses based on the available data.
        Be concise but informative, and don't be afraid to show a bit of personality!
        """
        
        messages = [
            {"role": "system", "content": system_message},
        ]
        
        # Add the chat history
        messages.extend(self.chat_history)
        
        # Add the new user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2800,
                n=1,
                stop=None,
                temperature=0.7,
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Keep only the last 10 messages (5 exchanges)
            self.chat_history = self.chat_history[-10:]
            
            return ai_response
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def prepare_coin_context(self) -> str:
        context = "Top 5 Ethereum-based tokens:\n"
        for token in self.ethereum_tokens[:5]:
            price = token['quote']['USD']['price']
            market_cap = token['quote']['USD']['market_cap']
            price_change_24h = token['quote']['USD']['percent_change_24h']
            context += f"- {token['symbol']} ({token['name']}): Price ${price:.2f}, Market Cap ${market_cap:,.0f}, 24h Change {price_change_24h:.2f}%\n"
        return context

    def prepare_reddit_context(self) -> str:
        if self.reddit_analysis_results:
            context = "Recent Reddit Analysis:\n"
            for currency, count in self.reddit_analysis_results['currency_mentions'].most_common(5):
                context += f"- {currency}: {count} mentions\n"
            context += f"Overall sentiment: {self.reddit_analysis_results['overall_sentiment']:.2f} (-1 to 1 scale)\n"
        else:
            context = "No recent Reddit analysis available.\n"
        return context

    def do_exit(self, arg):
        """Exit the program"""
        print("Thank you for using the AltcoinTracker. Goodbye!")
        return True

    def do_h(self, arg):
        """Display detailed help information"""
        print(f"\n{Fore.CYAN}Detailed Command Information:{Fore.RESET}")
        commands = {
            "u": "Update CoinMarketCap listings and process data",
            "a": "Analyze Reddit posts and comments for cryptocurrency mentions and sentiment",  # Changed from "ra" to "a"
            "d <symbol>": "Perform detailed analysis on a specific token",
            "l": "List all current Ethereum-based tokens",
            "s": "Show the current status of the tool",
            "p <number>": "Display top <number> tokens with highest profit potential",
            "c": "Start a chat session with the AI about scanned coins",
            "al": "Check for any triggered price alerts",
            "t": "Display current market trends",
            "n": "Fetch and display recent crypto news",
            "h": "Display this detailed help information",
            "q": "Quit the program"
        }
        for cmd, desc in commands.items():
            print(f"{Fore.GREEN}{cmd:<15}{Fore.RESET} - {desc}")
        print(f"\n{Fore.YELLOW}Recommended workflow:{Fore.RESET}")
        print("1. Use 'u' to update data")
        print("2. Use 'a' to analyze Reddit")  # Changed from "ra" to "a"
        print("3. Use 'd <symbol>' to analyze specific tokens")
        print("4. Explore other commands as needed")

def main():
    init(autoreset=True)  # Initialize colorama
    tracker = AltcoinTracker()
    tracker.do_u("")  # Initialize data on startup
    tracker.cmdloop()
    
if __name__ == "__main__":
    main()
