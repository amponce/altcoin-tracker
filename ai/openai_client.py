import logging
from typing import Tuple
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MODEL
class OpenAIClient:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def analyze_sentiment_openai(self, symbol: str, description: str) -> float:
        logging.debug(f"Analyzing sentiment for {symbol}")
        try:
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the sentiment of the given cryptocurrency description and respond with a single number between -1 (very negative) and 1 (very positive)."},
                    {"role": "user", "content": f"Cryptocurrency: {symbol}\nDescription: {description}\n\nSentiment score:"}
                ],
                max_tokens=1,
                temperature=0.5
            )
            
            response_content = response.choices[0].message.content.strip()
            logging.debug(f"Raw sentiment score response: {response_content}")
            
            try:
                sentiment_score = float(response_content)
                return max(-1, min(1, sentiment_score))
            except ValueError:
                logging.error(f"Error converting sentiment score to float: {response_content}")
                return 0.0
        except Exception as e:
            logging.error(f"Error analyzing sentiment with OpenAI: {e}")
            raise

    def generate_summary(self, title: str, post_text: str, comments_text: str) -> Tuple[str, bool]:
        """Generate a summary of a post and its comments using AI"""
        full_text = f"Title: {title}\n\nPost: {post_text}\n\nComments: {comments_text}"
        
        try:
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that summarizes Reddit discussions about cryptocurrencies. Provide a concise summary focusing on the main points, key arguments, and any notable information about cryptocurrencies mentioned."},
                    {"role": "user", "content": f"Please summarize the following Reddit post and its comments:\n\n{full_text}"}
                ],
                max_tokens=150,
                temperature=0.7,
            )
            
            summary = response.choices[0].message.content.strip()
            return summary, True
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return "Error generating summary.", False
