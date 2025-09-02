#!/usr/bin/env python3
"""
Financial Sentiment Analysis Module
Integrates FinBERT and news sentiment for stock prediction
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers (optional dependency)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# Try to import additional libraries for news scraping
try:
    import yfinance as yf
    from newsapi import NewsApiClient
    NEWS_API_AVAILABLE = True
except ImportError:
    print("⚠️  News APIs not available. Install with: pip install yfinance python-newsapi")
    NEWS_API_AVAILABLE = False

class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text
    Uses ProsusAI/finbert model for financial sentiment classification
    """
    
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for FinBERT")
        
        print("📊 Loading FinBERT model...")
        
        # Load pre-trained FinBERT model
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Create sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        
        print("✅ FinBERT model loaded successfully")
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment of financial texts
        
        Args:
            texts: List of financial texts to analyze
            
        Returns:
            List of sentiment dictionaries with scores
        """
        results = []
        
        for text in texts:
            try:
                # Clean and truncate text
                clean_text = self._clean_text(text)
                
                # Get sentiment scores
                scores = self.sentiment_pipeline(clean_text)[0]
                
                # Convert to standardized format
                sentiment_dict = {
                    'text': text,
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 0.0,
                    'compound': 0.0
                }
                
                for score in scores:
                    label = score['label'].lower()
                    sentiment_dict[label] = score['score']
                
                # Calculate compound score
                sentiment_dict['compound'] = (
                    sentiment_dict['positive'] - sentiment_dict['negative']
                )
                
                results.append(sentiment_dict)
                
            except Exception as e:
                print(f"⚠️  Error analyzing text: {str(e)}")
                # Return neutral sentiment on error
                results.append({
                    'text': text,
                    'positive': 0.33,
                    'negative': 0.33,
                    'neutral': 0.34,
                    'compound': 0.0
                })
        
        return results
    
    def _clean_text(self, text: str, max_length: int = 512) -> str:
        """Clean and truncate text for FinBERT"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate to max length (FinBERT has 512 token limit)
        if len(text) > max_length:
            text = text[:max_length]
        
        return text

class NewsDataCollector:
    """
    Collect financial news data from various sources
    """
    
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key
        
        if news_api_key and NEWS_API_AVAILABLE:
            self.newsapi = NewsApiClient(api_key=news_api_key)
        else:
            self.newsapi = None
            print("⚠️  NewsAPI key not provided or NewsAPI not available")
    
    def get_stock_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Get recent news for a specific stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days of news to fetch
            
        Returns:
            List of news articles
        """
        news_articles = []
        
        try:
            # Try multiple sources
            sources = [
                self._get_newsapi_articles,
                self._get_yahoo_news,
                self._get_free_news_sources
            ]
            
            for source_func in sources:
                try:
                    articles = source_func(symbol, days)
                    news_articles.extend(articles)
                except Exception as e:
                    print(f"⚠️  Error from news source: {str(e)}")
                    continue
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_articles = []
            
            for article in news_articles:
                title = article.get('title', '').lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)
            
            print(f"📰 Collected {len(unique_articles)} unique news articles for {symbol}")
            return unique_articles
            
        except Exception as e:
            print(f"❌ Error collecting news: {str(e)}")
            return []
    
    def _get_newsapi_articles(self, symbol: str, days: int) -> List[Dict]:
        """Get articles from NewsAPI"""
        if not self.newsapi:
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Search for articles
        articles = self.newsapi.get_everything(
            q=f"{symbol} stock",
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=50
        )
        
        news_list = []
        for article in articles['articles']:
            news_list.append({
                'title': article['title'],
                'description': article['description'],
                'content': article['content'] or article['description'],
                'published_at': article['publishedAt'],
                'source': article['source']['name'],
                'url': article['url']
            })
        
        return news_list
    
    def _get_yahoo_news(self, symbol: str, days: int) -> List[Dict]:
        """Get news from Yahoo Finance"""
        if not NEWS_API_AVAILABLE:
            return []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            news_list = []
            for item in news[:20]:  # Limit to recent articles
                news_list.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'content': item.get('summary', ''),
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'source': item.get('publisher', ''),
                    'url': item.get('link', '')
                })
            
            return news_list
            
        except Exception as e:
            print(f"⚠️  Error getting Yahoo news: {str(e)}")
            return []
    
    def _get_free_news_sources(self, symbol: str, days: int) -> List[Dict]:
        """Get news from free sources (placeholder)"""
        # This is a placeholder for free news sources
        # In practice, you might scrape financial websites or use free APIs
        return []

class SentimentFeatureEngineer:
    """
    Engineer sentiment-based features for stock prediction
    """
    
    def __init__(self, sentiment_analyzer: FinBERTSentimentAnalyzer = None):
        self.analyzer = sentiment_analyzer or FinBERTSentimentAnalyzer()
    
    def create_sentiment_features(self, news_data: List[Dict], stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment features aligned with stock price data
        
        Args:
            news_data: List of news articles with sentiment
            stock_data: DataFrame with stock price data
            
        Returns:
            DataFrame with sentiment features
        """
        print("🔧 Engineering sentiment features...")
        
        # Convert stock data dates
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        # Create sentiment DataFrame
        sentiment_df = self._create_daily_sentiment_scores(news_data, stock_data['date'])
        
        # Merge with stock data
        merged_df = stock_data.merge(sentiment_df, on='date', how='left')
        
        # Forward fill missing sentiment values
        sentiment_columns = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 
                           'sentiment_compound', 'news_volume', 'sentiment_volatility']
        merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(method='ffill')
        merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(0.0)
        
        # Create additional sentiment features
        merged_df = self._add_sentiment_indicators(merged_df)
        
        print(f"✅ Created {len(sentiment_columns) + 3} sentiment features")
        return merged_df
    
    def _create_daily_sentiment_scores(self, news_data: List[Dict], dates: pd.Series) -> pd.DataFrame:
        """Create daily aggregated sentiment scores"""
        
        # Analyze sentiment for all news
        news_texts = [article.get('content', article.get('title', '')) for article in news_data]
        sentiment_results = self.analyzer.analyze_sentiment(news_texts)
        
        # Combine news data with sentiment
        for i, article in enumerate(news_data):
            if i < len(sentiment_results):
                article.update(sentiment_results[i])
        
        # Convert to DataFrame
        df = pd.DataFrame(news_data)
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['date'] = df['published_at'].dt.date
        
        # Aggregate by date
        daily_sentiment = df.groupby('date').agg({
            'positive': 'mean',
            'negative': 'mean', 
            'neutral': 'mean',
            'compound': 'mean',
            'title': 'count'  # News volume
        }).reset_index()
        
        # Rename columns
        daily_sentiment.rename(columns={
            'positive': 'sentiment_positive',
            'negative': 'sentiment_negative',
            'neutral': 'sentiment_neutral',
            'compound': 'sentiment_compound',
            'title': 'news_volume'
        }, inplace=True)
        
        # Calculate sentiment volatility (standard deviation of compound scores)
        sentiment_volatility = df.groupby('date')['compound'].std().reset_index()
        sentiment_volatility.rename(columns={'compound': 'sentiment_volatility'}, inplace=True)
        
        # Merge volatility
        daily_sentiment = daily_sentiment.merge(sentiment_volatility, on='date', how='left')
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_volatility'].fillna(0.0)
        
        # Convert date back to datetime
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        return daily_sentiment
    
    def _add_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators based on sentiment"""
        
        # Sentiment moving averages
        df['sentiment_sma_5'] = df['sentiment_compound'].rolling(window=5).mean()
        df['sentiment_sma_20'] = df['sentiment_compound'].rolling(window=20).mean()
        
        # Sentiment momentum
        df['sentiment_momentum'] = df['sentiment_compound'] - df['sentiment_compound'].shift(5)
        
        # Sentiment RSI-like indicator
        sentiment_change = df['sentiment_compound'].diff()
        gain = sentiment_change.where(sentiment_change > 0, 0.0)
        loss = -sentiment_change.where(sentiment_change < 0, 0.0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        df['sentiment_rsi'] = 100 - (100 / (1 + rs))
        
        return df

class SentimentStockPredictor:
    """
    Main class that combines sentiment analysis with stock prediction
    """
    
    def __init__(self, news_api_key: str = None):
        self.sentiment_analyzer = FinBERTSentimentAnalyzer() if TRANSFORMERS_AVAILABLE else None
        self.news_collector = NewsDataCollector(news_api_key)
        self.feature_engineer = SentimentFeatureEngineer(self.sentiment_analyzer)
    
    def enhance_stock_data_with_sentiment(self, stock_data: pd.DataFrame, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Enhance stock price data with sentiment features
        
        Args:
            stock_data: DataFrame with stock price data
            symbol: Stock symbol
            days: Number of days of news to collect
            
        Returns:
            Enhanced DataFrame with sentiment features
        """
        if not self.sentiment_analyzer:
            print("⚠️  Sentiment analysis not available, returning original data")
            return stock_data
        
        print(f"🔍 Enhancing {symbol} stock data with sentiment analysis...")
        
        # Collect news data
        news_data = self.news_collector.get_stock_news(symbol, days)
        
        if not news_data:
            print("⚠️  No news data collected, using dummy sentiment features")
            return self._add_dummy_sentiment_features(stock_data)
        
        # Create sentiment features
        enhanced_data = self.feature_engineer.create_sentiment_features(news_data, stock_data)
        
        print(f"✅ Enhanced stock data with sentiment features")
        print(f"   Original features: {len(stock_data.columns)}")
        print(f"   Enhanced features: {len(enhanced_data.columns)}")
        
        return enhanced_data
    
    def _add_dummy_sentiment_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Add dummy sentiment features when real sentiment is not available"""
        
        # Add neutral sentiment features
        sentiment_features = {
            'sentiment_positive': 0.33,
            'sentiment_negative': 0.33,
            'sentiment_neutral': 0.34,
            'sentiment_compound': 0.0,
            'news_volume': 1.0,
            'sentiment_volatility': 0.1,
            'sentiment_sma_5': 0.0,
            'sentiment_sma_20': 0.0,
            'sentiment_momentum': 0.0,
            'sentiment_rsi': 50.0
        }
        
        for feature, value in sentiment_features.items():
            stock_data[feature] = value
        
        return stock_data

def main():
    """Test sentiment analysis functionality"""
    print("🧪 Testing sentiment analysis functionality...")
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not available, skipping tests")
        return
    
    # Test FinBERT sentiment analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "Apple stock soars on strong earnings report and positive outlook",
        "Market crash concerns grow as inflation fears mount", 
        "Tesla announces new factory opening, stock remains stable",
        "Federal Reserve signals potential interest rate cuts"
    ]
    
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    
    results = analyzer.analyze_sentiment(test_texts)
    
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {result['text'][:60]}...")
        print(f"Positive: {result['positive']:.3f}")
        print(f"Negative: {result['negative']:.3f}")
        print(f"Neutral: {result['neutral']:.3f}")
        print(f"Compound: {result['compound']:.3f}")
    
    # Test with dummy stock data
    print("\n" + "="*50)
    print("Testing with dummy stock data...")
    
    dummy_stock_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
        'close': np.random.normal(150, 10, 30),
        'volume': np.random.normal(1000000, 100000, 30)
    })
    
    predictor = SentimentStockPredictor()
    enhanced_data = predictor.enhance_stock_data_with_sentiment(
        dummy_stock_data, 'AAPL', days=7
    )
    
    print(f"\nEnhanced data shape: {enhanced_data.shape}")
    print(f"New sentiment columns: {[col for col in enhanced_data.columns if 'sentiment' in col or 'news' in col]}")
    
    print("\n✅ Sentiment analysis tests completed!")

if __name__ == "__main__":
    main()