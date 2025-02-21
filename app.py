# app.py - Main Flask Application

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go
import plotly
import json
import os
import logging
import traceback
import requests
import re
from textblob import TextBlob
import praw
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from dotenv import load_dotenv
import os

load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache for model predictions and sentiment analysis
model_cache = {}
sentiment_cache = {}

# Reddit API configuration
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID', REDDIT_CLIENT_ID)
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET', REDDIT_CLIENT_SECRET)
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', 'stock_sentiment_app:v1.0 (by /u/Regular-Smell-5433)')

# Twitter API configuration
TWITTER_BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN', TWITTER_BEARER_TOKEN)

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'statsmodels', 'sklearn', 
        'tensorflow', 'plotly', 'textblob', 'praw', 'requests'
    ]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Install them using: pip install " + " ".join(missing_packages))
        return False
    return True

def fetch_stock_data(symbol, period='1y'):
    """Fetch historical stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            logger.warning(f"No data available for symbol: {symbol}")
            return None, "Invalid stock symbol or no data available"
        
        # Reset index to make Date a column
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        
        return data, None
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, f"Error fetching data: {str(e)}"

def fetch_twitter_sentiment(symbol, days=7):
    """Fetch and analyze Twitter sentiment for a stock symbol"""
    if not TWITTER_BEARER_TOKEN:
        logger.warning("Twitter API token not set. Skipping Twitter sentiment analysis.")
        return None
    
    try:
        # Check cache first
        cache_key = f"twitter_{symbol}_{days}"
        if cache_key in sentiment_cache and (datetime.now() - sentiment_cache[cache_key]['timestamp']).seconds < 3600:
            return sentiment_cache[cache_key]['data']
        
        # Prepare search query (cashtag and company name if available)
        search_query = f"${symbol}"
        
        try:
            company_name = yf.Ticker(symbol).info.get('shortName', '')
            if company_name:
                search_query += f" OR \"{company_name}\""
        except:
            pass
            
        # Set up Twitter API endpoint
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Calculate start time (days ago)
        start_time = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Set query parameters
        params = {
            'query': search_query,
            'max_results': 100,
            'start_time': start_time,
            'tweet.fields': 'created_at,public_metrics',
            'expansions': 'author_id'
        }
        
        # Set headers with bearer token
        headers = {
            'Authorization': f'Bearer {TWITTER_BEARER_TOKEN}'
        }
        
        # Make request to Twitter API
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Twitter API error: {response.status_code} - {response.text}")
            return None
            
        # Process tweets
        result = response.json()
        tweets = result.get('data', [])
        
        if not tweets:
            logger.warning(f"No tweets found for {symbol}")
            return None
            
        # Analyze sentiment
        daily_sentiment = defaultdict(list)
        overall_sentiment = []
        
        for tweet in tweets:
            # Clean tweet text
            text = tweet.get('text', '')
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)     # Remove mentions
            text = re.sub(r'#\w+', '', text)     # Remove hashtags
            
            # Skip empty tweets
            if not text.strip():
                continue
                
            # Get sentiment using TextBlob
            analysis = TextBlob(text)
            sentiment_score = analysis.sentiment.polarity
            
            # Add to overall sentiment
            overall_sentiment.append(sentiment_score)
            
            # Group by day
            created_at = datetime.strptime(
                tweet.get('created_at', ''), 
                '%Y-%m-%dT%H:%M:%S.%fZ'
            ).date()
            daily_sentiment[created_at.strftime('%Y-%m-%d')].append(sentiment_score)
        
        # Calculate average sentiment by day
        daily_avg_sentiment = {}
        for day, scores in daily_sentiment.items():
            daily_avg_sentiment[day] = sum(scores) / len(scores) if scores else 0
            
        # Calculate overall metrics
        avg_sentiment = sum(overall_sentiment) / len(overall_sentiment) if overall_sentiment else 0
        sentiment_trend = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
        
        sentiment_data = {
            'average_sentiment': avg_sentiment,
            'sentiment_trend': sentiment_trend,
            'daily_sentiment': daily_avg_sentiment,
            'tweet_count': len(tweets),
            'source': 'Twitter'
        }
        
        # Cache the results
        sentiment_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': sentiment_data
        }
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error fetching Twitter sentiment: {str(e)}")
        return None

def fetch_reddit_sentiment(symbol, days=7):
    """Fetch and analyze Reddit sentiment for a stock symbol"""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.warning("Reddit API credentials not set. Skipping Reddit sentiment analysis.")
        return None
    
    try:
        # Check cache first
        cache_key = f"reddit_{symbol}_{days}"
        if cache_key in sentiment_cache and (datetime.now() - sentiment_cache[cache_key]['timestamp']).seconds < 3600:
            return sentiment_cache[cache_key]['data']
        
        # Initialize PRAW client
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        # Get company name for more thorough searching
        try:
            company_name = yf.Ticker(symbol).info.get('shortName', symbol)
        except:
            company_name = symbol
        
        # Subreddits to search
        subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
        
        # Combine all posts with ThreadPoolExecutor for parallel processing
        all_posts = []
        all_comments = []
        
        def get_posts_from_subreddit(subreddit_name):
            subreddit = reddit.subreddit(subreddit_name)
            search_terms = [symbol]
            
            # Add cashtag and company name variations
            search_terms.append(f"${symbol}")
            
            if company_name != symbol:
                search_terms.append(company_name)
                
            posts = []
            for term in search_terms:
                posts.extend(list(subreddit.search(
                    term, 
                    time_filter='week',
                    limit=20
                )))
            
            return posts
            
        with ThreadPoolExecutor(max_workers=len(subreddits)) as executor:
            subreddit_posts = list(executor.map(get_posts_from_subreddit, subreddits))
            
        for posts in subreddit_posts:
            all_posts.extend(posts)
            
        # Filter posts by creation time
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_posts = [
            post for post in all_posts 
            if datetime.fromtimestamp(post.created_utc) >= cutoff_date
        ]
        
        # If no posts found, return None
        if not filtered_posts:
            logger.warning(f"No Reddit posts found for {symbol}")
            return None
            
        # Extract comments from top posts (limit to 100 comments per post)
        for post in filtered_posts[:10]:  # Limit to top 10 posts
            post.comments.replace_more(limit=0)
            all_comments.extend(post.comments.list())
            
        # Analyze sentiment
        post_sentiments = []
        comment_sentiments = []
        daily_sentiment = defaultdict(list)
        
        # Analyze post sentiment
        for post in filtered_posts:
            text = post.title + " " + (post.selftext if hasattr(post, 'selftext') else "")
            analysis = TextBlob(text)
            sentiment_score = analysis.sentiment.polarity
            post_sentiments.append(sentiment_score)
            
            # Group by day
            post_date = datetime.fromtimestamp(post.created_utc).date()
            daily_sentiment[post_date.strftime('%Y-%m-%d')].append(sentiment_score)
            
        # Analyze comment sentiment
        for comment in all_comments:
            if hasattr(comment, 'body'):
                analysis = TextBlob(comment.body)
                sentiment_score = analysis.sentiment.polarity
                comment_sentiments.append(sentiment_score)
                
                # Group by day
                comment_date = datetime.fromtimestamp(comment.created_utc).date()
                daily_sentiment[comment_date.strftime('%Y-%m-%d')].append(sentiment_score)
        
        # Calculate average sentiment by day
        daily_avg_sentiment = {}
        for day, scores in daily_sentiment.items():
            daily_avg_sentiment[day] = sum(scores) / len(scores) if scores else 0
            
        # Calculate overall metrics
        all_sentiments = post_sentiments + comment_sentiments
        avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
        sentiment_trend = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
        
        sentiment_data = {
            'average_sentiment': avg_sentiment,
            'sentiment_trend': sentiment_trend,
            'daily_sentiment': daily_avg_sentiment,
            'post_count': len(filtered_posts),
            'comment_count': len(all_comments),
            'source': 'Reddit'
        }
        
        # Cache the results
        sentiment_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': sentiment_data
        }
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error fetching Reddit sentiment: {str(e)}")
        return None

def get_sentiment_data(symbol):
    """Get sentiment data from available sources"""
    # Try Reddit first
    reddit_sentiment = fetch_reddit_sentiment(symbol)
    
    # If Reddit fails, try Twitter
    if not reddit_sentiment:
        twitter_sentiment = fetch_twitter_sentiment(symbol)
        return twitter_sentiment
    
    return reddit_sentiment

def prepare_data_for_lstm(data, target_col='Close', sequence_length=60, include_sentiment=False, sentiment_data=None):
    """Prepare data for LSTM model with optional sentiment features"""
    try:
        # Check if we have enough data
        if len(data) <= sequence_length:
            logger.warning(f"Not enough data points: {len(data)} <= {sequence_length}")
            raise ValueError(f"Need at least {sequence_length+1} data points for training")
        
        # Create additional features if sentiment data is available
        features = []
        
        if include_sentiment and sentiment_data:
            # Add sentiment score as a feature
            # Map dates to sentiment scores
            date_to_sentiment = sentiment_data.get('daily_sentiment', {})
            
            # Default sentiment (neutral) for days without data
            default_sentiment = 0
            
            # Add sentiment to historical data
            data['sentiment'] = data['Date'].dt.strftime('%Y-%m-%d').map(
                lambda x: date_to_sentiment.get(x, default_sentiment)
            )
            
            # Fill NaN values with default sentiment
            data['sentiment'] = data['sentiment'].fillna(default_sentiment)
            
            features = ['sentiment']
            
        # Extract the target column and convert to numpy array
        dataset = data[target_col].values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Prepare additional feature scalers if needed
        feature_scalers = {}
        scaled_features = []
        
        for feature in features:
            feature_data = data[feature].values.reshape(-1, 1)
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_feature = feature_scaler.fit_transform(feature_data)
            scaled_features.append(scaled_feature)
            feature_scalers[feature] = feature_scaler
        
        # Create sequences with target and additional features
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            # Start with price sequence
            seq = [scaled_data[i-sequence_length:i, 0]]
            
            # Add additional feature sequences if available
            for feature_data in scaled_features:
                seq.append(feature_data[i-sequence_length:i, 0])
                
            X.append(np.column_stack(seq) if len(seq) > 1 else seq[0].reshape(-1, 1))
            y.append(scaled_data[i, 0])
        
        # Convert to numpy arrays
        X, y = np.array(X), np.array(y)
        
        # Calculate feature dimension
        feature_dim = 1 + len(features)
        
        # Reshape for LSTM input [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], sequence_length, feature_dim))
        
        return X, y, scaler, feature_scalers
    except Exception as e:
        logger.error(f"Error preparing data for LSTM: {str(e)}")
        raise

def create_lstm_model(input_shape):
    """Create an LSTM model for time series forecasting"""
    try:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        logger.error(f"Error creating LSTM model: {str(e)}")
        raise

def train_lstm_model(data, epochs=20, batch_size=32, sequence_length=60, include_sentiment=False, sentiment_data=None):
    """Train LSTM model and make predictions with optional sentiment data"""
    try:
        # Prepare data with sentiment if available
        X, y, scaler, feature_scalers = prepare_data_for_lstm(
            data, 
            sequence_length=sequence_length,
            include_sentiment=include_sentiment,
            sentiment_data=sentiment_data
        )
        
        # Feature dimension (price + any additional features)
        feature_dim = X.shape[2]
        
        # Create and train model
        model = create_lstm_model((sequence_length, feature_dim))
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Prepare input for prediction (last sequence_length days)
        last_sequence = []
        
        # Add price data
        price_sequence = data['Close'].values[-sequence_length:].reshape(-1, 1)
        last_sequence.append(scaler.transform(price_sequence))
        
        # Add sentiment data if available
        if include_sentiment and sentiment_data and 'sentiment' in feature_scalers:
            sentiment_vals = data['sentiment'].values[-sequence_length:].reshape(-1, 1)
            last_sequence.append(feature_scalers['sentiment'].transform(sentiment_vals))
        
        # Stack features
        last_sequence_stack = np.column_stack(last_sequence)
        
        # Predict next 7 days
        predictions = []
        current_batch = last_sequence_stack.reshape(1, sequence_length, feature_dim)
        
        for _ in range(7):
            # Predict next day
            next_pred = model.predict(current_batch, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update the sequence for next iteration
            new_sequence = np.zeros((1, feature_dim))
            new_sequence[0, 0] = next_pred  # Set predicted price
            
            # Set sentiment value (use the last known sentiment or neutral)
            if feature_dim > 1:
                # For simplicity, use the last sentiment value
                new_sequence[0, 1:] = current_batch[0, -1, 1:]
            
            # Remove first element and add the new prediction
            current_batch = np.concatenate([
                current_batch[:, 1:, :],
                new_sequence.reshape(1, 1, feature_dim)
            ], axis=1)
        
        # Inverse transform to get actual price values
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = scaler.inverse_transform(predictions_array)
        
        return predictions_rescaled.flatten()
    except Exception as e:
        logger.error(f"Error in LSTM model training: {str(e)}")
        # Return fallback predictions (based on last known value)
        if len(data) > 0:
            last_value = data['Close'].iloc[-1]
            return np.array([last_value] * 7)
        else:
            return np.zeros(7)

def train_arima_model(data, order=(5, 1, 0), include_sentiment=False, sentiment_data=None):
    """Train ARIMA model and make predictions with optional sentiment influence"""
    try:
        # Basic ARIMA model without sentiment
        if not include_sentiment or not sentiment_data:
            model = ARIMA(data['Close'].values, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)
            return forecast
        
        # For ARIMA with sentiment, we'll use a simple adjustment approach
        # since ARIMA doesn't directly support external regressors without moving to ARIMAX
        
        # Get sentiment trend
        sentiment_trend = sentiment_data.get('sentiment_trend', 'neutral')
        avg_sentiment = sentiment_data.get('average_sentiment', 0)
        
        # Fit base ARIMA model
        model = ARIMA(data['Close'].values, order=order)
        model_fit = model.fit()
        
        # Get base forecast
        base_forecast = model_fit.forecast(steps=7)
        
        # Apply sentiment adjustment
        # More positive sentiment = slightly higher forecast
        # More negative sentiment = slightly lower forecast
        sentiment_factor = 1 + (avg_sentiment * 0.05)  # 5% influence per sentiment unit
        
        # Apply the adjustment
        adjusted_forecast = base_forecast * sentiment_factor
        
        return adjusted_forecast
    except Exception as e:
        logger.error(f"Error in ARIMA model training: {str(e)}")
        # Return fallback predictions
        if len(data) > 0:
            last_value = data['Close'].iloc[-1]
            return np.array([last_value] * 7)
        else:
            return np.zeros(7)

def train_linear_regression_model(data, future_days=7, include_sentiment=False, sentiment_data=None):
    """Train Linear Regression model with optional sentiment features"""
    try:
        # Create features based on day numbers
        data_copy = data.copy()
        data_copy['Day'] = range(len(data_copy))
        
        features = ['Day']
        
        # Add sentiment feature if available
        if include_sentiment and sentiment_data:
            # Map dates to sentiment scores
            date_to_sentiment = sentiment_data.get('daily_sentiment', {})
            
            # Default sentiment (neutral) for days without data
            default_sentiment = 0
            
            # Add sentiment to historical data
            data_copy['sentiment'] = data_copy['Date'].dt.strftime('%Y-%m-%d').map(
                lambda x: date_to_sentiment.get(x, default_sentiment)
            )
            
            # Fill NaN values with default sentiment
            data_copy['sentiment'] = data_copy['sentiment'].fillna(default_sentiment)
            
            features.append('sentiment')
        
        # Train model with selected features
        model = LinearRegression()
        model.fit(data_copy[features], data_copy['Close'])
        
        # Predict future days
        future_days_df = pd.DataFrame({
            'Day': range(len(data_copy), len(data_copy) + future_days)
        })
        
        # Add sentiment for future days
        if include_sentiment and sentiment_data and 'sentiment' in features:
            # For simplicity, use average sentiment for all future days
            future_days_df['sentiment'] = sentiment_data.get('average_sentiment', 0)
            
        predictions = model.predict(future_days_df[features])
        
        return predictions
    except Exception as e:
        logger.error(f"Error in Linear Regression model training: {str(e)}")
        # Return fallback predictions
        if len(data) > 0:
            last_value = data['Close'].iloc[-1]
            return np.array([last_value] * 7)
        else:
            return np.zeros(7)

def get_or_train_models(symbol, data, sentiment_data=None):
    """Get cached models or train new ones with sentiment analysis"""
    try:
        current_time = datetime.now()
        cache_key = f"{symbol}"
        
        if sentiment_data:
            cache_key += "_with_sentiment"
        
        # Check if we have a recent cache for this symbol
        if cache_key in model_cache and (current_time - model_cache[cache_key]['timestamp']).seconds < 3600:
            logger.info(f"Using cached predictions for {cache_key}")
            return model_cache[cache_key]['predictions']
        
        logger.info(f"Training new models for {cache_key}")
        
        # Flag for using sentiment
        use_sentiment = sentiment_data is not None
        
        # Train models with sentiment if available
        lstm_predictions = train_lstm_model(data, include_sentiment=use_sentiment, sentiment_data=sentiment_data)
        arima_predictions = train_arima_model(data, include_sentiment=use_sentiment, sentiment_data=sentiment_data)
        lr_predictions = train_linear_regression_model(data, include_sentiment=use_sentiment, sentiment_data=sentiment_data)
        
        # Cache results
        model_cache[cache_key] = {
            'timestamp': current_time,
            'predictions': {
                'LSTM': lstm_predictions,
                'ARIMA': arima_predictions,
                'Linear Regression': lr_predictions
            }
        }
        
        return model_cache[cache_key]['predictions']
    except Exception as e:
        logger.error(f"Error in model training/caching: {str(e)}")
        raise

def create_forecast_plot(historical_data, predictions_dict, symbol, sentiment_data=None):
    """Create a plotly figure with historical data, predictions, and sentiment indicators"""
    try:
        # Generate future dates for prediction
        last_date = historical_data['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(7)]
        future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Create figure with dark theme
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['Date'], 
            y=historical_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#00FF7F', width=2)  # Bright green for historical data
        ))
        
        # Add predictions from each model with distinct colors and patterns
        prediction_colors = {
            'LSTM': '#FF1493',          # Deep pink
            'ARIMA': '#00BFFF',         # Deep sky blue
            'Linear Regression': '#FFD700'  # Gold
        }
        
        for model_name, predictions in predictions_dict.items():
            if len(predictions) == 7:  # Ensure we have the expected predictions length
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=predictions,
                    mode='lines+markers',
                    name=f'{model_name}',
                    line=dict(
                        color=prediction_colors[model_name],
                        dash='dash',
                        width=2
                    ),
                    marker=dict(
                        size=8,
                        symbol='diamond'
                    )
                ))
        
        # Add sentiment indicators if available
        if sentiment_data and 'daily_sentiment' in sentiment_data:
            sentiment_dates = []
            sentiment_values = []
            sentiment_colors = []
            
            for date_str, sentiment in sentiment_data['daily_sentiment'].items():
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    if date >= last_date - timedelta(days=30):
                        sentiment_dates.append(date)
                        sentiment_values.append(sentiment)
                        
                        if sentiment > 0.05:
                            sentiment_colors.append('#00FF00')  # Bright green
                        elif sentiment < -0.05:
                            sentiment_colors.append('#FF0000')  # Bright red
                        else:
                            sentiment_colors.append('#808080')  # Gray
                except:
                    continue
            
            if sentiment_dates:
                # Scale sentiment markers relative to price range
                price_range = historical_data['Close'].max() - historical_data['Close'].min()
                scaled_sentiment = [s * (price_range * 0.1) + historical_data['Close'].mean() for s in sentiment_values]
                
                fig.add_trace(go.Scatter(
                    x=sentiment_dates,
                    y=scaled_sentiment,
                    mode='markers',
                    name='Market Sentiment',
                    marker=dict(
                        size=12,
                        color=sentiment_colors,
                        symbol='star',
                        line=dict(width=2, color='#FFFFFF')
                    )
                ))
        
        # Update layout with dark theme
        title = f'{symbol} Stock Price Forecast'
        if sentiment_data:
            sentiment_trend = sentiment_data.get('sentiment_trend', 'neutral')
            source = sentiment_data.get('source', 'unknown')
            title += f' with {sentiment_trend.title()} {source} Sentiment'
            
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, color='#FFFFFF')
            ),
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            xaxis=dict(
                title='Date',
                titlefont=dict(color='#FFFFFF'),
                tickfont=dict(color='#FFFFFF'),
                gridcolor='#333333',
                showgrid=True
            ),
            yaxis=dict(
                title='Price (USD)',
                titlefont=dict(color='#FFFFFF'),
                tickfont=dict(color='#FFFFFF'),
                gridcolor='#333333',
                showgrid=True
            ),
            legend=dict(
                font=dict(color='#FFFFFF'),
                bgcolor='#333333',
                bordercolor='#666666'
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='#333333',
                font=dict(color='#FFFFFF')
            )
        )
        
        # Add range slider with dark theme
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor='#333333',
                    activecolor='#666666',
                    font=dict(color='#FFFFFF')
                ),
                rangeslider=dict(
                    visible=True,
                    bgcolor='#333333',
                    bordercolor='#666666'
                ),
                type="date"
            )
        )
        
        return fig, future_dates_str
        
    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Process prediction request and return results with sentiment analysis"""
    try:
        # Get stock symbol from form
        symbol = request.form.get('symbol', '').strip().upper()
        include_sentiment = request.form.get('include_sentiment', 'true').lower() == 'true'
        
        # Validate input
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'})
        
        # Fetch historical data
        data, error = fetch_stock_data(symbol)
        if error:
            return jsonify({'error': error})
        
        # Get sentiment data if requested
        sentiment_data = None
        if include_sentiment:
            sentiment_data = get_sentiment_data(symbol)
        
        # Train models and get predictions
        predictions = get_or_train_models(symbol, data, sentiment_data)
        
        # Create visualization
        fig, _ = create_forecast_plot(data, predictions, symbol, sentiment_data)
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Generate future dates
        last_date = data['Date'].iloc[-1]
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
        
        # Ensure predictions are valid numbers and properly formatted
        def format_prediction(value):
            try:
                if np.isnan(value) or value is None:
                    return None
                return round(float(value), 2)
            except:
                return None
        
        # Format predictions for each model
        lstm_predictions = [format_prediction(p) for p in predictions['LSTM']]
        arima_predictions = [format_prediction(p) for p in predictions['ARIMA']]
        lr_predictions = [format_prediction(p) for p in predictions['Linear Regression']]
        
        # Calculate average predictions
        avg_predictions = []
        for i in range(7):
            valid_predictions = []
            if lstm_predictions[i] is not None:
                valid_predictions.append(lstm_predictions[i])
            if arima_predictions[i] is not None:
                valid_predictions.append(arima_predictions[i])
            if lr_predictions[i] is not None:
                valid_predictions.append(lr_predictions[i])
            
            if valid_predictions:
                avg_predictions.append(round(sum(valid_predictions) / len(valid_predictions), 2))
            else:
                avg_predictions.append(round(float(data['Close'].iloc[-1]), 2))
        
        # Format sentiment data
        formatted_sentiment = {}
        if sentiment_data:
            formatted_sentiment = {
                'average_sentiment': round(sentiment_data.get('average_sentiment', 0), 2),
                'sentiment_trend': sentiment_data.get('sentiment_trend', 'neutral'),
                'source': sentiment_data.get('source', 'Data'),
                'sample_size': sentiment_data.get('tweet_count', sentiment_data.get('post_count', 0))
            }
        
        # Prepare response
        response = {
            'symbol': symbol,
            'last_price': round(float(data['Close'].iloc[-1]), 2),
            'predictions': {
                'dates': future_dates,
                'average': avg_predictions,
                'lstm': lstm_predictions,
                'arima': arima_predictions,
                'linear_regression': lr_predictions
            },
            'sentiment': formatted_sentiment,
            'plot': plot_json
        }
        
        # Log the predictions for debugging
        logger.info(f"Predictions for {symbol}:")
        logger.info(f"Dates: {future_dates}")
        logger.info(f"LSTM: {lstm_predictions}")
        logger.info(f"ARIMA: {arima_predictions}")
        logger.info(f"Linear Regression: {lr_predictions}")
        logger.info(f"Average: {avg_predictions}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f"An error occurred: {str(e)}"})
        

@app.route('/api/symbols')
def symbols_autocomplete():
    """Fetch matching stock symbols for autocomplete"""
    try:
        query = request.args.get('q', '').strip().upper()
        
        if not query or len(query) < 1:
            return jsonify([])
        
        # Load popular symbols (could be cached or loaded from a file)
        popular_symbols = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'FB': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp.',
            'DIS': 'The Walt Disney Company',
            'NFLX': 'Netflix Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'ADBE': 'Adobe Inc.',
            'INTC': 'Intel Corporation',
            'CSCO': 'Cisco Systems Inc.',
            'PEP': 'PepsiCo Inc.',
            'CMCSA': 'Comcast Corporation',
            'XOM': 'Exxon Mobil Corporation',
            'T': 'AT&T Inc.'
        }
        
        results = []
        
        # Look for matching symbols in our predefined list
        for symbol, name in popular_symbols.items():
            if query in symbol:
                results.append({
                    'symbol': symbol,
                    'name': name
                })
        
        # If no matches found, try to look up online (optional)
        if not results and len(query) >= 2:
            try:
                search = yf.Tickers(query).tickers
                if search:
                    for symbol in search:
                        if query in symbol:
                            try:
                                info = search[symbol].info
                                results.append({
                                    'symbol': symbol,
                                    'name': info.get('shortName', info.get('longName', symbol))
                                })
                            except:
                                continue
            except:
                # If online lookup fails, just return what we have
                pass
                
        # Sort results by symbol
        results.sort(key=lambda x: x['symbol'])
        
        # Limit to top 10 results
        return jsonify(results[:10])
        
    except Exception as e:
        logger.error(f"Error in symbol lookup: {str(e)}")
        return jsonify([])

@app.route('/api/stock_info/<symbol>')
def get_stock_info(symbol):
    """Get current stock information"""
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        
        if not info:
            return jsonify({'error': 'Stock information not available'})
        
        # Get last day of trading data
        hist = stock.history(period='1d')
        
        if hist.empty:
            return jsonify({'error': 'Price data not available'})
        
        # Extract relevant information
        result = {
            'symbol': symbol.upper(),
            'name': info.get('shortName', info.get('longName', symbol)),
            'current_price': float(hist['Close'].iloc[-1]),
            'change': float(hist['Close'].iloc[-1] - hist['Open'].iloc[0]),
            'change_percent': float((hist['Close'].iloc[-1] / hist['Open'].iloc[0] - 1) * 100),
            'high': float(hist['High'].max()),
            'low': float(hist['Low'].min()),
            'volume': int(hist['Volume'].sum()),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('trailingPE', None),
            'sector': info.get('sector', None),
            'industry': info.get('industry', None),
            'description': info.get('longBusinessSummary', None)
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching stock info: {str(e)}")
        return jsonify({'error': f"Could not retrieve stock information: {str(e)}"})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    dependencies_ok = check_dependencies()
    return jsonify({
        'status': 'ok' if dependencies_ok else 'warning',
        'message': 'All systems operational' if dependencies_ok else 'Some dependencies might be missing',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/templates/index.html')
def get_index_template():
    """Return the index template for SPA"""
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors by returning to the main page"""
    return render_template('index.html'), 200  # SPA fallback

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors with a JSON response"""
    logger.error(f"500 error: {str(e)}")
    return jsonify({
        'error': 'Internal server error occurred',
        'status': 500
    }), 500

# Add cache headers to static assets
@app.after_request
def add_cache_headers(response):
    """Add cache headers to responses"""
    if request.path.startswith('/static/'):
        # Cache static files for 1 day
        response.cache_control.max_age = 86400
    return response

if __name__ == '__main__':
    # Check dependencies before startup
    check_dependencies()
    
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Start the application with debug mode in development
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)