from transformers import pipeline 
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import numpy as np
import time
import pandas as pd
import json
import os

# ------------------------------
# Initialize Models and Clients
# ------------------------------

# FinBERT for financial sentiment
finbert_model = pipeline("text-classification", model="ProsusAI/finbert", device=0)
# Zero-shot classification for broader labels
zero_shot_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# NewsAPI client
newsapi = NewsApiClient(api_key='commented out')

# ------------------------------
# Persistent Cache Setup
# ------------------------------

CACHE_FILE = "news_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        news_cache = json.load(f)
else:
    news_cache = {}

def save_cache():
    """Save the news cache to the persistent JSON file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(news_cache, f)

def update_news_cache_for_missing_dates(keyword):
    """
    Fill any gaps between the last cached date and today, to ensure each day
    has its headlines stored.
    """
    today = datetime.today().date()
    if news_cache:
        cached_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in news_cache.keys()]
        last_cached_date = max(cached_dates)
        date_to_update = last_cached_date + timedelta(days=1)
        while date_to_update <= today:
            date_str = date_to_update.strftime('%Y-%m-%d')
            if date_str not in news_cache:
                print(f"Updating cache for missing date: {date_str}")
                headlines = fetch_news_sentiment(keyword, date_str)
                news_cache[date_str] = headlines
                save_cache()
            date_to_update += timedelta(days=1)

def fetch_news_sentiment(keyword, date_str):
    """
    Fetch relevant news articles for a specific date using NewsAPI.
    If the date is older than 30 days, returns an empty list.
    """
    today = datetime.today().date()
    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    if (today - target_date).days > 30:
        return []

    if date_str in news_cache:
        return news_cache[date_str]

    try:
        all_articles = newsapi.get_everything(
            q=keyword,
            from_param=date_str,
            to=date_str,
            language='en',
            sort_by='relevancy',
            page_size=100
        )
        headlines = [article['title'] for article in all_articles['articles'] if article.get('title')]
        news_cache[date_str] = headlines
        save_cache()
        return headlines
    except Exception as e:
        print(f"Error fetching news for {date_str}: {e}")
        time.sleep(5)
        return []

def calculate_sentiment_scores(headlines):
    """
    Compute an average sentiment score using FinBERT + zero-shot classification.
    """
    if not headlines:
        return 0.0
    
    scores = []
    for headline in headlines:
        if headline is None:
            continue
        try:
            # FinBERT
            finbert_result = finbert_model(headline)[0]
            finbert_label = finbert_result['label'].lower()
            finbert_score = finbert_result['score']
            if "positive" in finbert_label:
                finbert_sent = 1 * finbert_score
            elif "negative" in finbert_label:
                finbert_sent = -1 * finbert_score
            else:
                finbert_sent = 0

            # Zero-shot analysis
            candidate_labels = ["bullish", "bearish", "neutral"]
            zero_shot_result = zero_shot_model(headline, candidate_labels=candidate_labels)
            top_label = zero_shot_result["labels"][0].lower()
            top_score = zero_shot_result["scores"][0]
            if top_label == "bullish":
                zero_shot_sent = 1 * top_score
            elif top_label == "bearish":
                zero_shot_sent = -1 * top_score
            else:
                zero_shot_sent = 0

            avg_sent = (finbert_sent + zero_shot_sent) / 2
            scores.append(avg_sent)
        except Exception as e:
            print(f"Error processing sentiment for headline '{headline}': {e}")
    
    return np.mean(scores) if scores else 0.0

def compute_news_sentiment(data, keyword):
    """
    For each date in the DataFrame index:
      - updates the cache for missing dates,
      - fetches headlines,
      - calculates an average sentiment score (News_Sentiment).
    """
    update_news_cache_for_missing_dates(keyword)
    
    sentiments = []
    for idx_date in data.index:
        date_str = idx_date.strftime('%Y-%m-%d')
        headlines = fetch_news_sentiment(keyword, date_str)
        sentiment = calculate_sentiment_scores(headlines)
        sentiments.append(sentiment)
    
    data['News_Sentiment'] = sentiments
    return data
