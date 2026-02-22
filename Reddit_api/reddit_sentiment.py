"""
Sentiment Analysis Module for Reddit Data

Performs sentiment analysis on Reddit posts using cardiffnlp/twitter-roberta-base-sentiment.
- Reads input from MongoDB (reddit_db.posts) or output/Reddit_data.json
- Analyzes sentiment of post titles and content
- Saves results to MongoDB (reddit_db.sentiment_results) and output/reddit_sentiment_results.json
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import torch
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGODB_DB", "reddit_db")
source_collection = os.getenv("MONGODB_COLLECTION", "posts")
sentiment_collection = os.getenv("SENTIMENT_COLLECTION", "sentiment_results")

client = MongoClient(mongo_uri)
db = client[db_name]
col_in = db[source_collection]
col_out = db[sentiment_collection]

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INPUT_FILE_FALLBACK = os.path.join(OUTPUT_DIR, "Reddit_data.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_sentiment_results.json")

# Load model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
logging.info(f"🤖 Loading sentiment analysis model: {MODEL_NAME}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    logging.info("✅ Sentiment analysis model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load sentiment model: {e}")
    logging.error("   Please install transformers: pip install transformers torch scipy")
    raise

# Sentiment labels
LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def preprocess_text(text: str) -> str:
    """
    Clean the text before feeding it to the model.
    Removes newlines and demojizes emojis.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    text = emoji.demojize(text)
    
    # Limit text length for performance
    if len(text) > 5000:
        text = text[:5000]
    
    return text


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of the given text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary with 'label', 'score', and 'scores' (all three probabilities)
    """
    try:
        if not text or len(text.strip()) < 3:
            return {
                "label": "Neutral",
                "score": 0.0,
                "scores": {"Negative": 0.0, "Neutral": 1.0, "Positive": 0.0}
            }
        
        text = preprocess_text(text)
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            output = model(**encoded_input)
        
        scores = softmax(output.logits[0].numpy())
        max_idx = scores.argmax()
        
        return {
            "label": LABELS[max_idx],
            "score": float(scores[max_idx]),
            "scores": {
                "Negative": float(scores[0]),
                "Neutral": float(scores[1]),
                "Positive": float(scores[2])
            }
        }
    except Exception as e:
        logging.warning(f"⚠️  Error in sentiment analysis: {e}")
        return {
            "label": "Error",
            "score": 0.0,
            "scores": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}
        }


def process_post(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single Reddit post for sentiment analysis.
    
    Args:
        doc (dict): Reddit post data
        
    Returns:
        dict: Processed post with sentiment analysis results or None if no text
    """
    try:
        # Get text content (prefer cleaned_text, then title + body/content)
        text = doc.get("cleaned_text", "")
        
        if not text:
            title = doc.get("title", "")
            body = doc.get("body", "")
            content = doc.get("content", "")
            
            # Combine available text
            text_parts = [title]
            if body:
                text_parts.append(body)
            elif content:
                text_parts.append(content)
            
            text = " ".join(text_parts).strip()
        
        if not text or len(text) < 3:
            return None
        
        # Analyze sentiment
        sentiment = analyze_sentiment(text)
        
        return {
            "_id": doc.get("_id"),
            "post_id": doc.get("post_id") or doc.get("id"),
            "subreddit": doc.get("subreddit"),
            "author": doc.get("author"),
            "created_utc": doc.get("created_utc"),
            "permalink": doc.get("permalink"),
            "title": doc.get("title"),
            "text_analyzed": text[:500],  # Store snippet of analyzed text
            "sentiment": sentiment,
            "processed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
    except Exception as e:
        logging.warning(f"⚠️  Error processing post {doc.get('id', 'unknown')}: {e}")
        return None


def load_from_file() -> List[Dict[str, Any]]:
    """Load Reddit posts from JSON file."""
    if not os.path.exists(INPUT_FILE_FALLBACK):
        logging.warning(f"Input file not found: {INPUT_FILE_FALLBACK}")
        return []
    
    try:
        with open(INPUT_FILE_FALLBACK, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.info(f"📂 Loaded {len(data):,} records from {INPUT_FILE_FALLBACK}")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"❌ Error decoding JSON from {INPUT_FILE_FALLBACK}: {e}")
        return []


def load_from_mongo(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load Reddit posts from MongoDB."""
    try:
        query = {}
        cursor = col_in.find(query).limit(limit) if limit else col_in.find(query)
        posts = list(cursor)
        logging.info(f"📂 Loaded {len(posts):,} records from MongoDB ({source_collection})")
        return posts
    except Exception as e:
        logging.error(f"❌ Error loading from MongoDB: {e}")
        return []


def process_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of Reddit posts for sentiment analysis.
    
    Args:
        posts (list): List of Reddit post dictionaries
        
    Returns:
        list: List of processed posts with sentiment analysis
    """
    results = []
    
    for post in tqdm(posts, desc="😊 Analyzing sentiment"):
        result = process_post(post)
        if result:
            results.append(result)
    
    return results


def save_to_mongo(results: List[Dict[str, Any]]) -> None:
    """Save sentiment results to MongoDB."""
    if not results:
        return
    
    try:
        ops: List[UpdateOne] = []
        for r in results:
            key = {"_id": r.get("_id")} if r.get("_id") is not None else {"post_id": r.get("post_id")}
            ops.append(UpdateOne(key, {"$set": r}, upsert=True))
        
        if ops:
            col_out.bulk_write(ops, ordered=False)
            logging.info(f"💾 Saved {len(results):,} sentiment results to MongoDB ({sentiment_collection})")
    except Exception as e:
        logging.error(f"❌ Error saving to MongoDB: {e}")


def save_to_file(results: List[Dict[str, Any]], path: str) -> None:
    """Save sentiment results to JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Saved {len(results):,} sentiment results to {path}")
    except Exception as e:
        logging.error(f"❌ Error saving to file: {e}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics of sentiment analysis results."""
    if not results:
        logging.info("❌ No sentiment analysis results.")
        return
    
    # Count sentiment labels
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0, "Error": 0}
    total_scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    
    for r in results:
        sentiment = r.get("sentiment", {})
        label = sentiment.get("label", "Error")
        score = sentiment.get("score", 0.0)
        
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        if label in total_scores:
            total_scores[label] += score
    
    total = len(results)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"📊 Sentiment Analysis Summary:")
    logging.info(f"   Total posts analyzed: {total:,}")
    logging.info(f"\n   Sentiment Distribution:")
    logging.info(f"      😊 Positive: {sentiment_counts['Positive']:,} ({sentiment_counts['Positive']/total*100:.1f}%)")
    logging.info(f"      😐 Neutral:  {sentiment_counts['Neutral']:,} ({sentiment_counts['Neutral']/total*100:.1f}%)")
    logging.info(f"      😞 Negative: {sentiment_counts['Negative']:,} ({sentiment_counts['Negative']/total*100:.1f}%)")
    
    if sentiment_counts['Error'] > 0:
        logging.info(f"      ⚠️  Errors:   {sentiment_counts['Error']:,} ({sentiment_counts['Error']/total*100:.1f}%)")
    
    logging.info(f"\n   Average Confidence Scores:")
    for label in ["Positive", "Neutral", "Negative"]:
        if sentiment_counts[label] > 0:
            avg_score = total_scores[label] / sentiment_counts[label]
            logging.info(f"      {label}: {avg_score:.3f}")
    
    logging.info(f"{'='*60}\n")


def main() -> None:
    """Main function to run sentiment analysis pipeline."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis for Reddit posts")
    parser.add_argument(
        "--source",
        type=str,
        choices=["mongo", "file"],
        default="file",
        help="Data source: 'mongo' for MongoDB or 'file' for JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of posts to process (for testing)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=OUTPUT_FILE,
        help=f"Path to save JSON results. Defaults to {OUTPUT_FILE}"
    )
    args = parser.parse_args()
    
    # Load data
    logging.info("🚀 Starting Sentiment Analysis pipeline...")
    if args.source == "mongo":
        posts = load_from_mongo(limit=args.limit)
    else:
        posts = load_from_file()
        if args.limit:
            posts = posts[:args.limit]
    
    if not posts:
        logging.info("❌ No input posts found.")
        return
    
    # Process posts
    results = process_posts(posts)
    
    if not results:
        logging.info("❌ No sentiment analysis results generated.")
        return
    
    # Save results
    save_to_mongo(results)
    save_to_file(results, args.output_file)
    
    # Print summary
    print_summary(results)
    
    logging.info("✅ Sentiment Analysis complete!")


if __name__ == "__main__":
    main()
