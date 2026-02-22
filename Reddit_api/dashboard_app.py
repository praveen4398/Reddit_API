"""
Interactive Dashboard for Reddit NLP Analysis

A Flask-based web dashboard providing:
1. Overview with graphs and statistics
2. Smart search across all NLP results
3. Recent search history
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import Counter

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'reddit-nlp-dashboard-secret-key')

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGODB_DB", "reddit_db")

client = MongoClient(mongo_uri)
db = client[db_name]

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Search history storage (in-memory for now, can be moved to MongoDB)
search_history = []

# Output files
OUTPUT_FILES = {
    "overview": os.path.join(OUTPUT_DIR, "reddit_overview.json"),
    "posts": os.path.join(OUTPUT_DIR, "post_summaries.json"),
}


def load_json_file(file_path: str) -> Any:
    """Load data from JSON file."""
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None


def get_overview_data() -> Dict[str, Any]:
    """Get overview statistics and data for dashboard."""
    overview = load_json_file(OUTPUT_FILES["overview"])
    if not overview:
        return {
            "error": "Overview data not found. Please run generate_overview.py first."
        }
    
    return overview


def search_posts(query: str, search_type: str = "all") -> List[Dict[str, Any]]:
    """
    Search posts based on query and type.
    
    Args:
        query: Search query string
        search_type: Type of search (all, keyword, hashtag, spam, sentiment, emotion, topic)
    """
    posts_data = load_json_file(OUTPUT_FILES["posts"])
    if not posts_data or "posts" not in posts_data:
        return []
    
    posts = posts_data["posts"]
    results = []
    query_lower = query.lower()
    
    for post in posts:
        match = False
        nlp = post.get("nlp_results", {})
        
        if search_type == "all":
            # Search in title
            if query_lower in post.get("title", "").lower():
                match = True
            
            # Search in keywords
            keywords = nlp.get("keywords_hashtags", {}).get("keywords", [])
            if any(query_lower in kw.lower() for kw in keywords):
                match = True
            
            # Search in hashtags
            hashtags = nlp.get("keywords_hashtags", {}).get("hashtags", [])
            if any(query_lower in ht.lower() for ht in hashtags):
                match = True
            
            # Search in entities
            entities = nlp.get("named_entities", {}).get("entities", [])
            if any(query_lower in ent.get("entity", "").lower() for ent in entities):
                match = True
            
            # Search in summary
            summary = nlp.get("generated_summary", "")
            if query_lower in summary.lower():
                match = True
        
        elif search_type == "keyword":
            keywords = nlp.get("keywords_hashtags", {}).get("keywords", [])
            if any(query_lower in kw.lower() for kw in keywords):
                match = True
        
        elif search_type == "hashtag":
            hashtags = nlp.get("keywords_hashtags", {}).get("hashtags", [])
            if any(query_lower in ht.lower() for ht in hashtags):
                match = True
        
        elif search_type == "spam":
            flags = nlp.get("behavior_flags", {})
            if flags.get("spam", False):
                match = True
        
        elif search_type == "offensive":
            flags = nlp.get("behavior_flags", {})
            if flags.get("offensive", False):
                match = True
        
        elif search_type == "sentiment":
            sentiment = nlp.get("sentiment_analysis", {})
            if query_lower in sentiment.get("label", "").lower():
                match = True
        
        elif search_type == "emotion":
            emotion = nlp.get("emotion_detection", {})
            if query_lower in emotion.get("label", "").lower():
                match = True
        
        elif search_type == "topic":
            topic = nlp.get("topic_classification", {})
            if query_lower in topic.get("label", "").lower():
                match = True
        
        if match:
            results.append(post)
    
    return results


def format_post_for_display(post: Dict[str, Any]) -> Dict[str, Any]:
    """Format post data for display in UI."""
    nlp = post.get("nlp_results", {})
    
    # Get sentiment
    sentiment = nlp.get("sentiment_analysis", {})
    sentiment_label = sentiment.get("label", "N/A")
    sentiment_confidence = sentiment.get("confidence", 0)
    
    # Get emotion
    emotion = nlp.get("emotion_detection", {})
    emotion_label = emotion.get("label", "N/A")
    emotion_confidence = emotion.get("confidence", 0)
    
    # Get topic
    topic = nlp.get("topic_classification", {})
    topic_label = topic.get("label", "N/A")
    
    # Get keywords
    kw_data = nlp.get("keywords_hashtags", {})
    keywords = kw_data.get("keywords", [])[:10]
    hashtags = kw_data.get("hashtags", [])[:5]
    
    # Get entities
    entities_data = nlp.get("named_entities", {})
    entities = [
        {
            "name": ent.get("entity"),
            "type": ent.get("type"),
            "wiki_url": ent.get("wikipedia_url")
        }
        for ent in entities_data.get("entities", [])[:5]
    ]
    
    # Get behavior flags
    flags = nlp.get("behavior_flags", {})
    
    # Get summary
    summary = nlp.get("generated_summary", "")
    
    return {
        "post_id": post.get("post_id"),
        "title": post.get("title"),
        "subreddit": post.get("subreddit"),
        "author": post.get("author"),
        "score": post.get("score", 0),
        "num_comments": post.get("num_comments", 0),
        "permalink": post.get("permalink"),
        "sentiment": {
            "label": sentiment_label,
            "confidence": round(sentiment_confidence * 100, 1)
        },
        "emotion": {
            "label": emotion_label,
            "confidence": round(emotion_confidence * 100, 1)
        },
        "topic": topic_label,
        "keywords": keywords,
        "hashtags": hashtags,
        "entities": entities,
        "flags": flags,
        "summary": summary
    }


def add_to_search_history(query: str, search_type: str, result_count: int):
    """Add search to history."""
    search_entry = {
        "query": query,
        "type": search_type,
        "result_count": result_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    search_history.insert(0, search_entry)
    
    # Keep only last 50 searches
    if len(search_history) > 50:
        search_history.pop()


# Routes
@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/overview')
def api_overview():
    """API endpoint for overview data."""
    overview = get_overview_data()
    return jsonify(overview)


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for searching posts."""
    data = request.get_json()
    query = data.get('query', '').strip()
    search_type = data.get('type', 'all')
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # Perform search
    results = search_posts(query, search_type)
    
    # Format results
    formatted_results = [format_post_for_display(post) for post in results[:100]]
    
    # Add to history
    add_to_search_history(query, search_type, len(results))
    
    return jsonify({
        "query": query,
        "type": search_type,
        "total_results": len(results),
        "results": formatted_results
    })


@app.route('/api/history')
def api_history():
    """API endpoint for search history."""
    return jsonify({"history": search_history[:20]})


@app.route('/api/stats')
def api_stats():
    """API endpoint for quick statistics."""
    posts_data = load_json_file(OUTPUT_FILES["posts"])
    if not posts_data or "posts" not in posts_data:
        return jsonify({"error": "No data available"})
    
    posts = posts_data["posts"]
    
    # Calculate statistics
    total_posts = len(posts)
    
    sentiment_counts = Counter()
    emotion_counts = Counter()
    topic_counts = Counter()
    spam_count = 0
    offensive_count = 0
    
    for post in posts:
        nlp = post.get("nlp_results", {})
        
        sentiment = nlp.get("sentiment_analysis", {})
        sentiment_counts[sentiment.get("label", "Unknown")] += 1
        
        emotion = nlp.get("emotion_detection", {})
        emotion_counts[emotion.get("label", "Unknown")] += 1
        
        topic = nlp.get("topic_classification", {})
        topic_counts[topic.get("label", "Unknown")] += 1
        
        flags = nlp.get("behavior_flags", {})
        if flags.get("spam"):
            spam_count += 1
        if flags.get("offensive"):
            offensive_count += 1
    
    return jsonify({
        "total_posts": total_posts,
        "sentiment_distribution": dict(sentiment_counts.most_common(5)),
        "emotion_distribution": dict(emotion_counts.most_common(5)),
        "topic_distribution": dict(topic_counts.most_common(5)),
        "spam_count": spam_count,
        "offensive_count": offensive_count
    })


def main():
    """Run the Flask application."""
    logging.info("🚀 Starting Reddit NLP Dashboard...")
    logging.info("📊 Dashboard will be available at: http://localhost:5000")
    
    # Check if data files exist
    if not os.path.exists(OUTPUT_FILES["posts"]):
        logging.warning("⚠️  post_summaries.json not found. Please run generate_post_summaries.py first.")
    
    if not os.path.exists(OUTPUT_FILES["overview"]):
        logging.warning("⚠️  reddit_overview.json not found. Please run generate_overview.py first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
