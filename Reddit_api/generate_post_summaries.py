"""
Post Title-wise NLP Pipeline Summary Generator

Aggregates all NLP processing outputs for each individual post/title.
Provides comprehensive analysis for each post including:
- Post metadata (title, subreddit, author, score, etc.)
- Sentiment analysis results
- Named entities and Wikipedia links
- Emotion detection results
- Keywords & hashtags
- Topic classification
- Behavior flags
- Generated summary

Saves results to both JSON file and MongoDB.
"""

import os
import json
import logging
from typing import Dict, List, Any

from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm
from datetime import datetime, timezone

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGODB_DB", "reddit_db")

client = MongoClient(mongo_uri)
db = client[db_name]

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "post_summaries.json")

# Output files mapping
OUTPUT_FILES = {
    "sentiment": os.path.join(OUTPUT_DIR, "reddit_sentiment_results.json"),
    "nel": os.path.join(OUTPUT_DIR, "reddit_nel_results.json"),
    "emotions": os.path.join(OUTPUT_DIR, "reddit_emotions.json"),
    "keywords": os.path.join(OUTPUT_DIR, "reddit_keywords_hashtags.json"),
    "topics": os.path.join(OUTPUT_DIR, "reddit_topics.json"),
    "behavior": os.path.join(OUTPUT_DIR, "reddit_behaviour_flags.json"),
    "preprocessed": os.path.join(OUTPUT_DIR, "reddit_preprocessed.json"),
    "summaries": os.path.join(OUTPUT_DIR, "summaries.json"),
}


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                logging.info(f"✅ Loaded {len(data):,} records from {os.path.basename(file_path)}")
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
    except json.JSONDecodeError as e:
        logging.error(f"❌ Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"❌ Error loading {file_path}: {e}")
        return []


def get_post_id(record: Dict[str, Any]) -> str:
    """Extract post ID from record."""
    return str(record.get("post_id") or record.get("id") or record.get("_id", ""))


def create_post_index(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create an index of posts by post_id."""
    index = {}
    for record in data:
        post_id = get_post_id(record)
        if post_id:
            index[post_id] = record
    return index


def merge_post_data(post_id: str, preprocessed: Dict, sentiment: Dict, nel: Dict, 
                    emotion: Dict, keywords: Dict, topic: Dict, behavior: Dict, 
                    summary: Dict) -> Dict[str, Any]:
    """Merge all NLP pipeline results for a single post."""
    
    # Base post information
    post_summary = {
        "post_id": post_id,
        "title": preprocessed.get("title", "N/A"),
        "subreddit": preprocessed.get("subreddit", "unknown"),
        "author": preprocessed.get("author", "unknown"),
        "created_utc": preprocessed.get("created_utc"),
        "score": preprocessed.get("score", 0),
        "num_comments": preprocessed.get("num_comments", 0),
        "permalink": preprocessed.get("permalink", ""),
        "url": preprocessed.get("url", ""),
        "nlp_results": {}
    }
    
    # Add sentiment analysis
    if sentiment:
        sentiment_data = sentiment.get("sentiment", {})
        post_summary["nlp_results"]["sentiment_analysis"] = {
            "label": sentiment_data.get("label", "N/A"),
            "confidence": sentiment_data.get("score", 0.0),
            "all_scores": sentiment_data.get("scores", {})
        }
    
    # Add named entity linking
    if nel:
        linked_entities = nel.get("linked_entities", [])
        post_summary["nlp_results"]["named_entities"] = {
            "entity_count": nel.get("entity_count", 0),
            "linked_count": nel.get("linked_entity_count", 0),
            "entities": [
                {
                    "entity": ent.get("entity"),
                    "type": ent.get("type"),
                    "wikipedia_title": ent.get("wikipedia", {}).get("title"),
                    "wikipedia_url": ent.get("wikipedia", {}).get("url"),
                    "snippet": ent.get("wikipedia", {}).get("snippet")
                }
                for ent in linked_entities
            ]
        }
    
    # Add emotion detection
    if emotion:
        emotion_data = emotion.get("emotion", {})
        post_summary["nlp_results"]["emotion_detection"] = {
            "label": emotion_data.get("label", "N/A"),
            "confidence": emotion_data.get("score", 0.0),
            "all_scores": emotion_data.get("scores", {})
        }
    
    # Add keywords and hashtags
    if keywords:
        post_summary["nlp_results"]["keywords_hashtags"] = {
            "keywords": keywords.get("keywords", []),
            "hashtags": keywords.get("hashtags", []),
            "keyword_count": len(keywords.get("keywords", [])),
            "hashtag_count": len(keywords.get("hashtags", []))
        }
    
    # Add topic classification
    if topic:
        topic_data = topic.get("topic", {})
        post_summary["nlp_results"]["topic_classification"] = {
            "label": topic_data.get("label", "N/A"),
            "confidence": topic_data.get("score", 0.0)
        }
    
    # Add behavior flags
    if behavior:
        post_summary["nlp_results"]["behavior_flags"] = behavior.get("flags", {})
    
    # Add generated summary
    if summary:
        post_summary["nlp_results"]["generated_summary"] = summary.get("summary", "")
    
    return post_summary


def generate_post_summaries() -> Dict[str, Any]:
    """Generate comprehensive summaries for each post/title."""
    logging.info("🚀 Starting post-wise summary generation...")
    
    # Load all data
    logging.info("\n📂 Loading all pipeline outputs...")
    sentiment_data = load_json_file(OUTPUT_FILES["sentiment"])
    nel_data = load_json_file(OUTPUT_FILES["nel"])
    emotion_data = load_json_file(OUTPUT_FILES["emotions"])
    keywords_data = load_json_file(OUTPUT_FILES["keywords"])
    topics_data = load_json_file(OUTPUT_FILES["topics"])
    behavior_data = load_json_file(OUTPUT_FILES["behavior"])
    summaries_data = load_json_file(OUTPUT_FILES["summaries"])
    preprocessed_data = load_json_file(OUTPUT_FILES["preprocessed"])
    
    # Create indices by post_id
    logging.info("\n📊 Indexing data by post_id...")
    preprocessed_index = create_post_index(preprocessed_data)
    sentiment_index = create_post_index(sentiment_data)
    nel_index = create_post_index(nel_data)
    emotion_index = create_post_index(emotion_data)
    keywords_index = create_post_index(keywords_data)
    topics_index = create_post_index(topics_data)
    behavior_index = create_post_index(behavior_data)
    summaries_index = create_post_index(summaries_data)
    
    # Get all unique post IDs
    all_post_ids = set(preprocessed_index.keys())
    logging.info(f"Found {len(all_post_ids)} unique posts")
    
    # Generate summary for each post
    post_summaries = []
    
    for post_id in tqdm(all_post_ids, desc="📝 Generating post summaries"):
        post_summary = merge_post_data(
            post_id=post_id,
            preprocessed=preprocessed_index.get(post_id, {}),
            sentiment=sentiment_index.get(post_id, {}),
            nel=nel_index.get(post_id, {}),
            emotion=emotion_index.get(post_id, {}),
            keywords=keywords_index.get(post_id, {}),
            topic=topics_index.get(post_id, {}),
            behavior=behavior_index.get(post_id, {}),
            summary=summaries_index.get(post_id, {})
        )
        post_summaries.append(post_summary)
    
    # Sort by subreddit and then by title
    post_summaries.sort(key=lambda x: (x.get("subreddit", ""), x.get("title", "")))
    
    # Create final output structure
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_posts": len(post_summaries),
        "posts": post_summaries
    }
    
    return output


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    """Save post summaries to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Saved {data.get('total_posts', 0):,} post summaries to {file_path}")
        
    except Exception as e:
        logging.error(f"❌ Error saving to JSON file: {e}")


def save_to_mongodb(data: Dict[str, Any]) -> None:
    """Save post summaries to MongoDB."""
    try:
        collection = db["post_summaries"]
        
        # Save individual post summaries
        posts = data.get("posts", [])
        if posts:
            for post in posts:
                doc = {
                    "_id": post.get("post_id"),
                    **post,
                    "last_updated": datetime.now(timezone.utc)
                }
                collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        
        logging.info(f"💾 Saved {len(posts):,} post summaries to MongoDB (collection: post_summaries)")
        
    except Exception as e:
        logging.error(f"❌ Error saving to MongoDB: {e}")


def print_summary_overview(data: Dict[str, Any]) -> None:
    """Print overview of post summaries."""
    logging.info(f"\n{'='*70}")
    logging.info("📊 POST-WISE NLP PIPELINE SUMMARIES")
    logging.info(f"{'='*70}")
    logging.info(f"Generated at: {data.get('generated_at')}")
    logging.info(f"Total posts analyzed: {data.get('total_posts'):,}")
    
    # Show sample posts
    posts = data.get("posts", [])
    if posts:
        logging.info(f"\n📄 Sample Posts (first 5):")
        for i, post in enumerate(posts[:5], 1):
            logging.info(f"\n{i}. {post.get('title', 'N/A')[:80]}...")
            logging.info(f"   Subreddit: r/{post.get('subreddit')}")
            logging.info(f"   Author: u/{post.get('author')}")
            logging.info(f"   Score: {post.get('score'):,} | Comments: {post.get('num_comments'):,}")
            
            nlp = post.get("nlp_results", {})
            
            # Sentiment
            sentiment = nlp.get("sentiment_analysis", {})
            if sentiment:
                logging.info(f"   Sentiment: {sentiment.get('label')} (confidence: {sentiment.get('confidence', 0):.3f})")
            
            # Emotion
            emotion = nlp.get("emotion_detection", {})
            if emotion:
                logging.info(f"   Emotion: {emotion.get('label')} (confidence: {emotion.get('confidence', 0):.3f})")
            
            # Topic
            topic = nlp.get("topic_classification", {})
            if topic:
                logging.info(f"   Topic: {topic.get('label')}")
            
            # Keywords
            kw = nlp.get("keywords_hashtags", {})
            if kw and kw.get("keywords"):
                top_kw = kw.get("keywords", [])[:5]
                logging.info(f"   Top Keywords: {', '.join(top_kw)}")
            
            # Entities
            entities = nlp.get("named_entities", {})
            if entities and entities.get("entities"):
                ent_names = [e.get("entity") for e in entities.get("entities", [])[:3]]
                logging.info(f"   Named Entities: {', '.join(ent_names)}")
    
    logging.info(f"\n{'='*70}\n")


def main():
    """Main function to generate post summaries."""
    try:
        # Generate summaries
        summaries = generate_post_summaries()
        
        if not summaries.get("posts"):
            logging.warning("⚠️  No post data found. Please run the processing pipelines first.")
            return
        
        # Save to JSON
        save_to_json(summaries, OUTPUT_FILE)
        
        # Save to MongoDB
        save_to_mongodb(summaries)
        
        # Print overview
        print_summary_overview(summaries)
        
        logging.info("✅ Post summary generation complete!")
        
    except Exception as e:
        logging.error(f"❌ Error generating post summaries: {e}")
        raise


if __name__ == "__main__":
    main()
