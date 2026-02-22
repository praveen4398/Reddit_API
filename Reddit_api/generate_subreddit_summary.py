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
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

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
    return record.get("post_id") or record.get("id") or record.get("_id", "")


def group_by_subreddit(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by subreddit."""
    grouped = defaultdict(list)
    for record in data:
        subreddit = record.get("subreddit", "unknown")
        if subreddit:
            grouped[subreddit].append(record)
    return dict(grouped)


def analyze_subreddit_sentiment(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze sentiment for a subreddit."""
    if not posts:
        return {}
    
    sentiment_counts = Counter()
    total_confidence = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    confidence_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    
    for post in posts:
        sentiment = post.get("sentiment", {})
        label = sentiment.get("label", "Unknown")
        score = sentiment.get("score", 0.0)
        
        sentiment_counts[label] += 1
        if label in total_confidence:
            total_confidence[label] += score
            confidence_counts[label] += 1
    
    # Calculate averages
    avg_confidence = {}
    for label in ["Positive", "Neutral", "Negative"]:
        if confidence_counts[label] > 0:
            avg_confidence[label] = round(total_confidence[label] / confidence_counts[label], 3)
        else:
            avg_confidence[label] = 0.0
    
    return {
        "total_posts": len(posts),
        "sentiment_distribution": dict(sentiment_counts),
        "sentiment_percentages": {
            label: round(count / len(posts) * 100, 2)
            for label, count in sentiment_counts.items()
        },
        "average_confidence": avg_confidence,
        "dominant_sentiment": sentiment_counts.most_common(1)[0][0] if sentiment_counts else "Unknown"
    }


def analyze_subreddit_entities(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze named entities for a subreddit."""
    if not posts:
        return {}
    
    total_entities = 0
    total_linked = 0
    entity_types = Counter()
    all_entities = Counter()
    
    for post in posts:
        total_entities += post.get("entity_count", 0)
        total_linked += post.get("linked_entity_count", 0)
        
        for entity in post.get("linked_entities", []):
            entity_type = entity.get("type", "UNKNOWN")
            entity_text = entity.get("entity", "")
            entity_types[entity_type] += 1
            if entity_text:
                all_entities[entity_text] += 1
    
    return {
        "total_posts_with_entities": len(posts),
        "total_entities_found": total_entities,
        "total_entities_linked": total_linked,
        "link_success_rate": round(total_linked / total_entities * 100, 2) if total_entities > 0 else 0,
        "entity_type_distribution": dict(entity_types.most_common(10)),
        "top_entities": dict(all_entities.most_common(15)),
        "avg_entities_per_post": round(total_entities / len(posts), 2) if len(posts) > 0 else 0
    }


def analyze_subreddit_emotions(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze emotions for a subreddit."""
    if not posts:
        return {}
    
    emotion_counts = Counter()
    total_confidence = Counter()
    confidence_counts = Counter()
    
    for post in posts:
        emotion = post.get("emotion", {})
        label = emotion.get("label", "Unknown")
        score = emotion.get("score", 0.0)
        
        emotion_counts[label] += 1
        total_confidence[label] += score
        confidence_counts[label] += 1
    
    # Calculate averages
    avg_confidence = {}
    for label in emotion_counts.keys():
        if confidence_counts[label] > 0:
            avg_confidence[label] = round(total_confidence[label] / confidence_counts[label], 3)
    
    return {
        "total_posts": len(posts),
        "emotion_distribution": dict(emotion_counts.most_common()),
        "emotion_percentages": {
            label: round(count / len(posts) * 100, 2)
            for label, count in emotion_counts.items()
        },
        "average_confidence": avg_confidence,
        "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else "Unknown"
    }


def analyze_subreddit_keywords(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze keywords and hashtags for a subreddit."""
    if not posts:
        return {}
    
    total_keywords = 0
    total_hashtags = 0
    all_keywords = Counter()
    all_hashtags = Counter()
    
    for post in posts:
        keywords = post.get("keywords", [])
        hashtags = post.get("hashtags", [])
        
        total_keywords += len(keywords)
        total_hashtags += len(hashtags)
        
        for kw in keywords:
            all_keywords[kw] += 1
        for ht in hashtags:
            all_hashtags[ht] += 1
    
    return {
        "total_posts": len(posts),
        "total_keywords_extracted": total_keywords,
        "total_hashtags_extracted": total_hashtags,
        "avg_keywords_per_post": round(total_keywords / len(posts), 2) if len(posts) > 0 else 0,
        "avg_hashtags_per_post": round(total_hashtags / len(posts), 2) if len(posts) > 0 else 0,
        "top_keywords": dict(all_keywords.most_common(30)),
        "top_hashtags": dict(all_hashtags.most_common(20))
    }


def analyze_subreddit_topics(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze topics for a subreddit."""
    if not posts:
        return {}
    
    topic_counts = Counter()
    
    for post in posts:
        topic = post.get("topic", {})
        label = topic.get("label", "Unknown")
        topic_counts[label] += 1
    
    return {
        "total_posts": len(posts),
        "topic_distribution": dict(topic_counts.most_common()),
        "topic_percentages": {
            label: round(count / len(posts) * 100, 2)
            for label, count in topic_counts.items()
        },
        "dominant_topic": topic_counts.most_common(1)[0][0] if topic_counts else "Unknown"
    }


def analyze_subreddit_behavior(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze behavior flags for a subreddit."""
    if not posts:
        return {}
    
    flag_counts = {
        "spam": 0,
        "offensive": 0,
        "promotional": 0,
        "question": 0,
        "discussion": 0
    }
    
    for post in posts:
        flags = post.get("flags", {})
        for flag, value in flags.items():
            if flag in flag_counts and value:
                flag_counts[flag] += 1
    
    return {
        "total_posts": len(posts),
        "behavior_flag_counts": flag_counts,
        "behavior_flag_percentages": {
            flag: round(count / len(posts) * 100, 2)
            for flag, count in flag_counts.items()
        }
    }


def analyze_subreddit_summaries(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze summaries for a subreddit."""
    if not posts:
        return {}
    
    total_summaries = len(posts)
    avg_summary_length = 0
    
    for post in posts:
        summary = post.get("summary", "")
        avg_summary_length += len(summary.split())
    
    return {
        "total_summaries": total_summaries,
        "avg_summary_length_words": round(avg_summary_length / total_summaries, 2) if total_summaries > 0 else 0
    }


def get_subreddit_metadata(subreddit: str, preprocessed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get metadata about a subreddit from preprocessed data."""
    subreddit_posts = [p for p in preprocessed_data if p.get("subreddit") == subreddit]
    
    if not subreddit_posts:
        return {
            "total_posts": 0,
            "description": f"Subreddit: r/{subreddit}"
        }
    
    # Get post statistics
    authors = set()
    total_score = 0
    total_comments = 0
    
    for post in subreddit_posts:
        author = post.get("author")
        if author:
            authors.add(author)
        total_score += post.get("score", 0)
        total_comments += post.get("num_comments", 0)
    
    return {
        "subreddit_name": f"r/{subreddit}",
        "total_posts": len(subreddit_posts),
        "unique_authors": len(authors),
        "avg_score": round(total_score / len(subreddit_posts), 2) if subreddit_posts else 0,
        "avg_comments": round(total_comments / len(subreddit_posts), 2) if subreddit_posts else 0,
        "total_score": total_score,
        "total_comments": total_comments
    }


def generate_subreddit_summaries() -> Dict[str, Any]:
    """Generate comprehensive summaries for each subreddit."""
    logging.info("🚀 Starting subreddit-wise summary generation...")
    
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
    
    # Group by subreddit
    logging.info("\n📊 Grouping data by subreddit...")
    sentiment_by_sub = group_by_subreddit(sentiment_data)
    nel_by_sub = group_by_subreddit(nel_data)
    emotion_by_sub = group_by_subreddit(emotion_data)
    keywords_by_sub = group_by_subreddit(keywords_data)
    topics_by_sub = group_by_subreddit(topics_data)
    behavior_by_sub = group_by_subreddit(behavior_data)
    summaries_by_sub = group_by_subreddit(summaries_data)
    
    # Get all unique subreddits
    all_subreddits = set()
    all_subreddits.update(sentiment_by_sub.keys())
    all_subreddits.update(nel_by_sub.keys())
    all_subreddits.update(emotion_by_sub.keys())
    all_subreddits.update(keywords_by_sub.keys())
    all_subreddits.update(topics_by_sub.keys())
    all_subreddits.update(behavior_by_sub.keys())
    all_subreddits.update(summaries_by_sub.keys())
    all_subreddits.discard("unknown")
    all_subreddits.discard("")
    all_subreddits.discard(None)
    
    logging.info(f"Found {len(all_subreddits)} unique subreddits")
    
    # Generate summary for each subreddit
    subreddit_summaries = {}
    
    for subreddit in tqdm(sorted(all_subreddits), desc="📝 Generating subreddit summaries"):
        summary = {
            "subreddit": subreddit,
            "metadata": get_subreddit_metadata(subreddit, preprocessed_data),
            "nlp_pipeline_results": {}
        }
        
        # Add sentiment analysis
        if subreddit in sentiment_by_sub:
            summary["nlp_pipeline_results"]["sentiment_analysis"] = analyze_subreddit_sentiment(
                sentiment_by_sub[subreddit]
            )
        
        # Add named entity linking
        if subreddit in nel_by_sub:
            summary["nlp_pipeline_results"]["named_entity_linking"] = analyze_subreddit_entities(
                nel_by_sub[subreddit]
            )
        
        # Add emotion detection
        if subreddit in emotion_by_sub:
            summary["nlp_pipeline_results"]["emotion_detection"] = analyze_subreddit_emotions(
                emotion_by_sub[subreddit]
            )
        
        # Add keywords & hashtags
        if subreddit in keywords_by_sub:
            summary["nlp_pipeline_results"]["keywords_hashtags"] = analyze_subreddit_keywords(
                keywords_by_sub[subreddit]
            )
        
        # Add topic detection
        if subreddit in topics_by_sub:
            summary["nlp_pipeline_results"]["topic_detection"] = analyze_subreddit_topics(
                topics_by_sub[subreddit]
            )
        
        # Add behavior flags
        if subreddit in behavior_by_sub:
            summary["nlp_pipeline_results"]["behavior_flags"] = analyze_subreddit_behavior(
                behavior_by_sub[subreddit]
            )
        
        # Add summaries
        if subreddit in summaries_by_sub:
            summary["nlp_pipeline_results"]["summaries"] = analyze_subreddit_summaries(
                summaries_by_sub[subreddit]
            )
        
        subreddit_summaries[subreddit] = summary
    
    # Create final output structure
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_subreddits": len(subreddit_summaries),
        "subreddits": subreddit_summaries
    }
    
    return output


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    """Save subreddit summaries to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Saved subreddit summaries to {file_path}")
        
        # Also save individual subreddit files
        subreddits_dir = os.path.join(OUTPUT_DIR, "subreddit_summaries")
        os.makedirs(subreddits_dir, exist_ok=True)
        
        for subreddit, summary in data.get("subreddits", {}).items():
            subreddit_file = os.path.join(subreddits_dir, f"{subreddit}_summary.json")
            with open(subreddit_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logging.info(f"💾 Saved individual subreddit files to {subreddits_dir}/")
        
    except Exception as e:
        logging.error(f"❌ Error saving to JSON file: {e}")


def save_to_mongodb(data: Dict[str, Any]) -> None:
    """Save subreddit summaries to MongoDB."""
    try:
        collection = db["subreddit_summaries"]
        
        # Save overall summary
        overall_doc = {
            "_id": f"summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "generated_at": data.get("generated_at"),
            "total_subreddits": data.get("total_subreddits"),
            "saved_at": datetime.now(timezone.utc)
        }
        collection.replace_one({"_id": overall_doc["_id"]}, overall_doc, upsert=True)
        
        # Save individual subreddit summaries
        for subreddit, summary in data.get("subreddits", {}).items():
            doc = {
                "_id": subreddit,
                **summary,
                "last_updated": datetime.now(timezone.utc)
            }
            collection.replace_one({"_id": subreddit}, doc, upsert=True)
        
        logging.info(f"💾 Saved {len(data.get('subreddits', {}))} subreddit summaries to MongoDB")
        
    except Exception as e:
        logging.error(f"❌ Error saving to MongoDB: {e}")


def print_summary_overview(data: Dict[str, Any]) -> None:
    """Print overview of subreddit summaries."""
    logging.info(f"\n{'='*70}")
    logging.info("📊 SUBREDDIT-WISE NLP PIPELINE SUMMARIES")
    logging.info(f"{'='*70}")
    logging.info(f"Generated at: {data.get('generated_at')}")
    logging.info(f"Total subreddits analyzed: {data.get('total_subreddits')}")
    
    # Show top 10 subreddits by post count
    subreddits = data.get("subreddits", {})
    sorted_subs = sorted(
        subreddits.items(),
        key=lambda x: x[1].get("metadata", {}).get("total_posts", 0),
        reverse=True
    )[:10]
    
    logging.info(f"\n📈 Top 10 Subreddits by Post Count:")
    for i, (subreddit, summary) in enumerate(sorted_subs, 1):
        metadata = summary.get("metadata", {})
        nlp = summary.get("nlp_pipeline_results", {})
        
        logging.info(f"\n{i}. r/{subreddit}")
        logging.info(f"   Posts: {metadata.get('total_posts', 0):,}")
        logging.info(f"   Authors: {metadata.get('unique_authors', 0):,}")
        
        # Sentiment
        sentiment = nlp.get("sentiment_analysis", {})
        if sentiment:
            logging.info(f"   Dominant Sentiment: {sentiment.get('dominant_sentiment', 'N/A')}")
        
        # Emotion
        emotion = nlp.get("emotion_detection", {})
        if emotion:
            logging.info(f"   Dominant Emotion: {emotion.get('dominant_emotion', 'N/A')}")
        
        # Topic
        topic = nlp.get("topic_detection", {})
        if topic:
            logging.info(f"   Dominant Topic: {topic.get('dominant_topic', 'N/A')}")
    
    logging.info(f"\n{'='*70}\n")


def main():
    """Main function to generate subreddit summaries."""
    try:
        # Generate summaries
        summaries = generate_subreddit_summaries()
        
        if not summaries.get("subreddits"):
            logging.warning("⚠️  No subreddit data found. Please run the processing pipelines first.")
            return
        
        # Save to JSON
        save_to_json(summaries, OUTPUT_FILE)
        
        # Save to MongoDB
        save_to_mongodb(summaries)
        
        # Print overview
        print_summary_overview(summaries)
        
        logging.info("✅ Subreddit summary generation complete!")
        
    except Exception as e:
        logging.error(f"❌ Error generating subreddit summaries: {e}")
        raise


if __name__ == "__main__":
    main()
