"""
Reddit Data Pipeline Overview Generator

Aggregates all processing outputs into a comprehensive overview/summary.
Saves results to both JSON file and MongoDB.

Processes:
- Sentiment Analysis
- Named Entity Linking (NEL)
- Emotion Detection
- Keywords & Hashtags
- Topics
- Behavior Flags
- Summaries (if available)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import Counter

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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_overview.json")

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

# MongoDB collections mapping
MONGO_COLLECTIONS = {
    "sentiment": "sentiment_results",
    "nel": "nel_results",
    "emotions": "emotion_results",
    "keywords": "keyword_and_hashtag_results",
    "topics": "topic_results",
    "behavior": "behaviour_flags",
    "preprocessed": "posts",
    "summaries": "summaries",
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
                logging.info(f"✅ Loaded dictionary from {os.path.basename(file_path)}")
                return [data]
            else:
                logging.warning(f"⚠️  Unexpected data type in {file_path}")
                return []
    except json.JSONDecodeError as e:
        logging.error(f"❌ Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"❌ Error loading {file_path}: {e}")
        return []


def analyze_sentiment_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze sentiment analysis results."""
    if not data:
        return {}
    
    sentiment_counts = Counter()
    total_confidence = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    confidence_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    
    for record in data:
        sentiment = record.get("sentiment", {})
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
        "total_analyzed": len(data),
        "sentiment_distribution": dict(sentiment_counts),
        "sentiment_percentages": {
            label: round(count / len(data) * 100, 2)
            for label, count in sentiment_counts.items()
        },
        "average_confidence": avg_confidence
    }


def analyze_nel_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Named Entity Linking results."""
    if not data:
        return {}
    
    total_entities = 0
    total_linked = 0
    entity_types = Counter()
    
    for record in data:
        total_entities += record.get("entity_count", 0)
        total_linked += record.get("linked_entity_count", 0)
        
        for entity in record.get("linked_entities", []):
            entity_type = entity.get("type", "UNKNOWN")
            entity_types[entity_type] += 1
    
    return {
        "total_posts_with_entities": len(data),
        "total_entities_found": total_entities,
        "total_entities_linked": total_linked,
        "link_success_rate": round(total_linked / total_entities * 100, 2) if total_entities > 0 else 0,
        "entity_type_distribution": dict(entity_types.most_common(10)),
        "avg_entities_per_post": round(total_entities / len(data), 2) if len(data) > 0 else 0
    }


def analyze_emotion_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze emotion detection results."""
    if not data:
        return {}
    
    emotion_counts = Counter()
    total_confidence = Counter()
    confidence_counts = Counter()
    
    for record in data:
        emotion = record.get("emotion", {})
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
        "total_analyzed": len(data),
        "emotion_distribution": dict(emotion_counts.most_common()),
        "emotion_percentages": {
            label: round(count / len(data) * 100, 2)
            for label, count in emotion_counts.items()
        },
        "average_confidence": avg_confidence
    }


def analyze_keywords_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze keywords and hashtags results."""
    if not data:
        return {}
    
    total_keywords = 0
    total_hashtags = 0
    all_keywords = Counter()
    all_hashtags = Counter()
    
    for record in data:
        keywords = record.get("keywords", [])
        hashtags = record.get("hashtags", [])
        
        total_keywords += len(keywords)
        total_hashtags += len(hashtags)
        
        for kw in keywords:
            all_keywords[kw] += 1
        for ht in hashtags:
            all_hashtags[ht] += 1
    
    return {
        "total_posts_analyzed": len(data),
        "total_keywords_extracted": total_keywords,
        "total_hashtags_extracted": total_hashtags,
        "avg_keywords_per_post": round(total_keywords / len(data), 2) if len(data) > 0 else 0,
        "avg_hashtags_per_post": round(total_hashtags / len(data), 2) if len(data) > 0 else 0,
        "top_keywords": dict(all_keywords.most_common(20)),
        "top_hashtags": dict(all_hashtags.most_common(20))
    }


def analyze_topics_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze topic detection results."""
    if not data:
        return {}
    
    topic_counts = Counter()
    
    for record in data:
        topic = record.get("topic", {})
        label = topic.get("label", "Unknown")
        topic_counts[label] += 1
    
    return {
        "total_posts_analyzed": len(data),
        "topic_distribution": dict(topic_counts.most_common()),
        "topic_percentages": {
            label: round(count / len(data) * 100, 2)
            for label, count in topic_counts.items()
        }
    }


def analyze_behavior_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze behavior flags results."""
    if not data:
        return {}
    
    flag_counts = {
        "spam": 0,
        "offensive": 0,
        "promotional": 0,
        "question": 0,
        "discussion": 0
    }
    
    for record in data:
        flags = record.get("flags", {})
        for flag, value in flags.items():
            if flag in flag_counts and value:
                flag_counts[flag] += 1
    
    return {
        "total_posts_analyzed": len(data),
        "behavior_flag_counts": flag_counts,
        "behavior_flag_percentages": {
            flag: round(count / len(data) * 100, 2)
            for flag, count in flag_counts.items()
        }
    }


def analyze_summaries_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze summary generation results."""
    if not data:
        return {}
    
    total_summaries = len(data)
    avg_summary_length = 0
    
    for record in data:
        summary = record.get("summary", "")
        avg_summary_length += len(summary.split())
    
    return {
        "total_summaries_generated": total_summaries,
        "avg_summary_length_words": round(avg_summary_length / total_summaries, 2) if total_summaries > 0 else 0
    }


def generate_overview() -> Dict[str, Any]:
    """Generate comprehensive overview of all processing results."""
    logging.info("🚀 Starting overview generation...")
    
    overview = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_version": "1.0",
        "data_sources": []
    }
    
    # Load and analyze each data source
    logging.info("\n📊 Analyzing Sentiment Analysis Results...")
    sentiment_data = load_json_file(OUTPUT_FILES["sentiment"])
    if sentiment_data:
        overview["sentiment_analysis"] = analyze_sentiment_data(sentiment_data)
        overview["data_sources"].append("sentiment_analysis")
    
    logging.info("\n📊 Analyzing Named Entity Linking Results...")
    nel_data = load_json_file(OUTPUT_FILES["nel"])
    if nel_data:
        overview["named_entity_linking"] = analyze_nel_data(nel_data)
        overview["data_sources"].append("named_entity_linking")
    
    logging.info("\n📊 Analyzing Emotion Detection Results...")
    emotion_data = load_json_file(OUTPUT_FILES["emotions"])
    if emotion_data:
        overview["emotion_detection"] = analyze_emotion_data(emotion_data)
        overview["data_sources"].append("emotion_detection")
    
    logging.info("\n📊 Analyzing Keywords & Hashtags Results...")
    keywords_data = load_json_file(OUTPUT_FILES["keywords"])
    if keywords_data:
        overview["keywords_hashtags"] = analyze_keywords_data(keywords_data)
        overview["data_sources"].append("keywords_hashtags")
    
    logging.info("\n📊 Analyzing Topic Detection Results...")
    topics_data = load_json_file(OUTPUT_FILES["topics"])
    if topics_data:
        overview["topic_detection"] = analyze_topics_data(topics_data)
        overview["data_sources"].append("topic_detection")
    
    logging.info("\n📊 Analyzing Behavior Flags Results...")
    behavior_data = load_json_file(OUTPUT_FILES["behavior"])
    if behavior_data:
        overview["behavior_flags"] = analyze_behavior_data(behavior_data)
        overview["data_sources"].append("behavior_flags")
    
    logging.info("\n📊 Analyzing Summaries Results...")
    summaries_data = load_json_file(OUTPUT_FILES["summaries"])
    if summaries_data:
        overview["summaries"] = analyze_summaries_data(summaries_data)
        overview["data_sources"].append("summaries")
    
    # Add overall statistics
    preprocessed_data = load_json_file(OUTPUT_FILES["preprocessed"])
    if preprocessed_data:
        overview["total_posts_in_dataset"] = len(preprocessed_data)
    
    return overview


def save_to_json(overview: Dict[str, Any], file_path: str) -> None:
    """Save overview to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(overview, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Saved overview to {file_path}")
    except Exception as e:
        logging.error(f"❌ Error saving to JSON file: {e}")


def save_to_mongodb(overview: Dict[str, Any]) -> None:
    """Save overview to MongoDB."""
    try:
        collection = db["pipeline_overview"]
        
        # Add MongoDB-specific metadata
        overview_doc = {
            **overview,
            "_id": f"overview_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "saved_at": datetime.now(timezone.utc)
        }
        
        # Insert or update
        collection.replace_one(
            {"_id": overview_doc["_id"]},
            overview_doc,
            upsert=True
        )
        
        logging.info(f"💾 Saved overview to MongoDB (collection: pipeline_overview)")
    except Exception as e:
        logging.error(f"❌ Error saving to MongoDB: {e}")


def print_overview_summary(overview: Dict[str, Any]) -> None:
    """Print a formatted summary of the overview."""
    logging.info(f"\n{'='*70}")
    logging.info("📊 REDDIT DATA PIPELINE OVERVIEW")
    logging.info(f"{'='*70}")
    logging.info(f"Generated at: {overview.get('generated_at')}")
    logging.info(f"Data sources processed: {len(overview.get('data_sources', []))}")
    logging.info(f"Total posts in dataset: {overview.get('total_posts_in_dataset', 'N/A'):,}")
    
    # Sentiment Analysis
    if "sentiment_analysis" in overview:
        sa = overview["sentiment_analysis"]
        logging.info(f"\n📈 Sentiment Analysis:")
        logging.info(f"   Posts analyzed: {sa.get('total_analyzed', 0):,}")
        logging.info(f"   Distribution: {sa.get('sentiment_percentages', {})}")
    
    # Named Entity Linking
    if "named_entity_linking" in overview:
        nel = overview["named_entity_linking"]
        logging.info(f"\n🔗 Named Entity Linking:")
        logging.info(f"   Posts with entities: {nel.get('total_posts_with_entities', 0):,}")
        logging.info(f"   Entities linked: {nel.get('total_entities_linked', 0):,}")
        logging.info(f"   Link success rate: {nel.get('link_success_rate', 0)}%")
    
    # Emotion Detection
    if "emotion_detection" in overview:
        ed = overview["emotion_detection"]
        logging.info(f"\n😊 Emotion Detection:")
        logging.info(f"   Posts analyzed: {ed.get('total_analyzed', 0):,}")
        top_emotions = list(ed.get('emotion_distribution', {}).items())[:3]
        logging.info(f"   Top emotions: {top_emotions}")
    
    # Keywords & Hashtags
    if "keywords_hashtags" in overview:
        kh = overview["keywords_hashtags"]
        logging.info(f"\n🔑 Keywords & Hashtags:")
        logging.info(f"   Posts analyzed: {kh.get('total_posts_analyzed', 0):,}")
        logging.info(f"   Keywords extracted: {kh.get('total_keywords_extracted', 0):,}")
        logging.info(f"   Hashtags extracted: {kh.get('total_hashtags_extracted', 0):,}")
    
    # Topic Detection
    if "topic_detection" in overview:
        td = overview["topic_detection"]
        logging.info(f"\n📚 Topic Detection:")
        logging.info(f"   Posts analyzed: {td.get('total_posts_analyzed', 0):,}")
        top_topics = list(td.get('topic_distribution', {}).items())[:5]
        logging.info(f"   Top topics: {top_topics}")
    
    # Behavior Flags
    if "behavior_flags" in overview:
        bf = overview["behavior_flags"]
        logging.info(f"\n🚩 Behavior Flags:")
        logging.info(f"   Posts analyzed: {bf.get('total_posts_analyzed', 0):,}")
        logging.info(f"   Flag counts: {bf.get('behavior_flag_counts', {})}")
    
    # Summaries
    if "summaries" in overview:
        sm = overview["summaries"]
        logging.info(f"\n📝 Summaries:")
        logging.info(f"   Summaries generated: {sm.get('total_summaries_generated', 0):,}")
        logging.info(f"   Avg length: {sm.get('avg_summary_length_words', 0)} words")
    
    logging.info(f"\n{'='*70}\n")


def main():
    """Main function to generate and save overview."""
    try:
        # Generate overview
        overview = generate_overview()
        
        if not overview.get("data_sources"):
            logging.warning("⚠️  No data sources found. Please run the processing pipelines first.")
            return
        
        # Save to JSON
        save_to_json(overview, OUTPUT_FILE)
        
        # Save to MongoDB
        save_to_mongodb(overview)
        
        # Print summary
        print_overview_summary(overview)
        
        logging.info("✅ Overview generation complete!")
        
    except Exception as e:
        logging.error(f"❌ Error generating overview: {e}")
        raise


if __name__ == "__main__":
    main()
