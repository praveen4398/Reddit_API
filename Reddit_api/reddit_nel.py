"""
Named Entity Linking (NEL) Module for Reddit Data

Performs Named Entity Recognition (NER) and links entities to Wikipedia pages.
- Reads input from MongoDB (reddit_db.posts) or output/Reddit_data.json
- Performs NER using spaCy
- Links entities to Wikipedia using Wikipedia API
- Saves results to MongoDB (reddit_db.nel_results) and output/reddit_nel_results.json
"""

import os
import re
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import requests
import spacy
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
nel_collection = os.getenv("NEL_COLLECTION", "nel_results")

client = MongoClient(mongo_uri)
db = client[db_name]
col_in = db[source_collection]
col_out = db[nel_collection]

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INPUT_FILE_FALLBACK = os.path.join(OUTPUT_DIR, "reddit_preprocessed.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_nel_results.json")

# Wikipedia API constants
WIKIPEDIA_API_DELAY = 0.5  # Delay between API calls in seconds
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("✅ Loaded spaCy model: en_core_web_sm")
except OSError:
    logging.error("❌ spaCy model 'en_core_web_sm' not found. Please install it:")
    logging.error("   python -m spacy download en_core_web_sm")
    raise


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        List[Dict]: List of entities with text, label, and position
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return []
    
    try:
        doc = nlp(text[:1000000])  # Limit text length for performance
        entities = []
        
        for ent in doc.ents:
            # Filter out very short entities and common noise
            if len(ent.text.strip()) < 2:
                continue
                
            entities.append({
                "entity": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    except Exception as e:
        logging.warning(f"⚠️  Entity extraction failed: {e}")
        return []


def link_entity_wikipedia(entity: str, entity_type: str = "") -> Optional[Dict[str, str]]:
    """
    Link an entity to a Wikipedia page using the Wikipedia API.
    
    Args:
        entity (str): The entity text to link
        entity_type (str): The entity type/label (e.g., PERSON, ORG, GPE)
        
    Returns:
        Optional[Dict[str, str]]: Dictionary with 'title', 'url', and 'snippet' if found
    """
    if not entity or len(entity) < 2:
        return None
    
    # Clean entity text
    entity_clean = entity.strip()
    
    params = {
        "action": "query",
        "list": "search",
        "srsearch": entity_clean,
        "format": "json",
        "srlimit": 1,
        "srprop": "snippet"
    }
    
    try:
        # Add delay to respect Wikipedia's API rate limits
        time.sleep(WIKIPEDIA_API_DELAY)
        
        resp = requests.get(WIKIPEDIA_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        
        results = resp.json().get("query", {}).get("search", [])
        if results:
            title = results[0]["title"]
            pageid = results[0]["pageid"]
            snippet = results[0].get("snippet", "")
            
            # Clean HTML tags from snippet
            snippet = re.sub(r'<[^>]+>', '', snippet)
            
            return {
                "title": title,
                "url": f"https://en.wikipedia.org/?curid={pageid}",
                "snippet": snippet[:200]  # Limit snippet length
            }
    except requests.RequestException as e:
        logging.debug(f"⚠️  Wikipedia API request failed for '{entity}': {e}")
    except (KeyError, json.JSONDecodeError) as e:
        logging.debug(f"⚠️  Error parsing Wikipedia API response for '{entity}': {e}")
    
    return None


def process_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a list of entities and link them to Wikipedia.
    
    Args:
        entities (List[Dict]): List of entity dictionaries
        
    Returns:
        List[Dict]: List of processed entities with Wikipedia links
    """
    if not entities:
        return []
    
    results = []
    seen_entities = set()
    
    for ent in entities:
        entity_text = ent.get("entity", "").strip()
        if not entity_text or len(entity_text) < 2:
            continue
        
        # Skip duplicate entities (case-insensitive)
        entity_key = entity_text.lower()
        if entity_key in seen_entities:
            continue
        
        seen_entities.add(entity_key)
        
        # Get Wikipedia link
        entity_type = ent.get("label", "")
        link = link_entity_wikipedia(entity_text, entity_type)
        
        result = {
            "entity": entity_text,
            "type": entity_type,
            "start": ent.get("start"),
            "end": ent.get("end")
        }
        
        if link:
            result["wikipedia"] = link
            results.append(result)
    
    return results


def process_post(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single Reddit post to extract and link named entities.
    
    Args:
        doc (dict): Reddit post data
        
    Returns:
        dict: Processed post with linked entities or None if no entities found
    """
    try:
        # Get text content (prefer translated_text, then cleaned_text, then title + body/content)
        text = doc.get("translated_text", "")
        
        if not text:
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
        
        if not text or len(text) < 10:
            return None
        
        # Extract entities using NER
        entities = extract_entities(text)
        if not entities:
            return None
        
        # Link entities to Wikipedia
        linked_entities = process_entities(entities)
        if not linked_entities:
            return None
        
        return {
            "_id": doc.get("_id"),
            "post_id": doc.get("post_id") or doc.get("id"),
            "subreddit": doc.get("subreddit"),
            "author": doc.get("author"),
            "created_utc": doc.get("created_utc"),
            "permalink": doc.get("permalink"),
            "title": doc.get("title", ""),
            "text_analyzed": text[:500],  # Store snippet of analyzed text
            "entity_count": len(entities),
            "linked_entity_count": len(linked_entities),
            "linked_entities": linked_entities,
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
    Process a batch of Reddit posts for NEL.
    
    Args:
        posts (list): List of Reddit post dictionaries
        
    Returns:
        list: List of processed posts with linked entities
    """
    results = []
    
    for post in tqdm(posts, desc="🔗 Extracting & linking entities"):
        result = process_post(post)
        if result:
            results.append(result)
    
    return results


def save_to_mongo(results: List[Dict[str, Any]]) -> None:
    """Save NEL results to MongoDB."""
    if not results:
        return
    
    try:
        ops: List[UpdateOne] = []
        for r in results:
            key = {"_id": r.get("_id")} if r.get("_id") is not None else {"post_id": r.get("post_id")}
            ops.append(UpdateOne(key, {"$set": r}, upsert=True))
        
        if ops:
            col_out.bulk_write(ops, ordered=False)
            logging.info(f"💾 Saved {len(results):,} NEL results to MongoDB ({nel_collection})")
    except Exception as e:
        logging.error(f"❌ Error saving to MongoDB: {e}")


def save_to_file(results: List[Dict[str, Any]], path: str) -> None:
    """Save NEL results to JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Saved {len(results):,} NEL results to {path}")
    except Exception as e:
        logging.error(f"❌ Error saving to file: {e}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics of NEL results."""
    if not results:
        logging.info("❌ No entities were linked.")
        return
    
    total_entities = sum(r.get("entity_count", 0) for r in results)
    total_linked = sum(r.get("linked_entity_count", 0) for r in results)
    
    # Count entity types
    entity_types = {}
    for r in results:
        for ent in r.get("linked_entities", []):
            ent_type = ent.get("type", "UNKNOWN")
            entity_types[ent_type] = entity_types.get(ent_type, 0) + 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"📊 NEL Summary:")
    logging.info(f"   Posts processed: {len(results):,}")
    logging.info(f"   Total entities found: {total_entities:,}")
    logging.info(f"   Entities linked to Wikipedia: {total_linked:,}")
    logging.info(f"   Link success rate: {(total_linked/total_entities*100):.1f}%")
    logging.info(f"\n   Entity types distribution:")
    for ent_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"      {ent_type}: {count:,}")
    logging.info(f"{'='*60}\n")


def main() -> None:
    """Main function to run NEL pipeline."""
    parser = argparse.ArgumentParser(description="Named Entity Linking for Reddit posts")
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
    logging.info("🚀 Starting Named Entity Linking pipeline...")
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
        logging.info("❌ No entities were extracted or linked.")
        return
    
    # Save results
    save_to_mongo(results)
    save_to_file(results, args.output_file)
    
    # Print summary
    print_summary(results)
    
    logging.info("✅ Named Entity Linking complete!")


if __name__ == "__main__":
    main()
