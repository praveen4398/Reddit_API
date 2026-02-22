import os
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

# Load env
load_dotenv()

# IO paths (optional JSON artifact)
OUTPUT_DIR = os.path.join("output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "keyword_and_hashtag_output.json")

# Mongo
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGODB_DB", "reddit_db")
source_collection_name = os.getenv("MONGODB_COLLECTION", "posts")
result_collection_name = os.getenv("RESULT_COLLECTION", "keyword_and_hashtag_results")

client = MongoClient(mongo_uri)
db = client[db_name]
source_col = db[source_collection_name]
result_col = db[result_collection_name]

# Same hashtag model as reddit_fetch.py: '#tag' or '#   tag' → normalized '#tag'
hashtag_pattern = re.compile(r"(?<!\w)#\s*([A-Za-z0-9_][A-Za-z0-9_-]*)")

# Mentions (generic @username-like)
mention_pattern = re.compile(r"@[A-Za-z0-9_]+")

# Minimal keyword extraction: frequent non-stopword tokens from title+body
STOPWORDS = {
    "the","and","for","that","with","this","from","have","you","your","are","was","were","will",
    "they","them","their","our","ours","has","had","but","not","all","any","can","could","should",
    "would","there","here","what","when","where","who","whom","why","how","a","an","in","on","of",
    "to","as","by","it","its","is","be","or","at","we","me","my","he","she","his","her","if",
    "about","into","over","under","no","yes","do","did","done","just","than","then","too","very",
}
word_pattern = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def extract_hashtags(text: str) -> List[str]:
    if not text:
        return []
    tags = {f"#{m.group(1).lower()}" for m in hashtag_pattern.finditer(text)}
    return list(tags)


def extract_mentions(text: str) -> List[str]:
    if not text:
        return []
    return [m.lower() for m in mention_pattern.findall(text)]


def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    if not text:
        return []
    tokens = [t.lower() for t in word_pattern.findall(text)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(max_keywords)]


def build_content(doc: Dict[str, Any]) -> str:
    title = doc.get("title") or ""
    body = doc.get("body") or ""
    return (title + ("\n\n" + body if body else "")).strip()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cursor = source_col.find({}, projection={
        "_id": 1,
        "id": 1,
        "title": 1,
        "body": 1,
        "subreddit": 1,
        "author": 1,
        "created_utc": 1,
        "permalink": 1,
    })

    results: List[Dict[str, Any]] = []
    ops: List[UpdateOne] = []
    processed = 0

    for doc in cursor:
        processed += 1
        content = build_content(doc)

        hashtags = list({*extract_hashtags(doc.get("title") or ""), *extract_hashtags(doc.get("body") or "")})
        mentions = extract_mentions(content)
        keywords = extract_keywords_simple(content, max_keywords=10)

        enriched = {
            "_id": doc.get("_id"),  # keep same key if possible
            "post_id": doc.get("id"),
            "subreddit": doc.get("subreddit"),
            "author": doc.get("author"),
            "created_utc": doc.get("created_utc"),
            "permalink": doc.get("permalink"),
            "title": doc.get("title"),
            "hashtags": hashtags if hashtags else None,
            "mentions": mentions if mentions else None,
            "keywords": keywords if keywords else None,
            "enriched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        results.append(enriched)
        ops.append(UpdateOne({"_id": enriched["_id"]}, {"$set": enriched}, upsert=True))

        if len(ops) >= 1000:
            result_col.bulk_write(ops, ordered=False)
            ops = []

    if ops:
        result_col.bulk_write(ops, ordered=False)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed: {processed} | Wrote: {len(results)} | Output: {OUTPUT_PATH} | Collection: {result_collection_name}")


if __name__ == "__main__":
    main()
