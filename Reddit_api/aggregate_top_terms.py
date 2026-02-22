import os
import json
from collections import Counter
from typing import List, Dict, Any

from dotenv import load_dotenv
from pymongo import MongoClient

# Load env
load_dotenv()

# Config
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGODB_DB", "reddit_db")
source_collection_name = os.getenv("RESULT_COLLECTION", "keyword_and_hashtag_results")
output_dir = os.path.join("output")
output_hashtags = os.path.join(output_dir, "top_hashtags.json")
output_keywords = os.path.join(output_dir, "top_keywords.json")
TOP_N = int(os.getenv("TOP_N", "100"))

client = MongoClient(mongo_uri)
db = client[db_name]
col = db[source_collection_name]


def flatten_lists(values: List[List[str] | None]) -> List[str]:
    flat: List[str] = []
    for v in values:
        if not v:
            continue
        for item in v:
            if not item:
                continue
            flat.append(item)
    return flat


def main() -> None:
    os.makedirs(output_dir, exist_ok=True)

    cursor = col.find({}, projection={"hashtags": 1, "keywords": 1}, no_cursor_timeout=True)

    hashtag_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()

    try:
        for doc in cursor:
            hashtags = doc.get("hashtags")
            keywords = doc.get("keywords")
            if hashtags:
                hashtag_counter.update(h for h in hashtags if isinstance(h, str) and h)
            if keywords:
                keyword_counter.update(k for k in keywords if isinstance(k, str) and k)
    finally:
        cursor.close()

    top_hashtags: List[Dict[str, Any]] = [
        {"hashtag": tag, "count": count} for tag, count in hashtag_counter.most_common(TOP_N)
    ]
    top_keywords: List[Dict[str, Any]] = [
        {"keyword": kw, "count": count} for kw, count in keyword_counter.most_common(TOP_N)
    ]

    with open(output_hashtags, "w", encoding="utf-8") as f:
        json.dump(top_hashtags, f, ensure_ascii=False, indent=2)
    with open(output_keywords, "w", encoding="utf-8") as f:
        json.dump(top_keywords, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote top {TOP_N} hashtags to {output_hashtags} and top {TOP_N} keywords to {output_keywords}"
    )


if __name__ == "__main__":
    main()
