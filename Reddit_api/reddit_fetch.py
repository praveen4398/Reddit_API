import os
import time
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
import praw
from pymongo import MongoClient, UpdateOne
import prawcore

# Load .env file
load_dotenv()

# Reddit API client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "data_project:v1.0 by script"),
)

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
_db_name = os.getenv("MONGODB_DB", "reddit_db")
_collection_name = os.getenv("MONGODB_COLLECTION", "posts")

client = MongoClient(mongo_uri)
collection = client[_db_name][_collection_name]

# Config
subreddits_raw = os.getenv("SUBREDDITS", "python,learnpython,programming").strip()
subreddits = [s.strip() for s in subreddits_raw.split(",") if s.strip()]
bulk_size = int(os.getenv("BULK_SIZE", "500"))

# Last X hours filter
since_hours = float(os.getenv("SINCE_HOURS", "720"))  # default 30 days
cutoff_ts = time.time() - since_hours * 3600

# Regex for hashtags
hashtag_pattern = re.compile(r"(?<!\w)#\s*([A-Za-z0-9_][A-Za-z0-9_-]*)")

def extract_hashtags(text: str) -> list[str]:
    if not text:
        return []
    tags = {f"#{m.group(1).lower()}" for m in hashtag_pattern.finditer(text)}
    return list(tags)

def get_existing_post_ids(sr_name: str):
    """Get set of existing post IDs for a subreddit from MongoDB."""
    try:
        existing_ids = set()
        cursor = collection.find(
            {"subreddit": sr_name}, 
            {"_id": 1}
        )
        for doc in cursor:
            existing_ids.add(doc["_id"])
        return existing_ids
    except Exception as e:
        print(f"Warning: Could not fetch existing IDs for {sr_name}: {e}")
        return set()

def fetch_subreddit(sr_name: str):
    """Fetch all posts from last X hours from one subreddit, skipping existing ones."""
    sr = reddit.subreddit(sr_name)
    scanned, kept, inserted, skipped = 0, 0, 0, 0
    ops = []
    
    # Get existing post IDs to avoid re-processing
    print(f"  📋 Checking existing posts for r/{sr_name}...")
    existing_ids = get_existing_post_ids(sr_name)
    print(f"  📊 Found {len(existing_ids)} existing posts in database")

    try:
        for post in sr.new(limit=None):   # fetch as many as Reddit allows (~1000)
            scanned += 1
            if post.created_utc < cutoff_ts:
                break  # stop once older than cutoff
            
            # Skip if post already exists in database
            if post.id in existing_ids:
                skipped += 1
                continue

            author = getattr(post.author, "name", None)
            title = post.title or ""
            body = post.selftext if post.is_self else ""
            content = (title + ("\n\n" + body if body else "")).strip()
            hashtags = list({*extract_hashtags(title), *extract_hashtags(body)})

            kept += 1
            doc = {
                "_id": post.id,
                "id": post.id,
                "author": author,
                "title": title,
                "content": content,
                "hashtags": hashtags if hashtags else None,  # null if none
                "flair": post.link_flair_text,
                "score": post.score,
                "url": post.url,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(
                    post.created_utc, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z"),
                "body": body if post.is_self else None,
                "subreddit": str(post.subreddit),
                "permalink": f"https://www.reddit.com{post.permalink}",
                "is_self": post.is_self,
                "over_18": post.over_18,
            }
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

            if len(ops) >= bulk_size:
                result = collection.bulk_write(ops, ordered=False)
                inserted += (result.upserted_count or 0) + (result.modified_count or 0)
                ops = []
    except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden, prawcore.exceptions.Redirect, prawcore.exceptions.BadRequest) as e:
        print(f"Skipping subreddit '{sr_name}' due to API error: {type(e).__name__}")
    except prawcore.exceptions.PrawcoreException as e:
        print(f"Skipping subreddit '{sr_name}' due to request error: {type(e).__name__}")

    # flush remaining ops
    if ops:
        result = collection.bulk_write(ops, ordered=False)
        inserted += (result.upserted_count or 0) + (result.modified_count or 0)

    return scanned, kept, inserted, skipped

# ---- MAIN ----
total_scanned, total_kept, total_inserted, total_skipped = 0, 0, 0, 0
for sr in subreddits:
    print(f"\n🔎 Fetching subreddit: {sr}")
    s, k, i, sk = fetch_subreddit(sr)
    total_scanned += s
    total_kept += k
    total_inserted += i
    total_skipped += sk
    print(f"  ✅ r/{sr}: Scanned: {s} | New: {k} | Skipped: {sk} | Inserted: {i}")
    time.sleep(0.5)  # polite pause between subreddits

print(f"\n🎯 SUMMARY:")
print(f"   Subreddits processed: {len(subreddits)}")
print(f"   Total posts scanned: {total_scanned}")
print(f"   New posts found: {total_kept}")
print(f"   Existing posts skipped: {total_skipped}")
print(f"   Posts inserted/updated: {total_inserted}")
print(f"   Time efficiency: {total_skipped}/{total_scanned} posts skipped ({100*total_skipped/max(total_scanned,1):.1f}%)")
