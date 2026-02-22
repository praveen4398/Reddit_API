"""
Preprocessing module for Reddit data collected by reddit_fetch.py.
Handles language detection, translation, and text cleaning for title/body content.
"""
import os
import re
import json
import emoji
import time
import argparse
import logging
from langdetect import detect
from deep_translator import GoogleTranslator
from unidecode import unidecode
from datetime import datetime, timezone
from typing import Dict, Any, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Load env
load_dotenv()

# Mongo setup
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGODB_DB", "reddit_db")
source_collection = os.getenv("MONGODB_COLLECTION", "posts")
output_collection = os.getenv("PREPROCESS_COLLECTION", "preprocessed_posts")

client = MongoClient(mongo_uri)
db = client[db_name]
col_in = db[source_collection]
col_out = db[output_collection]

# Output file paths
OUTPUT_DIR = os.path.join("output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "reddit_preprocessed.json")


# Language Detection

def detect_language(text: str) -> str:
	try:
		if not isinstance(text, str) or not text.strip():
			return "unknown"
		return detect(text)
	except Exception:
		return "unknown"


# Translation (only if non-English)

def translate_to_english(text: str, lang: str) -> str:
	if not isinstance(text, str) or not text.strip():
		return ""
	if lang and lang != "en" and lang != "unknown":
		try:
			# Add small delay to avoid rate limiting
			time.sleep(0.1)
			# The API has a 5000-character limit.
			return GoogleTranslator(source="auto", target="en").translate(text[:4999])
		except Exception as e:
			logging.error(f"Translation failed: {e}")
			return text
	return text


# Cleaning function

def clean_text(text: str) -> str:
	if not isinstance(text, str):
		text = ""
	text = emoji.replace_emoji(text, replace="")  # Remove emojis
	text = unidecode(text)  # Remove accented characters
	text = re.sub(r"http\S+", "", text)  # Remove URLs
	text = re.sub(r"@\w+", "", text)  # Remove mentions-like strings
	text = re.sub(r"#\w+", "", text)  # Remove hashtags
	text = re.sub(r"[^A-Za-z0-9\s]+", "", text)  # Remove special characters
	text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
	return text


def build_text(doc: Dict[str, Any]) -> str:
	# Prefer combined content from reddit_fetch.py; fallback to title/body
	content = doc.get("content")
	if not isinstance(content, str):
		content = ""
	content = content.strip()
	if content:
		return content
	title = doc.get("title") or ""
	body = doc.get("body") or ""
	return (title + ("\n\n" + body if body else "")).strip()


def process_post(doc: Dict[str, Any]) -> Dict[str, Any] | None:
	try:
		text = build_text(doc)
		lang = detect_language(text)
		translated = translate_to_english(text, lang)
		cleaned = clean_text(translated)

		return {
			"_id": doc.get("_id"),
			"post_id": doc.get("id"),
			"subreddit": doc.get("subreddit"),
			"author": doc.get("author"),
			"created_utc": doc.get("created_utc"),
			"permalink": doc.get("permalink"),
			"title": doc.get("title"),
			"language_detected": lang,
			"translated_text": translated,
			"cleaned_text": cleaned,
			"preprocessed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
		}
	except Exception as e:
		logging.error(f"Error processing post {doc.get('id')}: {e}")
		return None


def process_all(limit: int | None = None) -> List[Dict[str, Any]]:
	# Check for existing processed posts to skip
	existing_ids = set()
	try:
		for doc in col_out.find({}, {"_id": 1}):
			existing_ids.add(doc["_id"])
		logging.info(f"📊 Found {len(existing_ids)} already processed posts")
	except Exception:
		logging.info("📊 No existing processed posts found")

	cursor = col_in.find({}, projection={
		"_id": 1,
		"id": 1,
		"title": 1,
		"body": 1,
		"content": 1,
		"subreddit": 1,
		"author": 1,
		"created_utc": 1,
		"permalink": 1,
	})
	results: List[Dict[str, Any]] = []
	ops: List[UpdateOne] = []
	count = 0
	processed_count = 0
	skipped_count = 0
	batch_size = int(os.getenv("PREPROCESS_BULK_SIZE", "500"))
	start_time = time.time()

	logging.info(f"🚀 Starting preprocessing with batch size: {batch_size}")
	logging.info(f"⏱️  Progress updates every {batch_size} posts")

	for doc in cursor:
		count += 1
		
		# Skip if already processed (temporarily disabled to force reprocessing)
		# if doc.get("_id") in existing_ids:
		# 	skipped_count += 1
		# 	continue

		processed = process_post(doc)
		if not processed:
			continue
		
		results.append(processed)
		ops.append(UpdateOne({"_id": processed["_id"]}, {"$set": processed}, upsert=True))
		processed_count += 1

		if len(ops) >= batch_size:
			col_out.bulk_write(ops, ordered=False)
			ops = []
			
			# Progress update
			elapsed = time.time() - start_time
			rate = processed_count / elapsed if elapsed > 0 else 0
			eta = (count - processed_count - skipped_count) / rate if rate > 0 else 0
			logging.info(f"📈 Processed: {processed_count:,} | Skipped: {skipped_count:,} | Total: {count:,} | Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min")

		if limit is not None and processed_count >= limit:
			break

	if ops:
		col_out.bulk_write(ops, ordered=False)

	logging.info(f"\n✅ Final: Processed {processed_count:,} | Skipped {skipped_count:,} | Total time: {(time.time()-start_time)/60:.1f}min")
	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Preprocess Reddit posts.")
	parser.add_argument(
		"--output-file",
		type=str,
		default=OUTPUT_PATH,
		help=f"Path to save the preprocessed JSON output. Defaults to {OUTPUT_PATH}",
	)
	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
	results = process_all(limit=None)
	with open(args.output_file, "w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)
	logging.info(f"Preprocessed: {len(results)} | Output: {args.output_file} | Collection: {output_collection}")


if __name__ == "__main__":
	main()
