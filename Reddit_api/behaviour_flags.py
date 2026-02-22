import os
import re
import json
import argparse
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm


"""
Lightweight rule-based behavior flagging on Reddit posts.
- Reads input from MongoDB (preferred) or output/reddit_preprocessed.json
- Saves flagged results to MongoDB and output/reddit_behaviour_flags.json
"""

# Logging setup
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[
		logging.StreamHandler()
	]
)

# Load env
load_dotenv()

# Mongo setup (localhost by default)
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGODB_DB", "reddit_db")
preprocess_collection = os.getenv("PREPROCESS_COLLECTION", "preprocessed_posts")
flags_collection = os.getenv("BEHAVIOUR_FLAGS_COLLECTION", "behavior_flags")

client = MongoClient(mongo_uri)
db = client[db_name]
col_in = db[preprocess_collection]
col_out = db[flags_collection]

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INPUT_FILE_FALLBACK = os.path.join(OUTPUT_DIR, "reddit_preprocessed.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_behaviour_flags.json")


# === CONFIGURATION ===
FLAG_PATTERNS: Dict[str, List[str]] = {
	"toxicity": [
		r"\b(?:idiot|stupid|dumbass|retard|fuck|shit|bitch|asshole|dick|pussy|cunt|whore|slut|fag|nigg|chink|spic|kike|retard|moron)\b",
		r"\b(?:kill|die|hurt|harm|attack|beat|punch|hit|fight|war|murder|suicide)\s+(?:yourself|urself|u|myself|me|them|him|her|everyone|everybody)\b",
	],
	"spam": [
		r"\b(?:free|win|won|prize|trial|congrat|selected|winner|claim|reward|offer|discount|sale|limited|time|click|link|http|www\.|bit\.ly|goo\.gl|tinyurl)\b",
		r"\b(?:follow me|followers|retweet|rt|like|subscribe|share|comment|tag|dm|direct message|whatsapp|telegram|signal)\b",
	],
	"threats": [
		r"\b(?:kill|harm|hurt|attack|beat|punch|hit|fight|war|murder|suicide|bomb|shoot|stab|rape|torture|destroy|ruin|wreck|burn|bomb)\b",
		r"\b(?:i\s*(?:will|am going to|gonna|imma|im a|wanna|want to))\s+(?:kill|harm|hurt|attack|beat|punch|hit|fight|murder|bomb|shoot|stab|rape|torture|destroy|ruin|wreck|burn|bomb)\b",
	],
	"harassment": [
		r"\b(?:ugly|fat|stupid|idiot|retard|moron|loser|weirdo|freak|creep|pervert|stalker|harass|harassment|bully|bullying|hate|hater|hateful|racist|sexist|homophobic|transphobic|xenophobic|bigot|bigotry|discriminate|discrimination)\b",
	],
	"misinformation": [
		r"\b(?:fake news|hoax|conspiracy|cover[ -]up|false flag|deep state|government\s+lies|media\s+lies|mainstream\s+media\s+lies|fact[ -]check|fact[ -]checked|fact[ -]checking|fact[ -]checker|fact[ -]checkers|fact[ -]checking|fact[ -]checks|fact[ -]checked|fact[ -]checking|fact[ -]checker|fact[ -]checkers|fact[ -]checking|fact[ -]checks)\b",
	],
	"bot_like": [
		r"\b(?:follow me|followers|retweet|rt|like|subscribe|share|comment|tag|dm|direct message|whatsapp|telegram|signal|join now|click here|link in bio|check out|promote|promotion|sponsored|advertisement|ad|ads|advert|adverts|advertising|marketing|seo|traffic|leads|sales|conversion|conversions|converting|converted|converts|convert|converter|converters)\b",
	],
}

THRESHOLDS: Dict[str, float] = {
	"toxicity": 0.2,  # Lower threshold to catch more
	"spam": 0.2,      # Lower threshold to catch more
	"threats": 0.3,
	"harassment": 0.2,
	"misinformation": 0.3,
	"bot_like": 0.3,  # Lower threshold to catch more
}


def flag_text(text: str) -> Dict[str, float]:
	"""
	Analyze text for various behavior flags using rule-based matching.
	Returns a dictionary of flag types and their confidence scores.
	"""
	if not text or not isinstance(text, str):
		return {}

	text_lc = text.lower()
	flags: Dict[str, float] = {}

	for flag_type, patterns in FLAG_PATTERNS.items():
		score = 0.0
		max_matches = 0

		for pattern in patterns:
			matches = len(re.findall(pattern, text_lc, re.IGNORECASE))
			max_matches = max(max_matches, matches)

		if max_matches > 0:
			score = min(0.9, 0.3 + (max_matches * 0.2))
			word_count = len(text_lc.split())
			if word_count < 10:
				score = min(1.0, score + 0.2)
			if score >= THRESHOLDS.get(flag_type, 0.5):
				flags[flag_type] = round(score, 2)

	return flags


def load_preprocessed() -> List[Dict[str, Any]]:
	"""Load preprocessed posts strictly from output/reddit_preprocessed.json."""
	if not os.path.exists(INPUT_FILE_FALLBACK):
		logging.warning(f"Input file not found: {INPUT_FILE_FALLBACK}")
		return []
	with open(INPUT_FILE_FALLBACK, "r", encoding="utf-8") as f:
		data = json.load(f)
		logging.info(f"Loaded {len(data):,} records from {INPUT_FILE_FALLBACK}")
		return data


def process_flags(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	results: List[Dict[str, Any]] = []
	for doc in tqdm(records, desc="Analyzing posts for behaviour flags"):
		text = doc.get("cleaned_text") or doc.get("translated_text") or ""
		if not isinstance(text, str) or not text.strip():
			# Still add record with no flags
			result = {
				"_id": doc.get("_id"),
				"post_id": doc.get("post_id") or doc.get("id"),
				"subreddit": doc.get("subreddit"),
				"author": doc.get("author"),
				"created_utc": doc.get("created_utc"),
				"permalink": doc.get("permalink"),
				"title": doc.get("title"),
				"flags": {
					"spam": False,
					"offensive": False,
					"promotional": False,
					"question": False,
					"discussion": False
				},
			}
			results.append(result)
			continue
			
		detected_flags = flag_text(text)
		
		# Convert detected flags to boolean flags
		flags_dict = {
			"spam": "spam" in detected_flags or "bot_like" in detected_flags,
			"offensive": "toxicity" in detected_flags or "harassment" in detected_flags or "threats" in detected_flags,
			"promotional": "spam" in detected_flags or "bot_like" in detected_flags,
			"question": "?" in text,
			"discussion": len(text.split()) > 50
		}
		
		result = {
			"_id": doc.get("_id"),
			"post_id": doc.get("post_id") or doc.get("id"),
			"subreddit": doc.get("subreddit"),
			"author": doc.get("author"),
			"created_utc": doc.get("created_utc"),
			"permalink": doc.get("permalink"),
			"title": doc.get("title"),
			"flags": flags_dict,
			"detected_flags": detected_flags  # Keep original detection scores
		}
		results.append(result)
	return results


def save_to_mongo(results: List[Dict[str, Any]]) -> None:
	if not results:
		return
	ops: List[UpdateOne] = []
	for r in results:
		key = {"_id": r.get("_id")} if r.get("_id") is not None else {"post_id": r.get("post_id")}
		ops.append(UpdateOne(key, {"$set": r}, upsert=True))
	if ops:
		col_out.bulk_write(ops, ordered=False)
		logging.info(f"Saved {len(results):,} behaviour flag docs to Mongo ({flags_collection})")


def save_to_file(results: List[Dict[str, Any]], path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)
	logging.info(f"Saved {len(results):,} behaviour flag docs to {path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Behaviour flags on Reddit posts")
	parser.add_argument(
		"--output-file",
		type=str,
		default=OUTPUT_FILE,
		help=f"Path to save JSON results. Defaults to {OUTPUT_FILE}",
	)
	args = parser.parse_args()

	records = load_preprocessed()
	if not records:
		logging.info("✅ No input records found for behaviour flags.")
		return

	results = process_flags(records)
	if not results:
		logging.info("❌ No behaviour flags detected.")
		return

	save_to_mongo(results)
	save_to_file(results, args.output_file)
	logging.info("✅ Behaviour flags processing complete.")


if __name__ == "__main__":
	main()
