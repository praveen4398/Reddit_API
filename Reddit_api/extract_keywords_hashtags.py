"""
Keyword and Hashtag Extraction for Reddit Data
This script extracts keywords and hashtags from Reddit posts using YAKE.
"""

import json
import re
import os
import yake
from datetime import datetime

# === Configuration ===
INPUT_FILE = os.path.join("output", "Reddit_data.json")
OUTPUT_FILE = os.path.join("output", "reddit_keywords_hashtags.json")

# Initialize YAKE keyword extractor
KEYWORD_EXTRACTOR = yake.KeywordExtractor(
    lan="en",           # Language
    n=3,                # Max n-gram size
    dedupLim=0.9,       # Similarity threshold for deduplication
    top=20,             # Number of top keywords to extract
    features=None       # Use default features
)

def load_reddit_data(file_path):
    """Load Reddit data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def extract_hashtags(text):
    """
    Extract hashtags from text using regex.
    Example: #AI, #Python
    """
    if not isinstance(text, str):
        return []
    return [tag.lower() for tag in re.findall(r"#(\w+)", text)]

def extract_mentions(text):
    """
    Extract mentions from text using regex.
    Example: @OpenAI, @elonmusk
    """
    if not isinstance(text, str):
        return []
    return [mention.lower() for mention in re.findall(r"@([a-zA-Z0-9_]+)", text)]

def extract_keywords(text, max_keywords=15):
    """
    Extract keywords from text using YAKE.
    Returns a list of top keywords.
    """
    try:
        if not text or not isinstance(text, str):
            return []
        keywords_with_scores = KEYWORD_EXTRACTOR.extract_keywords(text)
        return [kw for kw, score in keywords_with_scores[:max_keywords]]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def process_posts(posts):
    """Process posts to extract keywords, hashtags, and mentions."""
    processed = []
    for post in posts:
        try:
            # Get text content (combine title and body if available)
            text = post.get('title', '')
            if 'body' in post and post['body']:
                text += ' ' + post['body']
            
            # Extract features
            hashtags = extract_hashtags(text)
            mentions = extract_mentions(text)
            keywords = extract_keywords(text)
            
            # Create enriched post
            enriched = {
                'post_id': post.get('post_id', post.get('id')),
                'subreddit': post.get('subreddit'),
                'author': post.get('author'),
                'created_utc': post.get('created_utc'),
                'hashtags': hashtags,
                'mentions': mentions,
                'keywords': keywords,
                'processed_at': datetime.utcnow().isoformat()
            }
            
            processed.append(enriched)
            print(f"Processed post {enriched['post_id']} | "
                  f"Hashtags: {len(hashtags)}, "
                  f"Mentions: {len(mentions)}, "
                  f"Keywords: {len(keywords)}")
                  
        except Exception as e:
            print(f"Error processing post {post.get('id')}: {e}")
    
    return processed

def save_results(data, output_path):
    """Save processed data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to {output_path}")
        print(f"Total posts processed: {len(data)}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main function to run the extraction process."""
    print("🚀 Starting keyword and hashtag extraction...")
    
    # Load data
    print(f"\n📂 Loading data from {INPUT_FILE}")
    posts = load_reddit_data(INPUT_FILE)
    if not posts:
        print("❌ No data found. Exiting...")
        return
    
    print(f"📊 Found {len(posts)} posts to process")
    
    # Process posts
    print("\n🔍 Extracting keywords and hashtags...")
    processed_posts = process_posts(posts)
    
    # Save results
    save_results(processed_posts, OUTPUT_FILE)
    
    # Print summary
    total_hashtags = sum(len(post['hashtags']) for post in processed_posts)
    total_mentions = sum(len(post['mentions']) for post in processed_posts)
    total_keywords = sum(len(post['keywords']) for post in processed_posts)
    
    print("\n📊 Extraction Summary:")
    print(f"- Total posts processed: {len(processed_posts)}")
    print(f"- Total hashtags found: {total_hashtags}")
    print(f"- Total mentions found: {total_mentions}")
    print(f"- Total keywords extracted: {total_keywords}")
    print("\n🎉 Extraction complete!")

if __name__ == "__main__":
    main()
