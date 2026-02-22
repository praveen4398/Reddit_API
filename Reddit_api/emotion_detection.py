"""
Reddit Emotion Detection

Detects emotions in Reddit posts and comments using j-hartmann/emotion-english-distilroberta-base.
Supports incremental processing of posts.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from collections import Counter
from tqdm import tqdm
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize emotion classifier
EMOTION_CLASSIFIER = None

def load_emotion_classifier():
    """Load the emotion classification model."""
    global EMOTION_CLASSIFIER
    if EMOTION_CLASSIFIER is None:
        logging.info("Loading emotion classification model: j-hartmann/emotion-english-distilroberta-base...")
        try:
            EMOTION_CLASSIFIER = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=-1  # Use CPU (-1) or GPU (0)
            )
            logging.info("Emotion detection model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load emotion detection model: {e}")
            raise
    return EMOTION_CLASSIFIER

def load_data(file_path: str, max_records: int = None) -> List[Dict]:
    """Load JSON or JSONL data from file with error handling."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to load as JSON array
            try:
                content = json.load(f)
                if isinstance(content, list):
                    return content[:max_records] if max_records else content
            except json.JSONDecodeError:
                pass
            
            # If not a JSON array, try reading as JSONL
            f.seek(0)
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"Could not parse line {i+1}: {e}")
                    continue
        return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return []

def save_data(data: List[Dict], file_path: str) -> None:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(data)} records to {file_path}")
    except Exception as e:
        logging.error(f"Error saving to {file_path}: {e}")

def detect_emotions(text: str) -> Optional[Dict[str, Any]]:
    """
    Detect emotions in the given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing emotion analysis results or None if processing fails
    """
    if not text or len(text.split()) < 3:
        return None
        
    try:
        classifier = load_emotion_classifier()
        scores = classifier(text)[0]
        top_emotion = max(scores, key=lambda x: x["score"])
        
        return {
            "label": top_emotion["label"],
            "score": round(float(top_emotion["score"]), 4),
            "scores": {item["label"]: round(float(item["score"]), 4) 
                      for item in scores}
        }
    except Exception as e:
        logging.error(f"Emotion detection failed: {e}")
        return None

def process_post(post: Dict) -> Optional[Dict]:
    """
    Process a single Reddit post for emotion detection.
    
    Args:
        post: Post data with 'title' and/or 'selftext'
        
    Returns:
        Processed post data with emotion analysis results or None if processing fails
    """
    try:
        # Combine title and selftext for analysis
        text = " ".join(filter(None, [
            post.get('title', ''),
            post.get('selftext', '')
        ])).strip()
        
        if not text:
            return None
            
        # Get emotion analysis
        emotion_result = detect_emotions(text)
        if not emotion_result:
            return None
            
        # Return the post with emotion analysis
        processed_post = post.copy()
        processed_post["emotion"] = emotion_result
        return processed_post
        
    except Exception as e:
        logging.error(f"Error processing post {post.get('id', 'unknown')}: {e}")
        return None

def process_posts(posts: List[Dict]) -> List[Dict]:
    """
    Process a batch of Reddit posts with emotion detection.
    
    Args:
        posts: List of post dictionaries
        
    Returns:
        List of processed posts with emotion analysis
    """
    processed = []
    emotion_counts = Counter()
    
    for post in tqdm(posts, desc="Analyzing emotions"):
        result = process_post(post)
        if result:
            emotion_label = result["emotion"]["label"]
            emotion_score = result["emotion"]["score"]
            emotion_counts[emotion_label] += 1
            
            processed.append(result)
    
    # Log emotion distribution
    if emotion_counts:
        logging.info("\n=== Emotion Distribution ===")
        for emotion, count in emotion_counts.most_common():
            logging.info(f"- {emotion}: {count}")
    
    return processed

def main():
    """Main function to run emotion detection on Reddit data."""
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "Reddit_data.json")
    output_file = os.path.join(OUTPUT_DIR, "reddit_emotions.json")
    
    # Load data
    logging.info(f"Loading data from {input_file}...")
    posts = load_data(input_file)
    
    if not posts:
        logging.error("No data loaded. Exiting...")
        return
    
    # Process posts
    logging.info(f"Processing {len(posts)} posts for emotion detection...")
    processed_posts = process_posts(posts)
    
    # Save results
    if processed_posts:
        save_data(processed_posts, output_file)
        logging.info(f"Emotion detection complete! Results saved to {output_file}")
    else:
        logging.warning("No posts were processed successfully.")

if __name__ == "__main__":
    main()
