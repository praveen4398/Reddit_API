"""
Reddit Topic Detection
This script detects topics in Reddit posts using rule-based keyword matching.
"""

import os
import json
import logging
from typing import Dict, Any, List
from tqdm import tqdm
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load English language model for better text processing
# Try to load the model, if not available, it will download it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Spacy model 'en_core_web_sm' not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

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

def detect_topic(text: str) -> Dict[str, Any]:
    """
    Detect the topic of a given text using rule-based keyword matching.
    
    Args:
        text: The input text to analyze
        
    Returns:
        Dict containing 'label' and 'score' for the detected topic
    """
    if not text or not text.strip():
        return {"label": "other", "score": 0.0}
    
    # Preprocess text: lowercase and lemmatize
    doc = nlp(text.lower())
    processed_text = " ".join([token.lemma_ for token in doc])
    
    # Define topics and their associated keywords with weights
    topics = {
        "technology": ["tech", "computer", "software", "app", "phone", "ai", "artificial intelligence", 
                      "coding", "program", "developer", "internet", "data", "code", "algorithm", "system"],
        "gaming": ["game", "play", "player", "xbox", "playstation", "nintendo", "steam", "level", 
                   "character", "quest", "rpg", "fps", "multiplayer", "console", "indie"],
        "movies_tv": ["movie", "film", "show", "episode", "season", "netflix", "hbo", "disney", 
                      "marvel", "star wars", "actor", "actress", "director", "cinema", "streaming"],
        "science": ["science", "research", "study", "scientist", "space", "nasa", "discovery", 
                    "experiment", "physics", "chemistry", "biology", "universe", "theory", "quantum"],
        "programming": ["python", "javascript", "java", "c++", "html", "css", "react", "node", 
                        "api", "framework", "library", "function", "variable", "loop", "debug"],
        "politics": ["politics", "government", "election", "president", "minister", "congress", 
                     "senate", "vote", "democrat", "republican", "policy", "law", "bill", "senator"],
        "sports": ["sport", "football", "basketball", "soccer", "game", "match", "player", 
                   "team", "win", "championship", "tournament", "olympic", "nba", "nfl", "fifa"],
        "music": ["music", "song", "album", "band", "artist", "listen", "spotify", "youtube", 
                  "concert", "guitar", "piano", "rap", "pop", "rock", "jazz"],
        "books": ["book", "read", "author", "novel", "story", "chapter", "page", "publish", 
                  "fiction", "fantasy", "sci-fi", "mystery", "thriller", "biography"],
        "health": ["health", "medical", "doctor", "hospital", "disease", "covid", "medicine", 
                   "patient", "treatment", "vaccine", "mental", "fitness", "exercise", "diet"]
    }
    
    # Initialize scores
    scores = {topic: 0 for topic in topics}
    
    # Score each topic based on keyword matches
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in processed_text:
                scores[topic] += 1
    
    # Get the topic with the highest score
    best_topic, best_score = max(scores.items(), key=lambda x: x[1])
    
    # If no keywords matched, return 'other'
    if best_score == 0:
        return {"label": "other", "score": 0.0}
    
    # Normalize score to 0-1 range
    max_possible_score = max(len(keywords) for keywords in topics.values())
    normalized_score = min(1.0, best_score / max_possible_score)
    
    return {
        "label": best_topic,
        "score": round(normalized_score, 3)
    }

def process_reddit_data(input_file: str, output_file: str, limit: int = None) -> None:
    """
    Process Reddit data to detect topics for each post.
    
    Args:
        input_file: Path to input JSON/JSONL file
        output_file: Path to save the processed data
        limit: Maximum number of records to process (for testing)
    """
    logging.info(f"Loading data from {input_file}...")
    data = load_data(input_file, limit)
    
    if not data:
        logging.error("No data loaded. Exiting...")
        return
    
    logging.info(f"Processing {len(data)} posts...")
    
    processed_data = []
    for post in tqdm(data, desc="Analyzing posts"):
        try:
            # Use title + selftext if available, otherwise use the available text
            text = " ".join(filter(None, [
                post.get('title', ''),
                post.get('selftext', '')
            ])).strip()
            
            # Skip if no text content
            if not text:
                continue
                
            # Detect topic
            topic_info = detect_topic(text)
            
            # Add topic information to the post
            processed_post = post.copy()
            processed_post["topic"] = topic_info
            processed_data.append(processed_post)
            
        except Exception as e:
            logging.error(f"Error processing post {post.get('id', 'unknown')}: {e}")
            continue
    
    # Save the processed data
    if processed_data:
        save_data(processed_data, output_file)
    else:
        logging.warning("No posts were processed successfully.")

def main():
    """Main function to run topic detection on Reddit data."""
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "Reddit_data.json")
    output_file = os.path.join(OUTPUT_DIR, "reddit_topics.json")
    
    # Process the data
    process_reddit_data(input_file, output_file)
    
    logging.info("Topic detection complete!")

if __name__ == "__main__":
    main()
