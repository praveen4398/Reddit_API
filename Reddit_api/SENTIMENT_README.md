# Sentiment Analysis Module for Reddit Data

## Overview
The `reddit_sentiment.py` module performs sentiment analysis on Reddit posts using the pre-trained `cardiffnlp/twitter-roberta-base-sentiment` model. It classifies posts as Positive, Neutral, or Negative with confidence scores.

## Features
- **Pre-trained Model**: Uses RoBERTa model fine-tuned on Twitter data
- **Three-class Classification**: Positive, Neutral, Negative
- **Confidence Scores**: Provides probability scores for all three classes
- **Dual Storage**: Saves results to both MongoDB and JSON file
- **Text Preprocessing**: Handles emojis, newlines, and long texts
- **Batch Processing**: Efficiently processes large datasets with progress tracking

## Installation

### 1. Install Dependencies
```bash
pip install transformers torch scipy emoji
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Ensure your `.env` file has:
```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=reddit_db
MONGODB_COLLECTION=posts
SENTIMENT_COLLECTION=sentiment_results
```

## Usage

### Basic Usage (from JSON file)
```bash
python reddit_sentiment.py
```
This reads from `output/Reddit_data.json` and saves results to:
- MongoDB: `reddit_db.sentiment_results`
- JSON: `output/reddit_sentiment_results.json`

### Load from MongoDB
```bash
python reddit_sentiment.py --source mongo
```

### Process Limited Posts (for testing)
```bash
python reddit_sentiment.py --limit 100
```

### Custom Output File
```bash
python reddit_sentiment.py --output-file output/my_sentiment.json
```

### Combined Options
```bash
python reddit_sentiment.py --source mongo --limit 1000 --output-file output/test_sentiment.json
```

## Output Format

### JSON Structure
```json
[
  {
    "_id": "1myh2vu",
    "post_id": "1myh2vu",
    "subreddit": "Python",
    "author": "AutoModerator",
    "created_utc": "2025-08-24T00:00:32Z",
    "permalink": "https://www.reddit.com/r/Python/comments/1myh2vu/...",
    "title": "Sunday Daily Thread: What's everyone working on this week?",
    "text_analyzed": "Sunday Daily Thread: What's everyone working on this week? Share your...",
    "sentiment": {
      "label": "Positive",
      "score": 0.8234,
      "scores": {
        "Negative": 0.0456,
        "Neutral": 0.1310,
        "Positive": 0.8234
      }
    },
    "processed_at": "2025-10-30T07:30:00.123456Z"
  }
]
```

### Field Descriptions
- **`sentiment.label`**: Primary sentiment classification (Positive/Neutral/Negative)
- **`sentiment.score`**: Confidence score for the primary label (0-1)
- **`sentiment.scores`**: Probability distribution across all three classes
- **`text_analyzed`**: First 500 characters of the analyzed text

## Model Information

### RoBERTa Sentiment Model
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Base**: RoBERTa (Robustly Optimized BERT)
- **Training Data**: ~58M tweets
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Max Length**: 512 tokens

### Performance Characteristics
- **Accuracy**: ~70% on Twitter data
- **Speed**: ~10-50 posts/second (CPU), faster on GPU
- **Memory**: ~500MB for model weights

## Example Results

### Positive Sentiment
```json
{
  "title": "Just got my first job as a Python developer!",
  "sentiment": {
    "label": "Positive",
    "score": 0.92,
    "scores": {
      "Negative": 0.02,
      "Neutral": 0.06,
      "Positive": 0.92
    }
  }
}
```

### Negative Sentiment
```json
{
  "title": "This bug has been driving me crazy for hours",
  "sentiment": {
    "label": "Negative",
    "score": 0.78,
    "scores": {
      "Negative": 0.78,
      "Neutral": 0.18,
      "Positive": 0.04
    }
  }
}
```

### Neutral Sentiment
```json
{
  "title": "Python 3.12 release notes",
  "sentiment": {
    "label": "Neutral",
    "score": 0.85,
    "scores": {
      "Negative": 0.05,
      "Neutral": 0.85,
      "Positive": 0.10
    }
  }
}
```

## Performance Considerations

### Processing Speed
- **CPU**: ~10-20 posts/second
- **GPU**: ~50-100 posts/second (if CUDA available)
- **125,000 posts**: ~2-3 hours on CPU

### Memory Usage
- **Model**: ~500MB RAM
- **Batch processing**: Minimal additional memory
- **GPU**: 2GB+ VRAM recommended

### Optimization Tips
1. **Test with small batches**: Use `--limit 10` first
2. **Use GPU if available**: Automatically detected by PyTorch
3. **Monitor progress**: Progress bar shows estimated completion time
4. **Run in background**: Process can run unattended

## Text Preprocessing

The module automatically:
1. **Demojizes emojis**: Converts 😊 to `:smiling_face:`
2. **Removes newlines**: Replaces with spaces
3. **Truncates long texts**: Limits to 5000 characters
4. **Tokenizes**: Splits into max 512 tokens for model

## Troubleshooting

### Model Download Issues
The model (~500MB) downloads automatically on first run. If it fails:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
```

### CUDA/GPU Issues
If you have GPU but it's not being used:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Errors
If you run out of memory:
- Process in smaller batches: `--limit 1000`
- Close other applications
- Use CPU instead of GPU

### Import Errors
```bash
pip install transformers torch scipy emoji
```

## Integration with Other Modules

### Pipeline Order
1. `reddit_fetch.py` - Fetch Reddit posts
2. `preprocess_reddit.py` - Clean and translate text
3. `reddit_sentiment.py` - Analyze sentiment ← **You are here**
4. `reddit_nel.py` - Extract and link entities
5. `behaviour_flags.py` - Flag problematic content

### Using Sentiment Results
```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["reddit_db"]
sentiment_results = db["sentiment_results"]

# Find positive posts
positive_posts = sentiment_results.find({
    "sentiment.label": "Positive",
    "sentiment.score": {"$gte": 0.8}
})

# Find negative posts in specific subreddit
negative_posts = sentiment_results.find({
    "subreddit": "mentalhealth",
    "sentiment.label": "Negative"
})

# Calculate average sentiment by subreddit
pipeline = [
    {"$group": {
        "_id": "$subreddit",
        "avg_positive": {"$avg": "$sentiment.scores.Positive"},
        "avg_negative": {"$avg": "$sentiment.scores.Negative"},
        "count": {"$sum": 1}
    }},
    {"$sort": {"avg_positive": -1}}
]
results = list(sentiment_results.aggregate(pipeline))
```

## Example Output Summary

```
🤖 Loading sentiment analysis model: cardiffnlp/twitter-roberta-base-sentiment...
✅ Sentiment analysis model loaded successfully.
🚀 Starting Sentiment Analysis pipeline...
📂 Loaded 10,000 records from output/Reddit_data.json
😊 Analyzing sentiment: 100%|████████████████| 10000/10000 [08:20<00:00, 20.00it/s]
💾 Saved 9,876 sentiment results to MongoDB (sentiment_results)
💾 Saved 9,876 sentiment results to output/reddit_sentiment_results.json

============================================================
📊 Sentiment Analysis Summary:
   Total posts analyzed: 9,876

   Sentiment Distribution:
      😊 Positive: 4,234 (42.9%)
      😐 Neutral:  3,456 (35.0%)
      😞 Negative: 2,186 (22.1%)

   Average Confidence Scores:
      Positive: 0.782
      Neutral: 0.691
      Negative: 0.745
============================================================

✅ Sentiment Analysis complete!
```

## Subreddit-Specific Insights

Different subreddits tend to have different sentiment patterns:

| Subreddit Type | Typical Distribution |
|----------------|---------------------|
| Mental Health | Higher negative (40-50%) |
| Technology | Balanced (30-40-30) |
| Programming Help | Slightly negative (frustration) |
| Success Stories | Higher positive (60-70%) |
| News | Slightly negative (30-40%) |

## API Reference

### Main Functions

#### `preprocess_text(text: str) -> str`
Cleans text by removing newlines and demojizing emojis.

#### `analyze_sentiment(text: str) -> Dict[str, Any]`
Analyzes sentiment and returns label, score, and all probabilities.

#### `process_post(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]`
Processes a single Reddit post for sentiment analysis.

#### `process_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
Batch processes multiple posts with progress tracking.

## Advanced Usage

### Custom Sentiment Thresholds
```python
# Filter high-confidence results
high_confidence = [
    r for r in results 
    if r['sentiment']['score'] >= 0.8
]

# Mixed sentiment (low confidence)
mixed_sentiment = [
    r for r in results 
    if r['sentiment']['score'] < 0.6
]
```

### Sentiment Trends Over Time
```python
from datetime import datetime
import matplotlib.pyplot as plt

# Group by date and calculate average sentiment
dates = []
avg_positive = []

for result in results:
    date = datetime.fromisoformat(result['created_utc'].replace('Z', '+00:00'))
    dates.append(date)
    avg_positive.append(result['sentiment']['scores']['Positive'])

# Plot sentiment over time
plt.plot(dates, avg_positive)
plt.xlabel('Date')
plt.ylabel('Average Positive Sentiment')
plt.title('Sentiment Trend Over Time')
plt.show()
```

## License
This module uses the `cardiffnlp/twitter-roberta-base-sentiment` model, which is licensed under MIT.

## Citation
If you use this model in research, please cite:
```
@inproceedings{barbieri-etal-2020-tweeteval,
    title = "{T}weet{E}val: Unified Benchmark and Comparative Evaluation for Tweet Classification",
    author = "Barbieri, Francesco and Camacho-Collados, Jose and Espinosa Anke, Luis and Neves, Leonardo",
    booktitle = "Findings of EMNLP",
    year = "2020"
}
```

## Support
For issues or questions, check the main project documentation or the Hugging Face model page.
