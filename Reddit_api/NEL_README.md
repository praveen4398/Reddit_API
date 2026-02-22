# Named Entity Linking (NEL) Module for Reddit Data

## Overview
The `reddit_nel.py` module performs Named Entity Recognition (NER) and links identified entities to Wikipedia pages. It processes Reddit posts, extracts named entities (people, organizations, locations, etc.), and provides Wikipedia links for context and verification.

## Features
- **Named Entity Recognition**: Uses spaCy to extract entities from Reddit posts
- **Wikipedia Linking**: Links entities to Wikipedia pages using the Wikipedia API
- **Dual Storage**: Saves results to both MongoDB and JSON file
- **Entity Types**: Identifies PERSON, ORG, GPE, DATE, MONEY, and more
- **Rate Limiting**: Respects Wikipedia API rate limits
- **Batch Processing**: Efficiently processes large datasets

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 3. Configure Environment
Ensure your `.env` file has the following settings:
```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=reddit_db
MONGODB_COLLECTION=posts
NEL_COLLECTION=nel_results
```

## Usage

### Basic Usage (from JSON file)
```bash
python reddit_nel.py
```
This reads from `output/Reddit_data.json` and saves results to:
- MongoDB: `reddit_db.nel_results`
- JSON: `output/reddit_nel_results.json`

### Load from MongoDB
```bash
python reddit_nel.py --source mongo
```

### Process Limited Posts (for testing)
```bash
python reddit_nel.py --limit 100
```

### Custom Output File
```bash
python reddit_nel.py --output-file output/my_nel_results.json
```

### Combined Options
```bash
python reddit_nel.py --source mongo --limit 500 --output-file output/test_nel.json
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
    "text_analyzed": "Sunday Daily Thread: What's everyone working on...",
    "entity_count": 5,
    "linked_entity_count": 3,
    "linked_entities": [
      {
        "entity": "Python",
        "type": "ORG",
        "start": 45,
        "end": 51,
        "wikipedia": {
          "title": "Python (programming language)",
          "url": "https://en.wikipedia.org/?curid=23862",
          "snippet": "Python is a high-level, general-purpose programming language..."
        }
      },
      {
        "entity": "GitHub",
        "type": "ORG",
        "start": 120,
        "end": 126,
        "wikipedia": {
          "title": "GitHub",
          "url": "https://en.wikipedia.org/?curid=2577781",
          "snippet": "GitHub is a developer platform that allows developers..."
        }
      }
    ],
    "processed_at": "2025-10-30T07:25:00.123456Z"
  }
]
```

### MongoDB Collection
Results are stored in the `nel_results` collection with the same structure.

## Entity Types Recognized

The module recognizes the following entity types using spaCy:

| Type | Description | Example |
|------|-------------|---------|
| PERSON | People, including fictional | "Guido van Rossum" |
| ORG | Organizations, companies | "Google", "Python Software Foundation" |
| GPE | Geopolitical entities | "United States", "New York" |
| LOC | Non-GPE locations | "Mount Everest", "Pacific Ocean" |
| DATE | Dates or periods | "2024", "last week" |
| TIME | Times smaller than a day | "3 PM", "morning" |
| MONEY | Monetary values | "$100", "50 euros" |
| PERCENT | Percentages | "50%", "half" |
| PRODUCT | Products, vehicles | "iPhone", "Tesla Model 3" |
| EVENT | Named events | "World War II", "Olympics" |
| WORK_OF_ART | Titles of books, songs | "Harry Potter", "Bohemian Rhapsody" |
| LAW | Named laws | "GDPR", "First Amendment" |
| LANGUAGE | Named languages | "English", "Python" |

## Performance Considerations

### Rate Limiting
- Wikipedia API calls are rate-limited to 0.5 seconds per request
- Processing 1000 posts with 5 entities each takes ~40 minutes
- Consider using `--limit` for testing

### Memory Usage
- spaCy model requires ~50MB RAM
- Large datasets are processed sequentially to manage memory

### Optimization Tips
1. **Test with small batches first**: Use `--limit 10` to verify setup
2. **Run during off-peak hours**: Wikipedia API is more responsive
3. **Monitor progress**: The module shows progress bars and logs
4. **Check MongoDB**: Results are saved incrementally

## Troubleshooting

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### MongoDB Connection Error
- Ensure MongoDB is running: `mongod`
- Check connection string in `.env`

### Wikipedia API Timeout
- Increase timeout in code if needed
- Check internet connection
- Wikipedia API may be temporarily unavailable

### No Entities Found
- Check if input data has text content
- Verify posts have `title`, `body`, or `content` fields
- Some posts may not contain recognizable entities

## Integration with Other Modules

### Pipeline Order
1. `reddit_fetch.py` - Fetch Reddit posts
2. `preprocess_reddit.py` - Clean and translate text
3. `reddit_nel.py` - Extract and link entities ← **You are here**
4. `behaviour_flags.py` - Flag problematic content
5. `topic_detection.py` - Detect topics

### Using NEL Results
```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["reddit_db"]
nel_results = db["nel_results"]

# Find posts with specific entity types
posts_with_orgs = nel_results.find({
    "linked_entities.type": "ORG"
})

# Find posts mentioning specific entities
posts_about_python = nel_results.find({
    "linked_entities.entity": {"$regex": "Python", "$options": "i"}
})
```

## Example Output Summary

```
🚀 Starting Named Entity Linking pipeline...
📂 Loaded 10,000 records from output/Reddit_data.json
🔗 Extracting & linking entities: 100%|████████████| 10000/10000 [42:15<00:00, 3.94it/s]
💾 Saved 7,523 NEL results to MongoDB (nel_results)
💾 Saved 7,523 NEL results to output/reddit_nel_results.json

============================================================
📊 NEL Summary:
   Posts processed: 7,523
   Total entities found: 23,456
   Entities linked to Wikipedia: 18,234
   Link success rate: 77.7%

   Entity types distribution:
      ORG: 8,234
      PERSON: 5,123
      GPE: 3,456
      DATE: 1,234
      PRODUCT: 187
============================================================

✅ Named Entity Linking complete!
```

## API Reference

### Main Functions

#### `extract_entities(text: str) -> List[Dict[str, Any]]`
Extracts named entities from text using spaCy.

#### `link_entity_wikipedia(entity: str, entity_type: str) -> Optional[Dict[str, str]]`
Links an entity to Wikipedia and returns title, URL, and snippet.

#### `process_post(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]`
Processes a single Reddit post for NER and entity linking.

#### `process_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
Batch processes multiple posts.

## License
This module is part of the Reddit API project.

## Support
For issues or questions, please check the main project documentation.
