# 📊 Reddit NLP Analysis Dashboard

An interactive web-based dashboard for visualizing and searching through Reddit NLP analysis results.

## Features

### 1. 📈 Overview Section
- **Statistics Cards**: Total posts, spam count, offensive content count
- **Interactive Charts**:
  - Sentiment Distribution (Doughnut Chart)
  - Emotion Distribution (Bar Chart)
  - Top Topics (Horizontal Bar Chart)
  - Content Flags (Pie Chart)
- Real-time data visualization using Chart.js

### 2. 🔍 Search & Analyze Section
- **Smart Search**: Search across all NLP results
- **Search Types**:
  - All (searches everywhere)
  - Keywords
  - Hashtags
  - Sentiment
  - Emotion
  - Topic
  - Spam Posts
  - Offensive Content
- **Quick Filters**: One-click filters for common searches
  - 😊 Positive Posts
  - 😞 Negative Posts
  - 😄 Joyful Posts
  - 😠 Angry Posts
  - 🚫 Spam Posts
- **Rich Results Display**:
  - Post title, subreddit, author
  - Score and comment count
  - Sentiment analysis with confidence
  - Emotion detection with confidence
  - Topic classification
  - Keywords and hashtags
  - Named entities with Wikipedia links
  - Behavior flags
  - Generated summaries

### 3. 🕐 Recent Searches Section
- View your search history
- Click any history item to replay the search
- Shows query, type, result count, and timestamp

## Installation

1. **Install Flask** (if not already installed):
```bash
pip install flask==3.0.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

2. **Ensure Data Files Exist**:
Before running the dashboard, make sure you have generated:
- `output/post_summaries.json` (run `python generate_post_summaries.py`)
- `output/reddit_overview.json` (run `python generate_overview.py`)

## Usage

### Start the Dashboard

```bash
python dashboard_app.py
```

The dashboard will be available at: **http://localhost:5000**

### Access from Other Devices

The dashboard runs on `0.0.0.0`, so you can access it from other devices on your network:
```
http://YOUR_IP_ADDRESS:5000
```

## Dashboard Sections

### Overview Tab
- Displays comprehensive statistics about your Reddit data
- Shows visual charts for sentiment, emotions, topics, and flags
- Auto-refreshes when you switch to this tab

### Search & Analyze Tab
- Enter any search query in the search box
- Select search type from dropdown
- Click "Search" or press Enter
- Use quick filter chips for common searches
- Results show complete NLP analysis for each matching post

### Recent Searches Tab
- View all your previous searches
- Click any search to replay it
- Shows search query, type, result count, and timestamp

## Search Examples

1. **Search for a keyword**: 
   - Query: "AI" or "machine learning"
   - Type: Keywords or All

2. **Find posts with specific sentiment**:
   - Query: "positive" or "negative"
   - Type: Sentiment

3. **Find spam posts**:
   - Query: (leave empty)
   - Type: Spam Posts

4. **Search by emotion**:
   - Query: "joy" or "anger"
   - Type: Emotion

5. **Search by topic**:
   - Query: "technology" or "politics"
   - Type: Topic

6. **Search hashtags**:
   - Query: "#AI" or "AI"
   - Type: Hashtags

## Features Highlights

✅ **No JSON Format**: All data is displayed in a user-friendly format
✅ **Interactive Charts**: Visual representation of data
✅ **Real-time Search**: Instant search results
✅ **Search History**: Track your analysis journey
✅ **Responsive Design**: Works on desktop and mobile
✅ **Modern UI**: Beautiful gradient design with smooth animations
✅ **Rich Data Display**: Complete NLP analysis for each post

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Chart.js
- **Data Source**: JSON files + MongoDB (optional)
- **Styling**: Custom CSS with gradients and animations

## API Endpoints

The dashboard exposes the following API endpoints:

- `GET /` - Main dashboard page
- `GET /api/overview` - Overview statistics
- `GET /api/stats` - Quick statistics for charts
- `POST /api/search` - Search posts
- `GET /api/history` - Search history

## Troubleshooting

### Dashboard shows "No data available"
- Run `python generate_post_summaries.py` first
- Run `python generate_overview.py` first
- Check that files exist in `output/` directory

### Charts not displaying
- Ensure internet connection (Chart.js loads from CDN)
- Check browser console for errors

### Search returns no results
- Verify data files are not empty
- Try different search queries
- Check search type matches your query

## Customization

### Change Port
Edit `dashboard_app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

### Modify Colors
Edit `templates/dashboard.html` CSS section to change gradient colors and theme.

### Add More Charts
Add new chart containers in HTML and create charts in JavaScript using Chart.js.

## Notes

- Search history is stored in-memory and will reset when server restarts
- For production use, consider adding authentication
- Large datasets may take longer to search
- Results are limited to 100 posts per search for performance

## Support

For issues or questions, check:
1. Console logs in browser (F12)
2. Terminal output where Flask is running
3. Ensure all dependencies are installed
4. Verify data files exist and are valid JSON

Enjoy analyzing your Reddit data! 🎉
