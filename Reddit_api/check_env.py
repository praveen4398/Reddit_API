"""
Reddit Data EDA - Comprehensive Analysis
This script performs exploratory data analysis on Reddit data with multiple visualizations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_reddit_data(file_path, max_records=5000):
    """Load and parse Reddit data with better error handling"""
    print(f"Loading data from {file_path}...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                # Try to load as JSON array first
                data = json.load(f)
                if isinstance(data, list):
                    print(f"Loaded {len(data)} records as JSON array")
                    return data[:max_records]
            except json.JSONDecodeError:
                # If not a JSON array, try reading line by line
                f.seek(0)
                data = []
                for i, line in enumerate(f):
                    if i >= max_records:
                        break
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                print(f"Loaded {len(data)} records from JSONL file")
                return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def analyze_reddit_data():
    """Main function to perform EDA on Reddit data"""
    # Load data
    raw_data = load_reddit_data(os.path.join(OUTPUT_DIR, "Reddit_data.json"), 5000)
    
    if not raw_data:
        print("No data loaded. Exiting...")
        return

    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    print(f"\nInitial data shape: {df.shape}")
    print("Available columns:", list(df.columns))
    
    # Basic info
    print("\n=== DATA OVERVIEW ===")
    
    # Check for required columns
    date_col = 'created_utc' if 'created_utc' in df.columns else \
              'created' if 'created' in df.columns else None
    
    if date_col:
        print(f"Time period: {df[date_col].min()} to {df[date_col].max()}")
    
    if 'subreddit' in df.columns:
        print(f"Unique subreddits: {df['subreddit'].nunique()}")
    
    if 'author' in df.columns:
        print(f"Unique authors: {df['author'].nunique()}")
    
    # Convert date if available
    if date_col:
        try:
            df['created_at'] = pd.to_datetime(df[date_col])
            df['hour'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.day_name()
        except Exception as e:
            print(f"Warning: Could not parse dates: {e}")
    
    # Create text length if title or body exists
    if 'title' in df.columns:
        df['text_length'] = df['title'].str.len()
        if 'body' in df.columns:
            df['text_length'] += df['body'].str.len().fillna(0)
    
    # 1. Posts by Subreddit (if available)
    if 'subreddit' in df.columns:
        plt.figure(figsize=(12, 6))
        top_subreddits = df['subreddit'].value_counts().head(10)
        sns.barplot(y=top_subreddits.index, x=top_subreddits.values, palette='viridis')
        plt.title('Top 10 Subreddits by Post Count', fontsize=14)
        plt.xlabel('Number of Posts')
        plt.ylabel('Subreddit')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'top_subreddits.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created top subreddits plot")
    
    # 2. Posts by Hour of Day (if date is available)
    if 'hour' in df.columns:
        plt.figure(figsize=(12, 6))
        hourly_posts = df['hour'].value_counts().sort_index()
        sns.lineplot(x=hourly_posts.index, y=hourly_posts.values, marker='o')
        plt.title('Posts by Hour of Day', fontsize=14)
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Posts')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'posts_by_hour.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created posts by hour plot")
    
    # 3. Score Distribution (if score exists)
    if 'score' in df.columns:
        plt.figure(figsize=(12, 6))
        scores = df[df['score'] < df['score'].quantile(0.95)]['score']
        sns.histplot(scores, bins=50, kde=True)
        plt.title('Distribution of Post Scores', fontsize=14)
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.axvline(scores.mean(), color='r', linestyle='--', 
                   label=f'Mean: {scores.mean():.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created score distribution plot")
    
    # 4. Text Length Analysis (if text_length exists)
    if 'text_length' in df.columns:
        plt.figure(figsize=(12, 6))
        text_lengths = df['text_length'][df['text_length'] < df['text_length'].quantile(0.95)]
        sns.histplot(text_lengths, bins=50, kde=True)
        plt.title('Distribution of Post Lengths (Characters)', fontsize=14)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.axvline(text_lengths.mean(), color='r', linestyle='--', 
                   label=f'Mean: {text_lengths.mean():.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'text_length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created text length distribution plot")
    
    # 5. Top Authors (if author exists)
    if 'author' in df.columns:
        plt.figure(figsize=(12, 6))
        top_authors = df['author'].value_counts().head(15)
        sns.barplot(y=top_authors.index, x=top_authors.values, palette='magma')
        plt.title('Top 15 Most Active Authors', fontsize=14)
        plt.xlabel('Number of Posts')
        plt.ylabel('Author')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'top_authors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created top authors plot")
    
    # 6. Word Cloud of Titles (if title exists)
    if 'title' in df.columns:
        try:
            plt.figure(figsize=(12, 8))
            text = ' '.join(df['title'].dropna().astype(str))
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=100).generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Words in Post Titles', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'title_wordcloud.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created title word cloud")
        except Exception as e:
            print(f"Could not create word cloud: {e}")

if __name__ == "__main__":
    print("🚀 Starting Reddit EDA...")
    analyze_reddit_data()
    print("\n🎉 Analysis complete! Check the output folder for results.")