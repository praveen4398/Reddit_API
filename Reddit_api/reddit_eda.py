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

def load_data_sample(file_path, max_records=5000):
    """Load a sample of JSON data with error handling, supports both JSON array and JSONL formats"""
    print(f"Loading data from {file_path}...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # First try to load as JSON array
            try:
                content = json.load(f)
                if isinstance(content, list):
                    print(f"Loaded {min(len(content), max_records)} records from JSON array")
                    return content[:max_records]
            except json.JSONDecodeError:
                pass
            
            # If not a JSON array, try reading as JSONL
            f.seek(0)
            for i, line in enumerate(f):
                if i >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {i+1}: {e}")
                    continue
        
        print(f"Loaded {len(data)} records from JSONL file")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def analyze_reddit_data():
    """Main function to perform EDA on Reddit data"""
    # Load data
    data_file = os.path.join(OUTPUT_DIR, "Reddit_data.json")
    print(f"\n{'='*50}")
    print(f"Starting EDA on file: {os.path.abspath(data_file)}")
    print(f"File exists: {os.path.exists(data_file)}")
    print(f"File size: {os.path.getsize(data_file) / (1024*1024):.2f} MB")
    
    raw_data = load_data_sample(data_file, 5000)
    
    if not raw_data:
        print("❌ No data loaded. Exiting...")
        return
    
    print(f"\n✅ Successfully loaded {len(raw_data)} records")
    print(f"First record keys: {list(raw_data[0].keys())}")
    print(f"First record sample: {str(raw_data[0])[:200]}...")

    # Convert to DataFrame with better error handling
    try:
        df = pd.DataFrame(raw_data)
        print(f"\n✅ Successfully created DataFrame with shape: {df.shape}")
        print("\n=== COLUMNS ===")
        print(df.dtypes)
        print("\n=== FIRST FEW ROWS ===")
        print(df.head(2).to_string())
        
        # Basic info
        print("\n=== DATA OVERVIEW ===")
        if 'created_utc' in df.columns:
            print(f"Time period: {df['created_utc'].min()} to {df['created_utc'].max()}")
        if 'subreddit' in df.columns:
            print(f"Unique subreddits: {df['subreddit'].nunique()}")
        if 'author' in df.columns:
            print(f"Unique authors: {df['author'].nunique()}")
            
    except Exception as e:
        print(f"\n❌ Error creating DataFrame: {e}")
        print("\nRaw data sample:", str(raw_data[:1]))
        return
    
    # Convert date with better error handling
    if 'created_utc' in df.columns:
        try:
            # Try to convert from timestamp (float or int)
            if pd.api.types.is_numeric_dtype(df['created_utc']):
                df['created_at'] = pd.to_datetime(df['created_utc'], unit='s')
            else:
                # Try to parse as ISO format string
                df['created_at'] = pd.to_datetime(df['created_utc'])
                
            # Extract time components
            df['hour'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.day_name()
            
            print(f"Successfully parsed dates from {df['created_at'].min()} to {df['created_at'].max()}")
        except Exception as e:
            print(f"Warning: Could not parse dates: {e}")
    else:
        print("Warning: 'created_utc' column not found in data")
    
    # Calculate text length
    df['text_length'] = 0
    if 'title' in df.columns:
        df['text_length'] += df['title'].str.len().fillna(0)
    if 'body' in df.columns:
        df['text_length'] += df['body'].str.len().fillna(0)
    
    # 1. Posts by Subreddit (Top 10)
    plt.figure(figsize=(12, 6))
    top_subreddits = df['subreddit'].value_counts().head(10)
    sns.barplot(y=top_subreddits.index, x=top_subreddits.values, palette='viridis')
    plt.title('Top 10 Subreddits by Post Count', fontsize=14)
    plt.xlabel('Number of Posts')
    plt.ylabel('Subreddit')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_subreddits.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Posts by Hour of Day
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
    
    # 3. Posts by Day of Week
    plt.figure(figsize=(10, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_posts = df['day_of_week'].value_counts().reindex(day_order)
    sns.barplot(x=weekly_posts.index, y=weekly_posts.values, palette='coolwarm')
    plt.title('Posts by Day of Week', fontsize=14)
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'posts_by_weekday.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Keywords Analysis
    plt.figure(figsize=(12, 8))
    try:
        # Count keywords (flatten the list of lists and count occurrences)
        all_keywords = [kw for sublist in df['keywords'].dropna() for kw in sublist]
        if all_keywords:
            keyword_counts = pd.Series(all_keywords).value_counts().head(20)
            sns.barplot(y=keyword_counts.index, x=keyword_counts.values, palette='viridis')
            plt.title('Top 20 Most Common Keywords', fontsize=14)
            plt.xlabel('Frequency')
            plt.ylabel('Keyword')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'top_keywords.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created top keywords plot")
        else:
            print("⚠ No keywords found for analysis")
    except Exception as e:
        print(f"⚠ Could not create keywords plot: {e}")
    
    # 5. Hashtags Analysis
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
    
    # 6. Top Authors
    plt.figure(figsize=(12, 6))
    top_authors = df['author'].value_counts().head(15)
    sns.barplot(y=top_authors.index, x=top_authors.values, palette='magma')
    plt.title('Top 15 Most Active Authors', fontsize=14)
    plt.xlabel('Number of Posts')
    plt.ylabel('Author')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_authors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Word Cloud of Titles
    if 'title' in df.columns:
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
    
    print("\n✅ EDA complete! Check the output folder for visualizations.")

if __name__ == "__main__":
    print("🚀 Starting Reddit EDA...")
    analyze_reddit_data()
    print("\n🎉 Analysis complete! Check the output folder for results.")
