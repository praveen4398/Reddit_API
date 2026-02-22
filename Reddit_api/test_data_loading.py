#!/usr/bin/env python3
"""
Simple test script to verify data loading and package installation
"""

import os
import json
import pandas as pd

def test_data_loading():
    """Test if we can load the data files"""
    print("Testing data loading...")

    output_dir = "output"

    # Test hashtags file
    hashtags_file = os.path.join(output_dir, "top_hashtags.json")
    try:
        with open(hashtags_file, 'r', encoding='utf-8') as f:
            hashtags_data = json.load(f)
        print(f"✓ Successfully loaded {len(hashtags_data)} hashtags")
        print(f"  Sample: {hashtags_data[0] if hashtags_data else 'No data'}")
    except Exception as e:
        print(f"✗ Error loading hashtags: {e}")
        return False

    # Test keywords file
    keywords_file = os.path.join(output_dir, "top_keywords.json")
    try:
        with open(keywords_file, 'r', encoding='utf-8') as f:
            keywords_data = json.load(f)
        print(f"✓ Successfully loaded {len(keywords_data)} keywords")
        print(f"  Sample: {keywords_data[0] if keywords_data else 'No data'}")
    except Exception as e:
        print(f"✗ Error loading keywords: {e}")
        return False

    # Test a small sample of raw data
    raw_data_file = os.path.join(output_dir, "Reddit_data.json")
    try:
        count = 0
        with open(raw_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    count += 1
                    if count >= 5:  # Just check first 5 records
                        break
        print(f"✓ Successfully read {count} records from raw data")
    except Exception as e:
        print(f"✗ Error reading raw data: {e}")
        return False

    return True

def test_packages():
    """Test if required packages are installed"""
    print("\nTesting package imports...")

    packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly'
    ]

    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            return False

    return True

if __name__ == "__main__":
    print("=== REDDIT EDA TEST SCRIPT ===")

    # Test packages
    if not test_packages():
        print("Package installation issues detected!")
        exit(1)

    # Test data loading
    if not test_data_loading():
        print("Data loading issues detected!")
        exit(1)

    print("\n✓ All tests passed! Ready to run full EDA.")
