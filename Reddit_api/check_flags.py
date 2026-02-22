import json

# Check behavior flags
print("Checking behavior flags data...")
with open('output/reddit_behaviour_flags.json', 'r', encoding='utf-8') as f:
    behavior_data = json.load(f)

print(f"Total behavior records: {len(behavior_data)}")

# Count spam and offensive
spam_count = 0
offensive_count = 0

for record in behavior_data[:5000]:  # Check first 5000
    flags = record.get('flags', {})
    if flags.get('spam'):
        spam_count += 1
    if flags.get('offensive'):
        offensive_count += 1

print(f"Spam posts (first 5000): {spam_count}")
print(f"Offensive posts (first 5000): {offensive_count}")

# Check sample
print("\nSample record:")
print(json.dumps(behavior_data[0], indent=2)[:500])

# Check post_summaries
print("\n\nChecking post_summaries.json...")
with open('output/post_summaries.json', 'r', encoding='utf-8') as f:
    posts_data = json.load(f)

print(f"Total posts: {posts_data.get('total_posts', 0)}")

# Check if behavior flags are in post summaries
sample_post = posts_data['posts'][0]
print("\nSample post NLP keys:", list(sample_post.get('nlp_results', {}).keys()))
print("Behavior flags:", sample_post.get('nlp_results', {}).get('behavior_flags', {}))
