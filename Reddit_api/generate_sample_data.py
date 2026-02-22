import os
import json
import random
from datetime import datetime, timedelta
import numpy as np

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Sample data
subreddits = ['python', 'learnpython', 'programming', 'webdev', 'datascience', 'machinelearning', 'javascript', 'reactjs']
authors = [f'user_{i}' for i in range(1, 101)]

# Common words for generating posts
nouns = ['code', 'project', 'problem', 'question', 'help', 'issue', 'error', 'bug', 'function', 'class',
         'variable', 'loop', 'array', 'dictionary', 'list', 'string', 'integer', 'float', 'boolean', 'none']
verbs = ['help', 'solve', 'debug', 'write', 'read', 'print', 'import', 'export', 'create', 'delete',
         'update', 'modify', 'test', 'run', 'execute', 'compile', 'optimize', 'refactor', 'deploy', 'scale']
adjectives = ['simple', 'complex', 'efficient', 'fast', 'slow', 'broken', 'working', 'recursive', 'iterative',
              'object-oriented', 'functional', 'procedural', 'dynamic', 'static', 'typed', 'untyped', 'compiled', 'interpreted']

def generate_post():
    # Random time in the last 30 days
    created_utc = (datetime.utcnow() - timedelta(days=random.randint(0, 30))).timestamp()
    
    # Generate title and body
    title = f"{' '.join(random.choices(adjectives, k=1) + random.choices(nouns, k=2) + random.choices(verbs, k=1))}"
    title = title.capitalize() + '?' if random.random() > 0.3 else title.capitalize()
    
    body = ' '.join([random.choice(verbs).capitalize() + ' ' + 
                    ' '.join(random.choices(nouns + adjectives, k=random.randint(3, 10))) + 
                    ('.' if random.random() > 0.3 else '!') 
                    for _ in range(random.randint(1, 3))])
    
    # Generate some random hashtags
    hashtags = [f"#{tag}" for tag in random.sample(nouns, min(random.randint(0, 3), len(nouns)))]
    
    # Generate some random metrics
    score = int(np.random.normal(100, 50))
    num_comments = int(np.random.poisson(5))
    
    return {
        "id": f"t3_{random.randint(1000000, 9999999)}",
        "title": title,
        "author": random.choice(authors),
        "subreddit": random.choice(subreddits),
        "score": max(0, score),  # Ensure non-negative
        "num_comments": max(0, num_comments),  # Ensure non-negative
        "created_utc": created_utc,
        "selftext": body,
        "url": f"https://www.reddit.com/r/{random.choice(subreddits)}/comments/{random.randint(100000, 999999)}/sample_post/",
        "is_self": True,
        "over_18": False,
        "hashtags": hashtags if random.random() > 0.5 else []
    }

# Generate sample data
print("Generating sample Reddit data...")
sample_data = [generate_post() for _ in range(1000)]

# Save to file
output_file = os.path.join('output', 'Reddit_data.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, indent=2)

print(f"Generated {len(sample_data)} sample posts in {output_file}")
