"""
Quick script to regenerate all data needed for the dashboard
"""

import subprocess
import sys

print("=" * 70)
print("REGENERATING DASHBOARD DATA")
print("=" * 70)

steps = [
    ("1. Regenerating behavior flags (with fixed format)...", "python behaviour_flags.py"),
    ("2. Regenerating post summaries (merging all NLP data)...", "python generate_post_summaries.py"),
    ("3. Regenerating overview statistics...", "python generate_overview.py"),
]

for i, (desc, cmd) in enumerate(steps, 1):
    print(f"\n{desc}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Step {i} failed!")
        sys.exit(1)
    print(f"✅ Step {i} complete!")

print("\n" + "=" * 70)
print("✅ ALL DATA REGENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nYou can now run the dashboard:")
print("  python dashboard_app.py")
print("\nThen open: http://localhost:5000")
