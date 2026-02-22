import json
import os

def inspect_file(file_path):
    print(f"\nInspecting file: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    # Try to read first 5 lines
    print("\nFirst 5 lines:")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"Line {i+1} (first 200 chars): {line[:200].strip()}")
    
    # Try to detect file format
    print("\nTrying to detect file format...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try as JSON array
            try:
                data = json.load(f)
                if isinstance(data, list):
                    print("✓ Detected: JSON array")
                    print(f"Number of items: {len(data)}")
                    if data:
                        print("\nFirst item keys:", list(data[0].keys()))
                    return
            except json.JSONDecodeError:
                pass
            
            # Try as JSONL
            f.seek(0)
            try:
                first_line = next(f, '').strip()
                if first_line:
                    json.loads(first_line)
                    print("✓ Detected: JSONL (newline-delimited JSON)")
                    return
            except (json.JSONDecodeError, StopIteration):
                pass
            
            print("✗ Could not detect format - file may be corrupted or in an unsupported format")
            
    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    file_path = os.path.join("output", "Reddit_data.json")
    if os.path.exists(file_path):
        inspect_file(file_path)
    else:
        print(f"File not found: {file_path}")
        print("Current directory:", os.getcwd())
        print("Files in output directory:", os.listdir("output"))
