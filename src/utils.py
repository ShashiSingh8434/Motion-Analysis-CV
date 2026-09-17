import os
import json

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
