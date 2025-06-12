import argparse
import numpy as np
import imageio
import os
from tqdm import tqdm
import multiprocessing
from collections import defaultdict
import json
import tensorflow_hub as hub
if __name__ == "__main__":
    # Set start method for multiprocessing compatibility, especially on macOS/Windows
    parser = argparse.ArgumentParser(description="Preprocess Calvin dataset for training")
    parser.add_argument("--data_path", type=str, help="Path to calvin data")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    args = parser.parse_args()
    import glob 

    cache = {}
    meta_data_list = glob.glob(f"{args.data_path}/*/*/*/*/metadata.json")
    print(len(meta_data_list), "metadata files found.")
    for meta_data in tqdm(meta_data_list, desc="Processing metadata files"):
        with open(meta_data, "r") as f:
            data = json.load(f)
        
        # Extract language and embedding
        lang = data.get("language", "")
        
        if lang not in cache:
            cache[lang] = model([lang]).numpy()[0].tolist()
            
        embedding = cache[lang]
        data['language_embedding'] = embedding

        json.dump(data, open(meta_data, "w"), indent=4)
    
        # break
    # print(meta_data_list)
    # print("Preprocessing complete.")