import argparse
import numpy as np
import imageio
import os
from tqdm import tqdm
import multiprocessing
from collections import defaultdict
import json
import tensorflow_datasets as tfds

task_key_lists = [
    "structured_language_instruction"
]

lang_key_lists = [
    "language_instruction",
    "natural_language_instruction",
    "instruction",
    "structured_language_instruction"
]

rgb_key_lists = [
    "rgb_static",
    "image",
    "agentview_rgb", "rgb", "front_rgb", "image_1"
]

def get_value_from_episode(episode, lists):
    """
    Utility function to extract a value from an episode based on the provided key.
    If the key is not present, it returns None.
    """
    for key in lists:
        if key in episode['steps']:
            return episode['steps'][key].numpy().tolist()
    for key in lists:
        if key in episode['steps']['observation']:
            return episode['steps']['observation'][key].numpy().tolist()
    

    for step in episode['steps']:
        for key in lists:
            if key in step:
                return step[key].numpy().tolist()
            if 'observation' in step and key in step['observation']:  
                return step['observation'][key].numpy().tolist()
        break   
    return None

def process_episode(args_tuple):
    """
    Processes a single episode: loads data and saves image frames.
    """
    i, episode, output_path = args_tuple
    
    data = defaultdict(list)

    data['length'] = len(episode['steps'])
    data['task'] = get_value_from_episode(episode, task_key_lists)
    data['language'] = get_value_from_episode(episode, lang_key_lists)

    for idx, step in enumerate(episode['steps']):
        rgb_static = get_value_from_episode(step, rgb_key_lists)
        if rgb_static is not None:
            print(idx, rgb_static.shape, rgb_static.dtype)

    # for cnt, idx in enumerate(range(start_idx, end_idx + 1)):
    #     try:
    #         cur = np.load(f"{path}/episode_{idx:07d}.npz", allow_pickle=True)
    #         for key in meta_keys:
    #             data[key].append(cur[key].tolist())
        
    #         for key in keys:
    #             ext = "png" if key.startswith("depth") else "jpg"
    #             img_path = f"{output_path}/episodes/{i}/{key}/{cnt:04d}.{ext}"
    #             if cnt == 0:
    #                 os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
    #             # Ensure the image data is in a savable format (e.g., uint8)
    #             img_data = cur[key]
    #             if ext == "png":
    #                 print(key, np.min(img_data), np.max(img_data), img_data.shape, img_path)
    #             if img_data.dtype != np.uint8:
    #                 img_data = img_data.astype(np.uint8)
                
    #             imageio.imwrite(img_path, img_data)
    #     except FileNotFoundError:
    #         # Optional: handle cases where an episode file might be missing
    #         print(f"Warning: File not found for episode {idx:07d}. Skipping.")
    #         continue
    
    # print(data)
    
    # json.dump(data, open(f"{output_path}/episodes/{i}/metadata.json", "w"), indent=4)



def dataset2path(dataset_name, root_dir="/nfs/mercedes/hdd1/rt-x"):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    elif dataset_name == "bridge":
        return "/nfs/ws3/hdd1/kanchana/data/bridge/0.1.0"
    else:
        version = "0.1.0"
    return f"{root_dir}/{dataset_name}/{version}"

if __name__ == "__main__":
    # Set start method for multiprocessing compatibility, especially on macOS/Windows
    multiprocessing.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser(description="Preprocess OpenX dataset for training")
    parser.add_argument("--data_path", type=str, help="Path to OpenX data")
    parser.add_argument("--subset", type=list, default=["fractal20220817_data, taco_play"], help="Subset of the dataset to preprocess (e.g., ['taco_play', 'fractal20220817_data'])")
    parser.add_argument("--output_path", type=str, help="Path to save preprocessed data")
    # Optional: allow user to specify number of CPU cores to use
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use")
    args = parser.parse_args()

    subsets = args.subset

    splits = ["train"]
    for subset in subsets:
        builder = tfds.builder_from_directory(
            builder_dir=dataset2path(subset, root_dir=args.data_path))
        for s in splits:
            ds = builder.as_dataset(split='train')
            print(f"Processing dataset_split: {subset}_{s}")
            output_path = f"{args.output_path}/{subset}/{s}"
            os.makedirs(f"{output_path}/episodes", exist_ok=True)
            
            tasks = []
            for i, episode in enumerate(tqdm(ds)):
                tasks.append((i, episode, output_path))
            
            # lang_data = np.load(f"{path}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
            
            # ep_start_end_ids = lang_data["info"]["indx"]
            # lang_ann = lang_data["language"]["ann"]
            
            # keys = ["rgb_static", "rgb_gripper"] # "depth_static", "depth_gripper", 
            # meta_keys = ["actions", "rel_actions"]
            # # Prepare a list of arguments for each task

            # set_lang = set(lang_ann)
            # print(f"Unique languages found: {len(set_lang)}")
            # lang_emb = {lang: model([lang]).numpy().tolist() for lang in set_lang}
            # tasks = []
            # from tqdm import tqdm
            # for i, ((start_idx, end_idx), lang) in enumerate(tqdm(zip(ep_start_end_ids, lang_ann))):
            #     embedding = lang_emb[lang]
            #     tasks.append((i, start_idx, end_idx, path, output_path, keys, meta_keys, lang, embedding))

            # Create a pool of worker processes
            # The 'with' statement ensures the pool is properly closed
            with multiprocessing.Pool(processes=args.num_workers) as pool:
                # Use tqdm to display a progress bar
                list(tqdm(pool.imap_unordered(process_episode, tasks), total=len(tasks)))

    print("Preprocessing complete.")