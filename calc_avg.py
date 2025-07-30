import argparse
import os
from collections import Counter
import glob
def count_success(results, seq_len=5):
    count = Counter(results)
    step_success = []
    for i in range(1, seq_len + 1):
        n_success = sum(count[j] for j in reversed(range(i, seq_len + 1)))
        sr = n_success
        step_success.append(sr)
    return step_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script to print a message.")
    parser.add_argument("--input", type=str)
    
    args = parser.parse_args()
    
    inp = args.input
    # files = os.listdir(inp)
    files = glob.glob(os.path.join(inp, "*.mp4"))
    results = [int(file.split("=")[-1].split(".")[0]) for file in files]
    
    print(f"Input directory: {inp}")
    print(f"Files found: {len(files)}")
    seq_len = 5
    success_count = count_success(results, seq_len)
    success_rates = [v / len(files) for v in success_count]
    average_rate = sum(success_rates) / len(success_rates) * seq_len
    description = " ".join([f"{i + 1}/{seq_len} : {v:.3f}% |" for i, v in enumerate(success_rates)])
    description += f" | Average: {average_rate:.3f} "    
    
    print(description)
    description = " ".join([f"{i + 1}/{seq_len} : {v}/{len(files)} |" for i, v in enumerate(success_count)])
    print(description)
    
    
    