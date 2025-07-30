import json
import glob
path = "/home/nero/Robotics/DAWN/outputs/DAWN_infer"

lst = sorted(glob.glob(f"{path}/*/results.json"))

for file in lst:
    data = json.load(open(file, "r"))
    print(f"File: {file}: {data['avg_seq_len']}")