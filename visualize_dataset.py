from src.util import compare_trajectory
from src.config import Config
import argparse
import random
import glob
import json
import copy

parser = argparse.ArgumentParser(description="Visualise OVITA dataset")

parser.add_argument(
        "--random_sample", 
        type=bool,
        default=True,
        help="Sample a random trajectory"
    )

parser.add_argument(
        "--path", 
        type=str,
        help="Speicfy the path of the datapoint"
    )


args=parser.parse_args()

if args.random_sample:
    dataset_path="Dataset"
    json_files = glob.glob(dataset_path+"/**/*.json", recursive=True)
    file_path=random.choice(json_files)
else:
    if not args.path:
        print("Error: --path argument is required! or select random sampling")
        exit(1)
    file_path=args.path

with open(file_path,"r") as infile:
    data=json.loads(infile.read())

config=Config()

def detect_objects():
    objs=copy.deepcopy(data['objects'])
    for item in objs:
        item['name']=item['name'].lower()
        if 'dimensions' not in item.keys():
            item.update({'dimensions': [config.DEFAULT_DIMENSION]*3})
    return objs

print("The instruction is ",data['instruction'])
compare_trajectory(original_trajectory=data['trajectory'],modified_trajectory=data['trajectory'],title=data['instruction'], points=detect_objects())


        





