import json
from pathlib import Path
import argparse
from data.schema import KeyPoints2D
from glob import glob
from dataclasses import asdict
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path",type=str,help="path to the experiment")
    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    bodies_path = Path(exp_path) / "bodies"
    keypoint_dir = exp_path / "keypoints"
    keypoint_dir.mkdir(parents=True,exist_ok=True)
    all_bodie_paths = list(bodies_path.iterdir())
    for path in tqdm(all_bodie_paths,desc="Extracting Keypoint"):
        bbox_obj = KeyPoints2D.from_path(path=path)
        filename = path.name
        bbox_dict = asdict(bbox_obj)

        save_to = keypoint_dir / filename
        with save_to.open('w') as fin:
            json.dump(bbox_dict,fp=fin,indent=4)
        
        
