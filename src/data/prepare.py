import argparse
from dataclasses import astuple
import json
import os
from pathlib import Path
from glob import glob
import re
import shutil
from typing import Tuple, List
import pandas as pd

from utils.data_utils import get_bbox_2d_from_raw_bodies, get_keypoints_2d_from_raw_bodies, read_json

def create_dataframe(paths:list):

  data = [
      {
          "id": int(re.search("\d+",os.path.basename(path)).group()),
          "path": str(path)
      }
      for path in paths
  ]
  return pd.DataFrame(data)

def move_all_files(entry:pd.Series,keypoint_dir:Path,image_dir:Path,depth_dir:Path):
    # print(entry)
    new_id = entry['id']
    old_keypoint_path = entry['keypoint']
    old_image_path = entry['image']
    old_depth_map_path = entry['depth_map']

    # Define new file paths
    new_keypoint_path = keypoint_dir / f"{new_id:05}.json"
    new_image_path = image_dir / f"{new_id:05}.png"
    new_depth_map_path = depth_dir / f"{new_id:05}.npy"

    try:
        # Copy files to new locations
        shutil.copy(src=old_keypoint_path, dst=str(new_keypoint_path))
        shutil.copy(src=old_image_path, dst=str(new_image_path))
        shutil.copy(src=old_depth_map_path, dst=str(new_depth_map_path))

        # Update entry with new file paths
        entry['keypoint'] = str(new_keypoint_path)
        entry['image'] = str(new_image_path)
        entry['depth_map'] = str(new_depth_map_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Optionally, log or handle the error in a way that suits your needs
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return entry

def create_easymocap_annotation(entry:pd.Series,annot_dir:Path,width:int=1280,height:int=720):
    id = entry['id']
    bodies = read_json(entry['keypoint'])

    keypoints_obj = get_keypoints_2d_from_raw_bodies(data=bodies)
    bbox_obj = get_bbox_2d_from_raw_bodies(data=bodies)

    area = bbox_obj.compute_area()

    keypoints = [
        list(astuple(point))[1:]
        for point in keypoints_obj.get_body25()
    ]
    keypoints_with_conf = [
        [*point,keypoints_obj.confidence]
        for point in keypoints
    ]
    annot = {
        "personID": 1,
        "bbox": list(astuple(bbox_obj)),
        "keypoints":keypoints_with_conf,
        "area": area,
    }
    all_annot = {
        "filename": str(entry['image']),
        "height": height,
        "width" : width,
        "annots": [
            annot
        ]
    }

    path_to_save = str(annot_dir / f"{id:05}.json")
    # print(path_to_save)
    with open(path_to_save,mode="w") as fin:
        json.dump(all_annot, fin, indent=4)

    entry['annot'] = path_to_save
    return entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_exp_path", type=str,
                        help="Path to raw data captured from ZED")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Output path to save the formatted dataset")
    parser.add_argument("-f", "--format_dir", action="store_true",
                        help="Flag to format the directory structure")
    parser.add_argument("-emc", "--easymocap_prepare", action="store_true",
                        help="Flag to format the directory structure")
    
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)


    if args.format_dir:

        input_path = Path(args.input_exp_path)
        
        image_pattern=str(input_path / '**' / '*.png')
        depth_pattern=str(input_path / '**' / '*.npy')
        keypoint_pattern=str(input_path / '**' / '*.json')

        image_paths = glob(image_pattern,recursive=True) 
        depth_paths = glob(depth_pattern,recursive=True) 
        keypoint_paths = glob(keypoint_pattern,recursive=True)

        images_df = create_dataframe(image_paths)
        depth_map_df = create_dataframe(depth_paths)
        keypoints_df = create_dataframe(keypoint_paths)

        columns_map= {
            "path_x": "keypoint",
            "path_y": "image",
            "path":"depth_map"
        }
        all_df = keypoints_df.merge(images_df,on="id",how="inner")\
                            .merge(depth_map_df,on="id",how="inner")\
                            .sort_values("id")\
                            .rename(columns=columns_map)
        # create new index
        all_df['id'] = range(len(all_df))
        
        IMAGE_DIR = output_path / "images"
        KEYPOINT_DIR = output_path / "keypoints"
        DEPTH_MAP_DIR = output_path / "depth-maps"

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)
        DEPTH_MAP_DIR.mkdir(parents=True, exist_ok=True)

        all_df = all_df.apply(move_all_files,axis=1,args=(KEYPOINT_DIR,IMAGE_DIR,DEPTH_MAP_DIR))
        all_df.to_csv(output_path/"annotation.csv",index=False)

    if args.easymocap_prepare:
        df = pd.read_csv(output_path/"annotation.csv")
        ANNOTS_DIR = output_path / "annots"
        ANNOTS_DIR.mkdir(parents=True, exist_ok=True) 
        df = df.apply(create_easymocap_annotation,axis=1,args=(ANNOTS_DIR,))
        df.to_csv(output_path/"annotation.csv",index=False)

if __name__ == "__main__":
    main()
