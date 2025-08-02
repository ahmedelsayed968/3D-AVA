
import argparse
from dataclasses import astuple
import json
from pathlib import Path
import re

from tqdm import tqdm

from data.schema import BBox, KeyPoints2D, Keypoint2D


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path",type=str,help="path to the experiment")
    parser.add_argument("--height",type=int,help="height of the image",default=720)
    parser.add_argument("--width",type=int,help="width of the image",default=1280)

    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    height = args.height
    width = args.width
    keypoint_dir = exp_path / "keypoints"
    keypoint_paths = list(keypoint_dir.iterdir())
    keypoint_paths.sort(key=lambda x: int(re.search("\d+",x.name).group()))
    bbox_dir = exp_path / "bbox"
    bbox_paths = list(bbox_dir.iterdir())
    bbox_paths.sort(key=lambda x: int(re.search("\d+",x.name).group()))
  
    annots = exp_path / "annots"
    annots.mkdir(parents=True,exist_ok=True)

 
    for kpath,bpath in tqdm(zip(keypoint_paths,bbox_paths),desc="Creating EasyMocap Data",total=len(keypoint_paths)):
        with kpath.open('r') as fin:
            keypoint_dict = json.load(fp=fin)
        
        points = []    
        for point in keypoint_dict['points']:
            point= Keypoint2D(**point)
            points.append(point)
        keypoint_dict['points'] = points
        keypoint_obj = KeyPoints2D(**keypoint_dict)

        with bpath.open('r') as fin:
            bbox_dict = json.load(fp=fin)
        
        bbox_obj = BBox(**bbox_dict)
        area = bbox_obj.compute_area()


        keypoints = [
            list(astuple(point))[1:]
            for point in keypoint_obj.get_body25()
        ]
        keypoints_with_conf = [
            [*point,keypoint_obj.confidence]
            for point in keypoints
        ]
        annot = {
            "personID": 1,
            "bbox": list(astuple(bbox_obj)),
            "keypoints":keypoints_with_conf,
            "area": area,
        }

        idx = re.search("\d+",kpath.name).group()
        
        all_annot = {
            "filename": str(exp_path / "images"/ f"{idx}.png"),
            "height": height,
            "width" : width,
            "annots": [
                annot
            ]
        }

        path_to_save =  annots / f"{idx}.json" 
        # print(path_to_save)
        with open(path_to_save,mode="w") as fin:
            json.dump(all_annot, fin, indent=4)
