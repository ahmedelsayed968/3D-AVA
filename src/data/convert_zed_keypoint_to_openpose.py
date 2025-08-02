"""
version	:	1.2
	people		[1]
        pose_keypoints_2d		[75] (x0,y0,c0,x1,y1,c1,.....,)
        face_keypoints_2d		[210]
        hand_left_keypoints_2d		[63]
        hand_right_keypoints_2d		[63]
        pose_keypoints_3d		[0]
        face_keypoints_3d		[0]
        hand_left_keypoints_3d		[0]
        hand_right_keypoints_3d		[0]
"""

import argparse
from dataclasses import astuple
import glob
import json
import os
from pathlib import Path
import re

from tqdm import tqdm

from data.schema import KeyPoints2D, Keypoint2D


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path",type=str,help="path to the experiment")
    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    
    keypoint_dir = exp_path / "keypoints"
    keypoint_paths = list(keypoint_dir.iterdir())

    output_folder = exp_path/ "openpose-keypoints"
    output_folder.mkdir(parents=True,exist_ok=True)

    for path in tqdm(keypoint_paths,desc="Converting Keypoints"):
        with path.open('r') as fin:
            keypoint_dict = json.load(fp=fin)
        points = []    
        for point in keypoint_dict['points']:
            point= Keypoint2D(**point)
            points.append(point)
        keypoint_dict['points'] = points
        keypoint_obj = KeyPoints2D(**keypoint_dict)

 
        index = re.search("\d+",path.name).group()

        keypoints = [
            list(astuple(point))[1:]
            for point in keypoint_obj.get_body25()
        ]
        keypoints_with_conf = [
            [*point,keypoint_obj.confidence]
            for point in keypoints
        ]
        flat_keypoint = [pnt for pair in keypoints_with_conf for pnt in pair]
        openpose_format = {
        "version": 1.2,
        "people": [
                {
                "pose_keypoints_2d": flat_keypoint,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": []		,
                "face_keypoints_3d"	: [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []		 
                }
        ]


    }
        output_name = f"{index}_keypoints.json"
            # Write to file
        with open(os.path.join(output_folder, output_name), 'w') as f:
            json.dump(openpose_format, f, indent=4)