
import argparse
import os
from pathlib import Path
import re
from typing import List

import pandas as pd

def sort_by_index(paths:List[Path]):
    return sorted(paths,key=lambda x: int(re.search("\d+",x.name).group()))

def create_dataframe(paths:list,key):

  data = [
      {
          "id": int(re.search("\d+",os.path.basename(path)).group()),
            key: path
      }
      for path in paths
  ]
  return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path",type=str,help="path to the experiment")

    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    
    bodies_dir = exp_path / "bodies"
    image_dir = exp_path / "images"
    camera_dir = exp_path / "camera"
    depth_dir = exp_path / "depth-maps"

    bodies_paths = list(bodies_dir.iterdir())
    image_paths =  list(image_dir.iterdir())
    camera_paths = list(camera_dir.iterdir())
    depth_paths =  list(depth_dir.iterdir())
    
    bodies_df = create_dataframe(bodies_paths,key="body")
    image_df  = create_dataframe(image_paths,key="image")
    camera_df = create_dataframe(camera_paths,key="camera")
    depth_df  = create_dataframe(depth_paths,key="depth-map")

    full_df = (
                bodies_df
                .merge(image_df,  on="id", how="outer")
                .merge(camera_df, on="id", how="outer")
                .merge(depth_df,  on="id", how="outer")
            )
       
    nan_entries = full_df[full_df.isna().any(axis=1)]
    # print(nan_entries)
    to_delete = []
    
    for idx,entry in nan_entries.iterrows():
        entry_dict = list(entry.to_dict().values())
        # print(entry_dict)
        for path in entry_dict:
            if path and isinstance(path,(str,Path)):
                to_delete.append(path)

    if to_delete:
        for dentry in to_delete:
            print(f"Deleting {dentry}")
            os.remove(dentry)
        
    else:
        print("No files to be deleted!")

