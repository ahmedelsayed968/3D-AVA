from dataclasses import astuple
import json
from pathlib import Path
from data.schema import BBox, KeyPoints2D
from transformers import SamModel, SamProcessor
from torchvision.utils import save_image
import torch
def get_keypoints_2d_from_raw_bodies(data:dict):
  return KeyPoints2D.from_zed_format(data['bodies'][0]['keypoints_2d'],
                                     confidence=get_confidence_from_raw_bodies(data))

def get_keypoints_3d_from_raw_bodies(data:dict):
  return data['bodies'][0]['keypoints_3d']

def get_bbox_2d_from_raw_bodies(data:dict):
  return BBox.from_zed_format(data['bodies'][0]['bounding_box_2d'],
                              confidence=get_confidence_from_raw_bodies(data))

def get_bbox_3d_from_raw_bodies(data:dict):
  return data['bodies'][0]['bounding_box_3d']
def get_confidence_from_raw_bodies(data):
  return data['bodies'][0]['confidence']

def read_json(path:str):
  with open(path,mode="r") as fin:
    data = json.load(fin)
  return data


def create_easymocap_annotation(entry,annotation_path:Path):
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
        "height": 720,
        "width" : 1280,
        "annots": [
            annot
        ]
    }

    path_to_save = str(annotation_path / f"{id:05}.json")
    # print(path_to_save)
    with open(path_to_save,mode="w") as fin:
        json.dump(all_annot, fin, indent=4)

    entry['annot'] = path_to_save
    return entry

def create_bbox_df_column(entry):
    keypoint_path =  Path(entry['keypoint'])
    bodies = read_json(str(keypoint_path))
    bbox_obj = get_bbox_2d_from_raw_bodies(data=bodies)
    return [[list(astuple(bbox_obj))[:-1]]]

def generate_mask_with_sam(entry,
                           processor:SamProcessor,
                           model:SamModel,
                           device:torch.device,
                           exp_path:Path):
    idx = entry['id']
    inputs = processor(
        images=entry['image'],
        input_boxes=entry['bbox'],
        return_tensors="pt").to(device)

    with torch.no_grad():
      outputs = model(**inputs, multimask_output=False)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                         inputs["original_sizes"].cpu(),
                                                         inputs["reshaped_input_sizes"].cpu())
    mask = masks[0].squeeze(0).squeeze(0)  # shape (H, W)
    mask = mask.unsqueeze(0).float()  # convert to (1, H, W) float
    path_to_save = exp_path / "masks" / f"{idx:05}.png"
    save_image(mask * 255, path_to_save)
    return {
        "mask": str(path_to_save)
    }
