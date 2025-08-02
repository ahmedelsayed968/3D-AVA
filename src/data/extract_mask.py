from abc import ABC
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import torch
from torchvision.utils import save_image
import json
from pathlib import Path
import argparse
import re 

class MaskGeneratorBase(ABC):
    def generate_mask(self,image:Image,bbox:List[List[List[float]]]):
        pass

class SAMMaskGenerator(MaskGeneratorBase):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    def generate_mask(self, image, bbox)->torch.Tensor:
        inputs = self.processor(
            images=image,
            input_boxes=bbox,
            return_tensors="pt").to(self.device)
        
        with torch.no_grad():
        
            outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                            inputs["original_sizes"].cpu(),
                                                            inputs["reshaped_input_sizes"].cpu())
        mask = masks[0].squeeze(0).squeeze(0)  # shape (H, W)
        mask = mask.unsqueeze(0).float()  # convert to (1, H, W) float
        return mask


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path",type=str,help="path to the experiment")
    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    frame_dir = exp_path / "images"
    bbox_dir = exp_path / "bbox"
    mask_dir = exp_path /"masks"
    mask_dir.mkdir(parents=True,exist_ok=True)

    all_bbox_paths = list(bbox_dir.iterdir())

    mask_generator = SAMMaskGenerator()

    for bbox_path in tqdm(all_bbox_paths,desc="Generating Masks"):
        file_name = re.search("\d+",bbox_path.name).group()
        with bbox_path.open("r") as fin:
             bbox_dict = json.load(fp=fin) 
        bbox_list = list(bbox_dict.values())[:-1] # ignore confidence   
        bbox = [[bbox_list]]
        
        image_path = frame_dir / f"{file_name}.png"
        image_raw = Image.open(image_path)
        
        mask = mask_generator.generate_mask(image=image_raw,
                                            bbox=bbox)

        to_save = mask_dir / f"{file_name}.png"
        save_image(mask * 255,to_save)
