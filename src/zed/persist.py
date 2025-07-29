import json
from pathlib import Path
from loguru import logger
import numpy as np
from PIL import Image


class ZedSaver:
    @staticmethod
    def save_depth_map(depth_image_np: np.ndarray, exp_path: Path, counter: int):
        save_to = exp_path / f"Pointcloud_{counter+1}.npy"
        depth_image_np = np.asarray(depth_image_np)
        np.save(save_to, depth_image_np)
        logger.debug(f"Saved depth map to {save_to}")

    @staticmethod
    def save_image_from_zed(image_np: np.ndarray, exp_path: Path, counter: int):
        if image_np is None:
            logger.error("Cannot save None image")
            return
            
        save_to = exp_path / f"image_{counter+1}.png"
        try:
            # Convert BGRA to RGBA if needed (ZED uses BGRA format)
            if image_np.shape[2] == 4:
                # Convert BGRA to RGBA
                image_np = image_np[:, :, [2, 1, 0, 3]]
            
            image = Image.fromarray(image_np)
            
            # Check if the image has an alpha channel
            if image.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                # Paste the RGBA image onto the background
                background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
                image = background

            # Save the image
            image.save(save_to)
            logger.debug(f"Saved image to {save_to}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")
    
    @staticmethod
    def save_mask(mask_np: np.ndarray, exp_path: Path, counter: int):
        """Save segmentation mask as image"""
        if mask_np is None:
            logger.error("Cannot save None mask")
            return
            
        save_to = exp_path / f"mask_{counter+1}.png"
        try:
            # Convert to grayscale if needed
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]  # Take first channel
            
            mask_image = Image.fromarray(mask_np, mode='L')
            mask_image.save(save_to)
            logger.debug(f"Saved mask to {save_to}")
        except Exception as e:
            logger.error(f"Error saving mask: {e}")
    
    @staticmethod
    def save_keypoints_and_masks(frame_data: dict, exp_path: Path, counter: int):
        """Save keypoints and mask data to JSON file"""
        if not frame_data['bodies']:
            logger.debug(f"No bodies detected in frame {counter}")
            return
        
        save_to = exp_path / f"keypoints_{counter+1}.json"
        try:
            with open(save_to, 'w') as f:
                json.dump(frame_data, f, indent=2)
            logger.debug(f"Saved keypoints data to {save_to}")
        except Exception as e:
            logger.error(f"Error saving keypoints data: {e}")

