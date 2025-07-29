from typing import List, Optional
from loguru import logger
import numpy as np
from pyzed import sl

class ZedRetrieval:
    def __init__(self,zed):
        self.zed = zed
        pass

    def zed_retrieve_object_detections(self, runtime_params: sl.ObjectDetectionRuntimeParameters, instance_id: int) -> Optional[List[sl.ObjectData]]:
        objects = sl.Objects()
        if self.zed.retrieve_objects(objects, runtime_params, instance_id) != sl.ERROR_CODE.SUCCESS:
            logger.error("Cannot retrieve detected objects")
            return None
        return objects.object_list


    def zed_retrieve_depth_map(self, camera_res, deep_copy: bool = True) -> Optional[np.ndarray]:
        depth = sl.Mat(camera_res.width,
                    camera_res.height,
                    memory_type=sl.MEM.CPU,
                    mat_type=sl.MAT_TYPE.F32_C1)
        
        err = self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error("Cannot retrieve the depth map")
            return None
        depth_image_np = depth.get_data(deep_copy=deep_copy)
        return depth_image_np
    
    def zed_retrieve_left_image(self, camera_res) -> Optional[np.ndarray]:
        left_image = sl.Mat(camera_res.width,
                            camera_res.height,
                            memory_type=sl.MEM.CPU,
                            mat_type=sl.MAT_TYPE.U8_C4)
        err = self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error("Cannot retrieve the left image")
            return None
        
        image_np = left_image.get_data(deep_copy=True)
        return image_np
    
    def zed_retrieve_bodies(self,tracking_params,instance_module_id:int) -> Optional[sl.Bodies]:
        """Retrieve body tracking data from ZED camera"""
        bodies = sl.Bodies()
        
        if self.zed.retrieve_bodies(bodies, tracking_params,instance_module_id) == sl.ERROR_CODE.SUCCESS:
            return bodies
        else:
            logger.error("Cannot retrieve body tracking data")
            return None
    
    def zed_extract_segmentation_masks(self, objects: List[sl.ObjectData]) -> List[np.ndarray]:
        masks = []
        for obj in objects:
            if obj.mask and obj.mask.is_init():
                masks.append(obj.mask.get_data(deep_copy=True))
        return masks

        

    @staticmethod
    def extract_keypoints_and_masks(bodies: sl.Bodies, camera_res) -> dict:
        """Extract 2D/3D keypoints and masks from body tracking data"""
        frame_data = {
            'bodies': [],
            'timestamp': bodies.timestamp.get_nanoseconds()
        }
        
        for body in bodies.body_list:
            if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                body_data = {
                    'id': body.id,
                    'confidence': body.confidence,
                    'action_state': str(body.action_state),
                    'keypoints_2d': [],
                    'keypoints_3d': [],
                    'bounding_box_2d': [],
                    'bounding_box_3d': [],
                    'mask': None
                }
                
                # Extract 2D keypoints
                keypoints_2d = body.keypoint_2d
                for i, kp in enumerate(keypoints_2d):
                    if i < len(keypoints_2d):
                        body_data['keypoints_2d'].append({
                            'joint_id': i,
                            'x': float(kp[0]),
                            'y': float(kp[1])
                        })
                
                # Extract 3D keypoints
                keypoints_3d = body.keypoint
                for i, kp in enumerate(keypoints_3d):
                    if i < len(keypoints_3d):
                        body_data['keypoints_3d'].append({
                            'joint_id': i,
                            'x': float(kp[0]),
                            'y': float(kp[1]),
                            'z': float(kp[2])
                        })
                
                # Extract 2D bounding box
                bbox_2d = body.bounding_box_2d
                for corner in bbox_2d:
                    body_data['bounding_box_2d'].append({
                        'x': float(corner[0]),
                        'y': float(corner[1])
                    })
                
                # Extract 3D bounding box
                bbox_3d = body.bounding_box
                for corner in bbox_3d:
                    body_data['bounding_box_3d'].append({
                        'x': float(corner[0]),
                        'y': float(corner[1]),
                        'z': float(corner[2])
                    })
                
                frame_data['bodies'].append(body_data)
        
        return frame_data


