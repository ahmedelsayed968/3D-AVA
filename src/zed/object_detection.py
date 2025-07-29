from pyzed import sl
from loguru import logger
import sys

# logger.remove(0)
# logger.add(sys.stdout,level="TRACE")

class ObjectDetection:
    def __init__(self,zed):
        self.enable_positional_tracking(zed)
        self.obj_param = self.enable_object_detection()
        self.obj_runtime_params = self.set_runtime_params()
        self.is_enabled = False
        if zed.enable_object_detection(self.obj_param) != sl.ERROR_CODE.SUCCESS:
            logger.error("Failed to enable object detection")
            zed.close()
            exit()
        else:
            self.is_enabled = True
            logger.success("Successfully enabled the object detection module")
       
    def enable_positional_tracking(self,zed):
        # Enable positional tracking module
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static in space, enabling this setting below provides better depth quality and faster computation
        # positional_tracking_parameters.set_as_static = True
        zed.enable_positional_tracking(positional_tracking_parameters)

    def enable_object_detection(self):
         # Enable object detection module
        obj_param = sl.ObjectDetectionParameters()
        obj_param.instance_module_id = 1  # Ensure unique ID
        obj_param.enable_tracking = True  # Enable object tracking
        obj_param.enable_segmentation = True # Enable Segmentation to get the mask
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM	  # Accurate detection model
        return obj_param
    
    def set_runtime_params(self):        
        # Set runtime parameters
        obj_runtime_params = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_params.detection_confidence_threshold = 50  # Confidence threshold
        # To select a set of specific object classes:
        obj_runtime_params.object_class_filter = [ sl.OBJECT_CLASS.PERSON]
        return obj_runtime_params
