from pyzed import sl
from loguru import logger
class BodyTracking:
    def __init__(self,zed:sl.Camera):
        self.body_param = self.initialize_body_tracker()

        # configure detection_model, instance_module_id, etc.
        self.runtime_param = sl.BodyTrackingRuntimeParameters()
        self.runtime_param.detection_confidence_threshold = 40
        
        self.enabled = False
        err = zed.enable_body_tracking(self.body_param)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error("Body tracking is not enabled")
        else:
            self.enabled = True
            logger.success("Body tracking is working")


    def initialize_body_tracker(self):
        body_param = sl.BodyTrackingParameters()
        body_param.instance_module_id = 0
        body_param.enable_tracking = True                # Track people across images flow
        body_param.enable_body_fitting = True            # Smooth skeleton move
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE 
        body_param.body_format =sl.BODY_FORMAT.BODY_38 # Choose the BODY_FORMAT you wish to use
        return body_param