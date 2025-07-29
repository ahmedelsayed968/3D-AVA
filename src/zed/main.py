from zed.object_detection import ObjectDetection
from zed.body_tracking import BodyTracking
from zed.utils import parse_args
import argparse
from pyzed import sl
from loguru import logger
from threading import Lock, Thread
import os
import sys
from pathlib import Path
import time
from zed.retrieve import ZedRetrieval
from zed.persist import ZedSaver
lock = Lock()
logger.remove(0)
logger.add(sys.stdout, level="TRACE")




def main(opt):
    ROOT_DIR = Path(__file__).parent.parent.parent / ".data" / "experiments"
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    COUNT = len(os.listdir(ROOT_DIR))
    EXP_DIR = ROOT_DIR / f"EXP_{COUNT+1}"
    EXP_DIR.mkdir(exist_ok=True, parents=True)

    logger.info("Running Zed Capture Module.......")
    
    # Initialize camera parameters
    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL_PLUS,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
        camera_resolution=sl.RESOLUTION.HD2K,
        camera_fps = 30
    )

    # Initialize the runtime parameters
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50  # Reasonable confidence threshold (0-100)
    runtime_parameters.texture_confidence_threshold = 100  # Filters weak textures (0-100)
    # runtime_parameters.remove_saturated_areas = True  

    # Parse additional arguments
    parse_args(init, opt=opt)
    
    # Create and open camera
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        logger.error(f"Cannot open the camera: {repr(status)}")
        exit(1)
    
    # Initialize object detection 
    object_detection = None
    if opt.enable_od:
        try:
            object_detection = ObjectDetection(zed=zed)
            logger.info("Object detection enabled")
        except Exception as e:
            logger.error(f"Failed to initialize object detection: {e}")

    # Initialize body tracking - Enable by default for keypoint extraction
    body_tracker = None
    if opt.enable_body_tracking or opt.extract_keypoints:
        try:
            body_tracker = BodyTracking(zed)
            # body_tracker.initialize_body_tracker()
        except Exception as e:
            logger.error(f"Failed to initialize body tracking: {e}")

    # Get camera information
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    calibration_params = camera_infos.camera_configuration.calibration_parameters
    left_cam_params = calibration_params.left_cam

    # Extract intrinsic parameters
    fx = left_cam_params.fx
    fy = left_cam_params.fy
    cx = left_cam_params.cx
    cy = left_cam_params.cy
    logger.debug(f"Camera intrinsics - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    
    # Initialize counters and timing
    counter = 0
    frame_counter = 0
    fps = 30
    save_interval_seconds = 0.05  # Save a frame every 0.05 seconds
    
    # Handle unrealistic FPS values - cap at reasonable maximum
    if fps > 200 or fps <= 0:
        logger.warning(f"Unrealistic FPS reported: {fps}, using time-based capture instead")
        use_time_based = True
        last_save_time = time.time()
    else:
        use_time_based = False
        frames_to_skip = max(1, int(fps * save_interval_seconds))
        logger.info(f"Camera FPS: {fps}, saving every {frames_to_skip} frames ({1/save_interval_seconds} captures per second)")
    
    if use_time_based:
        logger.info(f"Using time-based capture: saving every {save_interval_seconds} seconds")

    # initialize the zed_retrieval 
    zed_retrieval = ZedRetrieval(zed=zed)

    try:
        while True:
            # Grab a new frame
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                
                # Determine if we should process this frame
                should_process = False
                
                if use_time_based:
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval_seconds:
                        should_process = True
                        last_save_time = current_time
                else:
                    if frame_counter % frames_to_skip == 0:
                        should_process = True
                
                if should_process:
                    logger.info(f"Processing frame #{counter} (frame_counter: {frame_counter})")

                    # Retrieve data with proper error handling
                    depth_map = None
                    left_image = None
                    keypoints_data = None
                    mask_data = None
                    
                    with lock:  # Use context manager for lock
                        # Retrieve object detections if enabled
                        if object_detection:
                            all_objects = zed_retrieval.zed_retrieve_object_detections(runtime_params=object_detection.obj_runtime_params,
                                                                                       instance_id=object_detection.obj_param.instance_module_id)
                            if all_objects:
                                logger.debug(f"Detected {len(all_objects)} objects")

                        # Retrieve body tracking data and keypoints
                        if body_tracker and opt.extract_keypoints:
                            bodies = zed_retrieval.zed_retrieve_bodies(tracking_params=body_tracker.runtime_param,
                                                                       instance_module_id=body_tracker.body_param.instance_module_id)
                            if bodies:
                                keypoints_data = ZedRetrieval.extract_keypoints_and_masks(bodies, camera_res)
                                logger.debug(f"Extracted keypoints for {len(keypoints_data['bodies'])} bodies")

                        # Retrieve depth map
                        depth_map = zed_retrieval.zed_retrieve_depth_map(camera_res=camera_res)
                        if depth_map is None:
                            logger.warning(f"Failed to retrieve depth map for frame {counter}")

                        # Retrieve left image
                        left_image = zed_retrieval.zed_retrieve_left_image(camera_res=camera_res)
                        if left_image is None:
                            logger.warning(f"Failed to retrieve left image for frame {counter}")
                        
                        # Retrieve mask if requested
                        if opt.extract_masks:
                            mask_data = zed_retrieval.zed_extract_segmentation_masks(all_objects)
                            if mask_data is None:
                                logger.warning(f"Failed to retrieve mask for frame {counter}")
                            else:
                                logger.info(f"detected_masks {len(mask_data)}")

                                mask_data = mask_data[0]
                    # Save data if requested
                    if opt.save:
                        if depth_map is not None:
                            Thread(target=ZedSaver.save_depth_map, args=(depth_map, EXP_DIR, counter)).start()
                        if left_image is not None:
                            Thread(target=ZedSaver.save_image_from_zed, args=(left_image, EXP_DIR, counter)).start()
                        if keypoints_data is not None and opt.extract_keypoints:
                            Thread(target=ZedSaver.save_keypoints_and_masks, args=(keypoints_data, EXP_DIR, counter)).start()
                        if mask_data is not None and opt.extract_masks:
                            Thread(target=ZedSaver.save_mask, args=(mask_data, EXP_DIR, counter)).start()

                    counter += 1
                    
                frame_counter += 1
                
            elif zed.grab(runtime_parameters) == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                logger.info("End of SVO file reached")
                break
            else:
                logger.error(f"Failed to grab frame: {zed.grab(runtime_parameters)}")
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        logger.info("Closing camera...")
        if body_tracker:
            zed.disable_body_tracking()
        if object_detection:
            zed.disable_object_detection()
        zed.close()
        logger.info("Camera closed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--ip_address', type=str, help='IP Address, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default='')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default='')
    parser.add_argument('--enable_od', action='store_true', help="Enable object detection for filtering human from the environment")
    parser.add_argument('--enable_body_tracking', action='store_true', help="Enable body tracker")
    parser.add_argument('--extract_keypoints', action='store_true', help="Extract and save 2D/3D keypoints")
    parser.add_argument('--extract_masks', action='store_true', help="Extract and save segmentation masks")
    parser.add_argument("--save", action='store_true', help="Save captured data")
    
    opt = parser.parse_args()
    
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit(1)
    
    main(opt=opt)