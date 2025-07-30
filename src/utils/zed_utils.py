from pyzed import sl
from loguru import logger
import sys
import numpy as np
from plyfile import PlyData,PlyElement
from PIL import Image
import os
from pathlib import Path
from zed.body_tracking import BodyTracking
import json
# logger.remove(0)
# logger.add(sys.stdout,level="TRACE")

def parse_args(init,opt):
    _, file_extension = os.path.splitext(opt.input_svo_file)

    if len(opt.input_svo_file)>0 and file_extension in ['.svo','.svo2']:
        init.set_from_svo_file(opt.input_svo_file)
        logger.info(" Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            logger.info(" Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            logger.info(" Using Stream input, IP : ",ip_str)
        else :
            logger.info("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        logger.info(" Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        logger.info(" Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        logger.info(" Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        logger.info(" Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        logger.info(" Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        logger.info(" Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        logger.info(" No valid resolution entered. Using default")
    else : 
        logger.info(" Using default resolution")


