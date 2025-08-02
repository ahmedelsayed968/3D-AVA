from dataclasses import dataclass,astuple
import json
from typing import List,Dict,Union

def read_json(path:str):
  with open(path,mode="r") as fin:
    data = json.load(fin)
  return data

KEYPOINTS_BODY_38: dict[str,int] = {
    "PELIVIS": 0,
    "SPINE_1": 1,
    "SPINE_2": 2,
    "SPINE_3": 3,
    "NECK": 4,
    "NOSE": 5,
    "LEFT_EYE": 6,
    "RIGHT_EYE": 7,
    "LEFT_EAR": 8,
    "RIGHT_EAR": 9,
    "LEFT_CLAVICLE": 10,
    "RIGHT_CLAVICLE": 11,
    "LEFT_SHOULDER": 12,
    "RIGHT_SHOULDER": 13,
    "LEFT_ELBOW": 14,
    "RIGHT_ELBOW": 15,
    "LEFT_WRIST": 16,
    "RIGHT_WRIST": 17,
    "LEFT_HIP": 18,
    "RIGHT_HIP": 19,
    "LEFT_KNEE": 20,
    "RIGHT_KNEE": 21,
    "LEFT_ANKLE": 22,
    "RIGHT_ANKLE": 23,
    "LEFT_BIG_TOE": 24,
    "RIGHT_BIG_TOE": 25,
    "LEFT_SMALL_TOE": 26,
    "RIGHT_SMALL_TOE": 27,
    "LEFT_HEEL": 28,
    "RIGHT_HEEL": 29,
    "LEFT_HAND_THUMB_4": 30,
    "RIGHT_HAND_THUMB_4": 31,
    "LEFT_HAND_INDEX_1": 32,
    "RIGHT_HAND_INDEX_1": 33,
    "LEFT_HAND_MIDDLE_4": 34,
    "RIGHT_HAND_MIDDLE_4": 35,
    "LEFT_HAND_PINKY_1": 36,
    "RIGHT_HAND_PINKY_1": 37,
}

BODY_25_KEYPOINTS_ZED_STYLE: dict[str, int] = {
    "NOSE": 0,
    "NECK": 1,
    "RIGHT_SHOULDER": 2,
    "RIGHT_ELBOW": 3,
    "RIGHT_WRIST": 4,
    "LEFT_SHOULDER": 5,
    "LEFT_ELBOW": 6,
    "LEFT_WRIST": 7,
    "PELIVIS": 8,               # aka MidHip
    "RIGHT_HIP": 9,
    "RIGHT_KNEE": 10,
    "RIGHT_ANKLE": 11,
    "LEFT_HIP": 12,
    "LEFT_KNEE": 13,
    "LEFT_ANKLE": 14,
    "RIGHT_EYE": 15,
    "LEFT_EYE": 16,
    "RIGHT_EAR": 17,
    "LEFT_EAR": 18,
    "LEFT_BIG_TOE": 19,
    "LEFT_SMALL_TOE": 20,
    "LEFT_HEEL": 21,
    "RIGHT_BIG_TOE": 22,
    "RIGHT_SMALL_TOE": 23,
    "RIGHT_HEEL": 24
}

BODY25_FROM_ZED38 = {
   value: KEYPOINTS_BODY_38[key] 
   for key,value in BODY_25_KEYPOINTS_ZED_STYLE.items()
}


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    @classmethod
    def from_zed_format(cls,bbox:List[Dict[str,float]],confidence):
      bbox2d = [
          [point['x'],point['y']]
          for point in bbox
      ]
      tl = bbox2d[0]
      br = bbox2d[2]
      xmin, ymin = tl[0], tl[1]
      xmax, ymax = br[0], br[1]
      return cls(xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,confidence=confidence)

    def compute_area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    
    @classmethod
    def from_path(cls,path:str):
        data = read_json(path)
        bbox = data['bodies'][0]['bounding_box_2d']
        confidence = data['bodies'][0]['confidence']
        return cls.from_zed_format(bbox,confidence)
@dataclass
class Keypoint2D:
    joint_id: int
    x: float
    y: float

@dataclass
class KeyPoints2D:
    points: List[Keypoint2D]
    confidence: float
    @classmethod
    def from_zed_format(cls,keypoints2d:List[Dict[str,Union[int,float]]],confidence:float):
        return cls(
            [
                Keypoint2D(**keypoint)
                for keypoint in keypoints2d
            ],
            confidence=confidence
        )
    @classmethod
    def from_path(cls,path:str):
        data = read_json(path)
        keypoints_raw = data['bodies'][0]['keypoints_2d']
        confidence = data['bodies'][0]['confidence']
        return cls.from_zed_format(keypoints_raw,confidence=confidence)
    def get_body25(self):

      body25 = []
      for bp25_idx in sorted(BODY25_FROM_ZED38.keys()):
          zed_idx = BODY25_FROM_ZED38[bp25_idx]

          zed_keypoint = self.points[zed_idx]
          keypoint = Keypoint2D(joint_id=bp25_idx,x=zed_keypoint.x,y=zed_keypoint.y)
          body25.append(keypoint)

      return body25

@dataclass
class SMPLModel:
    id: int
    Rh: List[List[float]]
    Th: List[List[float]]
    poses: List[List[float]]
    shapes: List[List[float]]
    @classmethod
    def from_easymocap(cls,
                       data):
        return cls(**data)
    @classmethod
    def from_path(cls,path:str):
        data = read_json(path)
        return cls.from_easymocap(data[0])
    