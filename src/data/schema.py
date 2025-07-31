from dataclasses import dataclass,astuple
from typing import List,Dict,Union

from utils.data_utils import read_json
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

    def get_body25(self):
      BODY25_FROM_ZED38 = {
        0: 5,   1: 4,
        2: 13,  3: 15,  4: 17,
        5: 12,  6: 14,  7: 16,
        8: 0,   9: 19,
        10: 21, 11: 23,
        12: 18, 13: 20, 14: 22,

        15: 7, 16: 6, 17: 9, 18: 8,
        19: 24, 20: 26, 21: 28,
        22: 25, 23: 27,
        24: 29
    }

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
    