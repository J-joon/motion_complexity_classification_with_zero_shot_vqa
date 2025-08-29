import numpy as np
import h5py
import pickle
import json
from dataclasses import dataclass
from typing import runtime_checkable, TypeAlias, Optional
from numpy.typing import NDArray

with open("./ws/2.grounding_dino/result.json", "r") as file:
    boxes = json.load(file)
    id2label = { object_id: label for object_id, label in enumerate(boxes) }

with open("./object_states.pkl", "rb") as file:
    object_states = pickle.load(file)

@dataclass(frozen=True)
class ObjectState:
    mask: NDArray[np.bool]
    point: Point

def frame_information2point(frame_information: dict[int, dict) -> Point:

points = dict(map(lambda frame_index, frame_information: (frame_information, frame_information2point(frame_information)), object_states))
print(points)
"""
print(object_states[0][0]['box'])
