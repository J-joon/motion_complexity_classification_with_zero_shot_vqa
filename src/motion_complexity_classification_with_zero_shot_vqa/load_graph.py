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
print(id2label)

with open("./object_states.pkl", "rb") as file:
    object_states = pickle.load(file)

without_mask = dict()
for frame_index, frame_information in object_states.items():
    new_frame_information = dict()
    for object_index, object_information in frame_information.items():
        new_frame_information[object_index] = object_information["point"]
    without_mask[frame_index] = new_frame_information

with open("./object_graph.json", "w") as file:
    json.dump(without_mask, file)
