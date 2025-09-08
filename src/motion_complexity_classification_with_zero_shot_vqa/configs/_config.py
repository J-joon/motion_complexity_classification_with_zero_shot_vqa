from __future__ import annotations
from typing import (
    Protocol,
    TypeVar,
    runtime_checkable,
    Iterable,
    Generic,
    Any,
    Callable,
    TypeAlias,
)
import json
from more_itertools import windowed
from itertools import chain
from dataclasses import dataclass
import tyro
from functools import cache, partial
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from vqa_pipeline.vlm import VLM, ImageLabelProvider, InternVL3
from static_error_handler import Ok, Err, Result
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from motion_complexity_classification_with_zero_shot_vqa.configs import _movement_vqa_protocol


class Config[
        T_MovmenetVQA_State,
        T_MovementVQA_Input,
        T_MovmentVQA_Output,
        ](Protocol,):
    movement_vqa_config: _movement_vqa_protocol.MovementVQAConfig[
        T_MovmenetVQA_State,
        T_MovementVQA_Input,
        T_MovmentVQA_Output,
    ]


@dataclass(frozen=True)
class ConfigImpl[
    T_MovmenetVQA_State,
    T_MovementVQA_Input,
    T_MovmentVQA_Output,
](Config[
    T_MovmenetVQA_State,
    T_MovementVQA_Input,
    T_MovmentVQA_Output,
    ]):
    movement_vqa_config: _movement_vqa_protocol.MovementVQAConfig[
        T_MovmenetVQA_State,
        T_MovementVQA_Input,
        T_MovmentVQA_Output,
    ]
