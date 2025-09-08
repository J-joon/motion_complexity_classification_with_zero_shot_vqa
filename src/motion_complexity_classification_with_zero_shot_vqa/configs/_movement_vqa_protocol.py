from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping
from static_error_handler import Ok, Err, Result
import json
from typing import Protocol, TypeVar, runtime_checkable, Iterable, Generic, Any, Callable, TypeAlias
from more_itertools import windowed
from itertools import chain
from dataclasses import dataclass
import tyro
from functools import cache, partial
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from vqa_pipeline.vlm import VLM, ImageLabelProvider, InternVL3
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from . import _movement_vqa_protocol
from ._vlm_protocol import VLMProtocol

@runtime_checkable
class MovementVQAConfig[T_State, T_Input, T_Output](VLMProtocol[T_State, T_Input, T_Output], Protocol):
    pass

