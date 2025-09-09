from __future__ import annotations
import math
from dataclasses import dataclass
import os
import re
import random
import json
from functools import reduce
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer, AutoConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
import tyro
from typing import Protocol, runtime_checkable, Optional, Literal
from pathlib import Path
from motion_complexity_classification_with_zero_shot_vqa.configs import ObjectDetectionConfig, set_seed, get_configs

def main[S, I, O](config: ObjectDetectionConfig[S, I, O])->None:
    initial_state = config.initial_state
    detect_object = config.inference
    consume = config.consume
    data_stream = config.data_stream
    result = reduce(detect_object, data_stream, initial_state)
    consume(result)
    print("done")

def entrypoint()->None:
    set_seed()
    config = tyro.extras.overridable_config_cli(get_configs())
    main(config.object_detection_config)

if __name__=="__main__":
    entrypoint()
