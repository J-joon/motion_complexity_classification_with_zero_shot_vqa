from __future__ import annotations
from .configs import set_seed, get_configs
import math
from dataclasses import dataclass
import os
import re
import random
import json

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer, AutoConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from tqdm import tqdm
import tyro
from typing import Protocol, runtime_checkable, Optional, Literal
from pathlib import Path

def main(config: ConfigProvider):
    initial_state = config.initial_state
    detect_object = config.detect_object
    consume = config.consume
    data_stream = config.data_stream
    result = reduce(inference, data_stream, initial_state)
    consume(result)
    print("done")

def entrypoint():
    set_seed()
    config = tyro.extras.overridable_config_cli(get_configs())
    main(config)

if __name__=="__main__":
    entrypoint()
