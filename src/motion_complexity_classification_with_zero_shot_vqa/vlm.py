from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, runtime_checkable, TypeVar
from static_error_handler import *
import torchvision.transforms as T
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer, AutoConfig
import math
from static_error_handler import *
from vqa_pipeline.vlm import VLM

