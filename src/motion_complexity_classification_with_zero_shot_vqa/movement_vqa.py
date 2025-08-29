from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Protocol,
    Optional,
    runtime_checkable,
    Iterator,
    Iterable,
    Callable,
    TypeVar,
)
from static_error_handler import *
from collections import deque
from scripts.vlm import *
from functools import reduce, partial

T = TypeVar("T")
V = TypeVar("V")

import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer, AutoConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
import json
from tqdm import tqdm
import random
import os
import tyro
from pathlib import Path
from vqa_pipeline.vlm import InternVL3


def write_down(result, output_path):
    with open(output_path, "w") as file:
        json.dump(result, file)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class ConfigProvider(Protocol):
    dataloader: Iterator[tuple[list[Image.Image], str]]
    output_path: Path
    model: str


from collections import deque
from typing import Iterable, Iterator, TypeVar, Tuple

T = TypeVar("T")


class LenGenerator(Iterator[T]):
    def __init__(self, it: Iterable[T], length: int):
        self._it = it
        self._length = length

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __len__(self):
        return self._length


def sliding_window(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    it = iter(iterable)
    window = deque((next(it) for _ in range(n)), maxlen=n)  # initialize first window
    yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def map_then_window(
    iterable: Iterable[T], f: Callable[[T], V], n: int
) -> Iterator[tuple[V, ...]]:
    """Convert each item T -> V, then apply sliding window of size n."""
    return sliding_window((f(x) for x in iterable), n)


@dataclass(frozen=True)
class LeRobotDatasetConfig(ConfigProvider):
    iterable: Iterator[tuple[list[Image.Image], dict[str, str]]]
    output_path: Path
    model: str

    @staticmethod
    def create(
        repo_id: str,
        episode_index: int,
        window_size: int,
        prompts: dict[str, str],
        output_path: Path,
        model: str,
    ) -> Result[Config, str]:
        dataset = LeRobotDataset(repo_id, episodes=[episode_index])
        iterable = map_then_window(
            dataset, lambda frame: to_pil_image(frame["image"]), window_size
        )
        return LeRobotDatasetConfig(
            iterable=LenGenerator(
                (
                    (x, prompts)
                    for x in map_then_window(
                        dataset, lambda frame: to_pil_image(frame["image"]), window_size
                    )
                ),
                len(dataset) - window_size + 1,
            ),
            output_path=output_path,
            model=model,
        )


def vqa(vlm, iterable: Iterable) -> Result[dict[int, dict[str, str]], str]:
    result = dict()
    try:
        for i, (images, prompts) in enumerate(iterable):
            frame_information = vlm.question(images, prompts)
            match frame_information:
                case Ok(value=frame_information):
                    result[i] = frame_information
                case Err(error=e):
                    return Err(e)
            break
        return Ok(result)
    except Exception as e:
        return Err(e)


def main(config: ConfigProvider):
    """
    task = config.get_task()
    images = config.get_images()
    prompts = config.get_prompts()
    output_path = config.get_output_path()
    results = {}
    for frame_index in tqdm(range(len(images) - window_size)):
        ls_pixel_values = [ load_image(images[frame_index + delta], max_num=12).to(torch.bfloat16).cuda() for delta in range(window_size) ]
        pixel_values = torch.cat(ls_pixel_values, dim=0)
        num_patches_list = [ pixel_values.size(0) for pixel_values in ls_pixel_values ]
        prefix = ''.join([f'Frame-{i+1}: <image>\n' for i in range(len(num_patches_list))])
        history = None
        frame_information = dict()
        for key, prompt in prompts.items():
            question = prefix + prompt
            response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=history, return_history=True)
            frame_information[key] = response
        results[frame_index] = frame_information
    with open("result_motion.json", "w", encoding="utf-8") as file:
        json.dump(results, file)
    """
    result = (
        InternVL3.create(config.model)
        .inspect_err(print)
        .and_then(lambda vlm: vqa(vlm, tqdm(config.iterable)))
        .inspect(partial(write_down, output_path=config.output_path))
    )
    print("done")


def entrypoint():
    set_seed(42)
    libero_config = LeRobotDatasetConfig.create(
        "physical-intelligence/libero",
        9,
        16,
        {
            "scene": "Briefly describe the scene. And then descirbe what motion did the robot take to change the scene.",
            "motion": """
            You are the **System 2** of a robot in the scene.  
            The assigned task is: {task}.  

            You have already executed the action, and now you are reviewing the replay of your execution.  
            Your goal is to **analyze and describe the action chunk** performed by the robot across the given 16 consecutive frames.  

            When answering, provide two parts:
            1. **High-level description** – Summarize the overall action and intention, considering not only the given frames but also what may follow after them. Explain the reasoning and intention behind the action.  
            2. **Low-level description** – Provide a strict, detailed motion description limited to the given 16 frames (without speculating beyond them).  

            Make sure your answer reflects both the **intended goal** and the **observed movement**.
            """,
        },
        Path("./ws/4_movement_vqa/result.json"),
        "OpenGVLab/InternVL3-2B",
    )

    def get_size_test_config(size: int) -> LeRobotDatasetConfig:
        return LeRobotDatasetConfig.create(
            "physical-intelligence/libero",
            9,
            16,
            {
                "scene": "Briefly describe the scene. And then describe what motion did the robot take to change the scene."
            },
            Path(f"./ws/vlm_size_test/{size}B.json"),
            f"OpenGVLab/InternVL3-{size}B",
        )

    main(get_size_test_config(1))
    main(get_size_test_config(2))
    main(get_size_test_config(8))
    main(get_size_test_config(9))
    main(get_size_test_config(14))
    main(get_size_test_config(38))
    main(get_size_test_config(78))

# libero_config = Config.create("physical-intelligence/libero", 9, Path("assets/gripper_box_sam2/0009.pt"))
if __name__ == "__main__":
    entrypoint()
