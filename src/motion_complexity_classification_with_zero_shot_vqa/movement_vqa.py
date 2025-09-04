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
from itertools import islice
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


"""
def sliding_window(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    it = iter(iterable)
    window = deque((next(it) for _ in range(n)), maxlen=n)  # initialize first window
    yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)
"""

T = TypeVar("T")


def sliding_window(
    iterable: Iterable[T], n: int, step: int = 1
) -> Iterator[tuple[T, ...]]:
    """
    Yield size-n windows over `iterable`, moving the start by `step` each time.
    Stops when fewer than `step` new items remain.

    Example:
      list(sliding_window(range(10), 3, step=2))
      -> [(0,1,2), (2,3,4), (4,5,6), (6,7,8)]
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if step <= 0:
        raise ValueError("step must be >= 1")

    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) < n:
        return  # not enough to form the first window

    yield tuple(window)

    while True:
        chunk = list(islice(it, step))
        if len(chunk) < step:
            return  # can't advance by a full step -> stop
        window.extend(chunk)  # drops `step` leftmost items automatically
        yield tuple(window)


def map_then_window(
    iterable: Iterable[T], f: Callable[[T], V], n: int, step: int
) -> Iterator[tuple[V, ...]]:
    """Convert each item T -> V, then apply sliding window of size n."""
    return sliding_window((f(x) for x in iterable), n, step)


@dataclass(frozen=True)
class LeRobotDatasetConfig(ConfigProvider):
    iterable: Iterator[tuple[list[Image.Image], dict[str, str]]]
    output_path: Path
    model: str
    image_column: str = "image"

    @staticmethod
    def create(
        repo_id: str,
        episode_index: int,
        window_size: int,
        prompts: dict[str, str],
        output_path: Path,
        model: str,
        step: int,
        image_column: str = "image",
        factor: int = 4
    ) -> Result[Config, str]:
        dataset = LeRobotDataset(repo_id, episodes=[episode_index])
        def get_pil_image(image: Image.Image) -> Image.Image:
            return image.resize((image.width // factor, image.height // factor), Image.BILINEAR)
        iterable = map_then_window(
                dataset, lambda frame: get_pil_image(to_pil_image(frame[image_column])), window_size, step
        )
        return LeRobotDatasetConfig(
            iterable=LenGenerator(
                ((x, prompts) for x in iterable),
                (len(dataset) - window_size + 1) // step,
            ),
            output_path=output_path,
            model=model,
            image_column = image_column
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
        .inspect_err(lambda error: print(f"error to create InternVL3: {error}"))
        .and_then(lambda vlm: vqa(vlm, tqdm(config.iterable)))
        .inspect(partial(write_down, output_path=config.output_path))
        .inspect_err(print)
    )
    print("done")


def entrypoint():
    set_seed(42)
    '''
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
            1. **High-level description** – Summarize the overall action and intention, considering not only the given frames but also what may follow after them.
                Explain the reasoning and intention behind the action.  
            2. **Low-level description** – Provide a strict, detailed motion description limited to the given 16 frames (without speculating beyond them).  

            Make sure your answer reflects both the **intended goal** and the **observed movement**.
            """,
        },
        Path("./ws/4_movement_vqa/result.json"),
        "OpenGVLab/InternVL3-2B",
    )
    '''

    def get_simple_libero_config(
            size: int, output_path: Path, step: int, episode_index: int = 9
    ) -> LeRobotDatasetConfig:
        return LeRobotDatasetConfig.create(
            "physical-intelligence/libero",
            episode_index,
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
            output_path,
            f"OpenGVLab/InternVL3-{size}B",
            step,
        )

    def get_aloha_sim_insertion_scripted(
        size: int, output_path: Path, episode_index: int, step: int
    ) -> LeRobotDatasetConfig:
        return LeRobotDatasetConfig.create(
            repo_id = "J-joon/sim_insertion_scripted",
            episode_index = episode_index,
            window_size = 4,
            prompts = {
                "scene": "Briefly describe the scene. And then descirbe what motion did the robot take to change the scene.",
                "motion": """
                        You are the **System 2** of a bimanual robot in the scene.  

                        The assigned task is: "pick red block by right arm and blue block by left arm, then insert red block into blue block".  
                        You have already executed the action, and now you are reviewing the replay of your execution.  
                        Your goal is to **analyze and describe the action chunk** performed by the robot across the given 16 consecutive frames.  

                        When answering, provide two parts:
                        1. **High-level description** – Summarize the overall action and intention, considering not only the given frames but also what may follow after them. Explain the reasoning and intention behind the action.  
                        2. **Low-level description** – Provide a strict, detailed motion description limited to the given 16 frames (without speculating beyond them).  

                        Make sure your answer reflects both the **intended goal** and the **observed movement**.
                        """,
            },
            output_path = output_path,
            #model = "OpenGVLab/InternVL3_5-241B-A28B",
            #model = f"OpenGVLab/InternVL3-{size}B",
            model = f"OpenGVLab/InternVL3_5-{size}B",
            step = step,
            image_column = "observation.images.top",
            factor = 1,
        )
    """
    get_movement_description_config = partial(
        get_simple_libero_config,
        prompt="Briefly describe the scene. And then describe what motion did the robot take to change the scene.",
    )
    """
    aloha_sim_insertion_scripted_0 = get_aloha_sim_insertion_scripted(
        size=1,
        output_path=Path("./aloha_sim_insertion_scripted/4.movement_vqa/aloha_sim_insertion_scripted_0_78B.json"),
        episode_index=0,
        step=1,
    )
    _CONFIGS = {
            "aloha": ("aloha", aloha_sim_insertion_scripted_0),
            }
    for episode_index in [ 7, 9, 25, 29, 30, 41, 63, 74, 82, 83, 96, 98, 124, 135, 148, 160, 161, 163, 171, 174, 188, 195, 196, 205, 208, 221, 222, 223, 237, 246, 250, 256, 265, 266, 275, 281, 286, 289, 293, 297, 318, 331, 373]:
        _CONFIGS[f"libero_{episode_index}"] = (f"libero {episode_index}", get_simple_libero_config(78, Path(f"./ws/movement_description/78B/{episode_index:04d}.json"), 8, episode_index))
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    main(config)

if __name__ == "__main__":
    entrypoint()
