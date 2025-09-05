from __future__ import annotations
from mote_itertools import (
    windowed,
)
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import (
    Protocol,
    Optional,
    runtime_checkable,
    Iterator,
    Iterable,
    Callable,
    TypeAlias,
    TypeVar,
    SupportsIndex,
)
from static_error_handler import *
from collections import deque
from typing import Iterable, Iterator, TypeVar, Tuple
from collections import deque
from scripts.vlm import *
from functools import reduce, partial
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
from vqa_pipeline.vlm import InternVL3, T_Image, ImageLabelProvider
from jaxtyping import PyTree


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


T_InferenceState = TypeVar("T_InferenceState", covariance=True)
T_Input = TypeVar("T_Input", covariance=True)


@runtime_checkable
class InferenceFn(Protocol[T_InferenceState, T_Input]):
    def __call__(
        self, state: T_InferenceState, input_data: T_Input
    ) -> T_InferenceState: ...


@runtime_checkable
class InferenceConfig(Protocol[T_InferenceState, T_Input]):
    initial_state: T_InferenceState
    data_stream: Iterable[T_Input]
    inference: InferenceFn[T_InferenceState, T_Input]


def vqa(
    vlm, iterable: Iterable[tuple[list[T_Image], dict[str, str]]]
) -> Result[dict[int, dict[str, str]], str]:
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


@dataclass(frozen=True)
class AIWorkerData:
    image: Image.Image
    camera_type: str
    frame_index: int

    @static_method
    def from_frame(frame) -> list[AIWorkerData]:
        image_columns = {
            "observation.images.cam_head": "Frame-{frame_index}_HEAD",
            "observation.images.cam_wrist_left": "Frame-{frame_index}_LEFT_WRIST",
            "observation.images.cam_wrist_right": "Frame-{frame_index}_RIGHT_WRIST",
        }
        frame_index = frame["frame_index"].item()
        data = [
            AIWorkerData(
                image=frame[column_name],
                camera_type=image_columns[column_name],
                frame_index=frame_index,
            )
            for column_name in image_columns
        ]
        return data


@dataclass(frozen=True)
class AIWorkerConfig(
    InferenceConfig[
        Result[VLM, str],
        tuple[list[ImageLabelProvider], dict[str, str]],
        Result[dict[str, str], str],
    ]
):
    repo_id: str
    episode_index: int
    step: int = 1
    window_size: int = 4
    prompt: dict[str, str]
    _cached_data_stream: Optional[
        Iterable[tuple[list[ImageLabelProvider], dict[str, str]]]
    ] = None

    @property
    def initial_state(self) -> Result[VLM, str]:
        return InternVL3.create(config.model)

    @property
    def data_stream(self) -> Iterable[tuple[list[ImageLabelProvider], dict[str, str]]]:
        if self._cached_data_stream is None:
            self._cached_data_stream = (
                (sum(win, ()), self.prompt)
                for win in windowed(
                    [
                        AIWorkerData.from_frame(frame)
                        for frame in LeRobotDataset(
                            self.repo_id, episodes=[self.episode_index]
                        )
                    ],
                    self.window_size,
                    self.step,
                )
            )
        return self._cached_data_stream

    def inference(
        self, state: VLM, input_data: tuple[list[ImageLabelProvider], dict[str, str]]
    ) -> VLM:
        vlm = state
        images, prompts = input_data
        result = vlm.question(images, prompts)
        result.inspect(foo)
        return vlm


def entrypoint():
    set_seed(42)
    _CONFIGS = {
        "conveyor": (
            "conveyor",
            AIWorkerConfig.create(
                repo_id="noisyduck/ffw_bg2_rev4_tr_conveyor_250830_06",
                episode_index=0,
                prompt={"ask": "ask"},
                output_path=Path("./test.json"),
                model="OpenGVLab/InternVL3_5-241B-A28B",
            ),
        ),
    }
    result = (
        InternVL3.create(config.model)
        .inspect_err(lambda error: print(f"error to create InternVL3: {error}"))
        .and_then(lambda vlm: vqa(vlm, tqdm(config.iterable)))
        .inspect(partial(write_down, output_path=config.output_path))
        .inspect_err(print)
    )
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    main(config)


def main(config: InferenceConfig):
    initial_state = config.initial_state
    inference = config.inference

    def inference_step(state: T_InferenceState, input_data: T_Data) -> T_InferenceState:
        next_state, inference_output = inference(state, input_data)
        return next_state

    data_stream = config.data_stream
    consume_result = config.consume_result
    result = reduce(inference_step, data_stream, initial_state)
    consume_result(result)
    print("done")


def test_AIWorkerConfig(config: AIWorkerConfig):
    cnt = 0
    for data in config.data_stream:
        print(data)
        cnt += 1
        if cnt == 4:
            break


if __name__ == "__main__":
    conveyor_config = AIWorkerConfig(
        repo_id="noisyduck/ffw_bg2_rev4_tr_conveyor_250830_06",
        episode_index=1,
    )
    test_AIWorkerConfig(conveyor_config)
    # entrypoint()
