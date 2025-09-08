from __future__ import annotations
from more_itertools import (
    windowed,
    take,
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
from functools import reduce, partial, cache
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
from vqa_pipeline.vlm import VLM, InternVL3
from vqa_pipeline.vlm.vlms import ImageLabelProvider


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


T_InferenceState = TypeVar("T_InferenceState", covariant=True)
T_Input = TypeVar("T_Input", covariant=True)
T_Output = TypeVar("T_Output", covariant=True)


@runtime_checkable
class InferenceFn(Protocol[T_InferenceState, T_Input]):
    def __call__(
        self, state: T_InferenceState, input_data: T_Input
    ) -> T_InferenceState: ...


@runtime_checkable
class ConsumeFn(Protocol[T_InferenceState, T_Output]):
    def __call__(self, state: T_InferenceState) -> T_Output: ...


@runtime_checkable
class InferenceConfig(Protocol[T_InferenceState, T_Input]):
    initial_state: T_InferenceState
    data_stream: Iterable[T_Input]
    inference: InferenceFn[T_InferenceState, T_Input]
    consume: ConsumeFn[T_InferenceState, T_Output]


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
class ImageLabelProviderImpl(ImageLabelProvider):
    image: Image.Image
    camera_type: str
    frame_index: int

    @staticmethod
    def from_frame(
        frame, image_columns: tuple[tuple[str, str], ...]
    ) -> list[ImageLabelProviderImpl]:
        frame_index = frame["frame_index"].item()
        data = [
            ImageLabelProviderImpl(
                image=to_pil_image(frame[column_name]),
                camera_type=label.format(frame_index=frame_index),
                frame_index=frame_index,
            )
            for column_name, label in image_columns
        ]
        return data


@dataclass(frozen=True)
class InferenceState:
    vlm: VLM
    frame_index: int
    data: tuple[tuple[int, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True)
class LeRobotConfig(
    InferenceConfig[
        Result[InternVL3, str],
        tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]],
    ]
):
    repo_id: str
    episode_index: int
    model: str
    prompt: tuple[tuple[str, str], ...]
    output_file: Path
    image_columns: tuple[tuple[str, str], ...]
    step: int = 1
    window_size: int = 4

    @property
    def initial_state(self) -> Result[InferenceState, str]:
        return (
            InternVL3.create(self.model)
            .inspect(lambda _: print(f"{self.model} created successfully"))
            .inspect_err(print)
            .map(lambda vlm: InferenceState(vlm, -1, ()))
        )

    @cache
    def _load_data_stream(
        self,
    ) -> Iterable[tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]]]:
        return (
            (sum(win, []), self.prompt)
            for win in windowed(
                [
                    ImageLabelProviderImpl.from_frame(
                        frame,
                        self.image_columns,
                    )
                    for frame in LeRobotDataset(
                        self.repo_id, episodes=[self.episode_index]
                    )
                ],
                self.window_size,
                self.step,
            )
        )

    @property
    def data_stream(
        self,
    ) -> Iterable[tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]]]:
        return self._load_data_stream()

    def inference(
        self,
        state: Result[InferenceState, str],
        input_data: tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]],
    ) -> Result[InferenceState, str]:
        def update_state(
            result: tuple[tuple[str, str], ...], state: InferenceState
        ) -> InferenceState:
            return InferenceState(
                state.vlm,
                state.frame_index + 1,
                state.data + ((state.frame_index + 1, result),),
            )

        def handle_state(
            state: Result[InferenceState, str],
        ) -> Result[InferenceState, str]:
            return (
                state.vlm.question(*input_data)
                .inspect(print)
                .map(partial(update_state, state=state))
            )

        return state.and_then(handle_state)

    def consume(self, state: Result[InferenceState, str]):
        def save(state: InferenceState):
            with open(self.output_file, "w") as file:
                json.dump(state.data, file)

        state.inspect(save)


def entrypoint():
    set_seed(42)
    AIWorkerColumns = (
        (
            "observation.images.cam_head",
            "Frame-{frame_index}_HEAD: <image>",
        ),
        (
            "observation.images.cam_wrist_left",
            "Frame-{frame_index}_LEFT_WRIST: <image>",
        ),
        (
            "observation.images.cam_wrist_right",
            "Frame-{frame_index}_RIGHT_WRIST: <image>",
        ),
    )
    AlohaColumns = (
        (
            "observation.images.top",
            "Frame-{frame_index}: <image>",
        ),
    )
    LiberoColumns = (
        (
            "image",
            "Frame-{frame_index}: <image>",
        ),
    )
    test_prompt = (
        (
            "test",
            "briefly describe the scene",
        ),
    )

    def ai_worker_prompt(task: str) -> tuple[tuple[str, str], ...]:
        return (
            (
                "motion",
                f"""
            You are the **System 2** of a bimanual humanoid robot in the scene.
            Briefly describe the scene. And then descirbe what motion did the each arm take to change the scene.

            The assigned task is: "{task}".
            You have already executed the action, and now you are reviewing the replay of your execution.
            Your goal is to **analyze and describe the action chunk** performed by the robot across the given 4 consecutive frames, each of which consists of head camera, left wrist camera, and right wrist camera.

            Provide a strict, detailed motion description limited to the given 4 frames (without speculating beyond them).

            Make sure your answer reflects only the **observed movement**.
            output should be the format of following json: {{ 'scene': 'scene description', 'right arm motion': 'motion description', 'left arm motion': 'motion description' }}
            """,
            ),
        )
    def libero_prompt(task: str) -> tuple[tuple[str, str], ...]:
        return (
        (
            "motion",
            f"""
        You are the **System 2** of a bimanual robot in the scene.
        Briefly describe the scene. And then descirbe what motion did the each arm take to change the scene.

        The assigned task is: "{task}".

        You have already executed the action, and now you are reviewing the replay of your execution.
        Your goal is to **analyze and describe the action chunk** performed by the robot across the given 4 consecutive frames.

        Provide a strict, detailed motion description limited to the given 4 frames (without speculating beyond them).

        Make sure your answer reflects only the **observed movement**.

        output should be the format of following json: {{ 'scene': 'scene description', 'right arm motion': 'motion description', 'left arm motion': 'motion description' }}
        """,
        ),
    )
    def aloha_prompt(task: str) -> tuple[tuple[str, str], ...]:
        return (
            (
                "motion",
                f"""
            You are the **System 2** of a bimanual robot in the scene.
            Briefly describe the scene. And then descirbe what motion did the each arm take to change the scene.

            The assigned task is: "{task}".
            You have already executed the action, and now you are reviewing the replay of your execution.
            Your goal is to **analyze and describe the action chunk** performed by the robot across the given 4 consecutive frames.

            Provide a strict, detailed motion description limited to the given 4 frames (without speculating beyond them).

            Make sure your answer reflects only the **observed movement**.
            output should be the format of following json: {{ 'scene': 'scene description', 'right arm motion': 'motion description', 'left arm motion': 'motion description' }}
            """,
            ),
        )

    test_model = "OpenGVLab/InternVL3_5-1B"
    _CONFIGS = {
        "aloha-sim-insertion-scripted": (
            "aloha sim insertion scripted",
            LeRobotConfig(
                repo_id="J-joon/sim_insertion_scripted",
                episode_index=0,
                model=test_model,
                prompt=aloha_prompt(
                    "pick red block by right arm and blue block by left arm, then insert red block into blue block"
                    ),
                image_columns=AlohaColumns,
                output_file=Path("./test_sim_insertion_scripted.json"),
            ),
        ),
        "aloha-sim-transfer-cube-scripted": (
            "aloha sim insertion scripted",
            LeRobotConfig(
                repo_id="J-joon/sim_transfer_cube_scripted",
                episode_index=0,
                model=test_model,
                prompt=aloha_prompt(
                    "pick red block by right arm and transfer it to left arm"
                    ),
                image_columns=AlohaColumns,
                output_file=Path("./test_sim_transfer_cube_scripted.json"),
            ),
        ),
        "libero": (
            "aloha sim insertion scripted",
            LeRobotConfig(
                repo_id="physical-intelligence/libero",
                episode_index=7,
                model=test_model,
                prompt=test_prompt,
                image_columns=LiberoColumns,
                output_file=Path("./test_libero_7.json"),
            ),
        ),
        "conveyor": (
            "conveyor",
            LeRobotConfig(
                repo_id="noisyduck/ffw_bg2_rev4_tr_conveyor_250830_06",
                episode_index=0,
                model=test_model,
                prompt=ai_worker_prompt("Pick up items from the conveyor belt, scan their barcodes, and put them into the basket."),
                image_columns=AIWorkerColumns,
                output_file=Path("./test_conveyor.json"),
            ),
        ),
    }
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    main(config)

_TEST = False
def main(config: InferenceConfig):
    initial_state = config.initial_state
    inference = config.inference
    consume = config.consume
    data_stream = config.data_stream if not _TEST else take(4, config.data_stream)
    result = reduce(inference, data_stream, initial_state)
    consume(result)
    print("done")


if __name__ == "__main__":
    entrypoint()
