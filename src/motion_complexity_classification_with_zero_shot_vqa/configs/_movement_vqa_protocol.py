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


__all__ = ["MovementVQAConfig", "MovementVQAConfigGenerics", "T_Input", "T_Output", "T_State"]

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")
T_State = TypeVar("T_State")

@runtime_checkable
class MovementVQAConfig[T_State, T_Input, T_Output](Protocol):
    @property
    def initial_state(self,) -> T_State: ...
    @property
    def data_stream(self,) -> Iterable[T_Input]: ...
    @property
    def inference(self,) -> Callable[[T_State, T_Input], T_State]: ...
    @property
    def consume(self,) -> Callable[[T_State],T_Output]: ...

@dataclass(frozen=True)
class ImageLabelProviderImpl(ImageLabelProvider):
    image: Image.Image
    camera_type: str
    frame_index: int

    @staticmethod
    def from_frame(
            frame: Any, image_columns: tuple[tuple[str, str], ...]
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
    MovementVQAConfig[
        Result[InferenceState, str],
        tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]],
        None
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
            (list(chain.from_iterable(x for x in win if x is not None)), self.prompt)
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
                step = self.step,
            )
        )

    @property
    def data_stream(
        self,
    ) -> Iterable[tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]]]:
        return self._load_data_stream()

    def _inference(
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

        def run(st: InferenceState) -> Result[InferenceState, str]:
            return st.vlm.question(*input_data).inspect(print).map(
                    lambda result: update_state(result, st)
                    )
        return state.and_then(run)

    @property
    def inference(self,) -> Callable[[Result[InferenceState, str], tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]]], Result[InferenceState, str]]:
        return self._inference

    @property
    def consume(self,) ->  Callable[[Result[InferenceState, str]], None]:
        return self._consume

    def _consume(self, state: Result[InferenceState, str]) -> None:
        def save(state: InferenceState) -> None:
            with open(self.output_file, "w") as file:
                json.dump(state.data, file)

        state.inspect(save)

"""
def get_movement_vqa_configs[S, I, O]() -> Mapping[
            str, tuple[str, MovementVQAConfig[Result[InferenceState, str], tuple[list[ImageLabelProvider], tuple[tuple[str, str], ...]], None]]
            ]:
"""
@cache
def get_movement_vqa_configs[S, I, O]() -> Mapping[str, tuple[str, MovementVQAConfig[S, I, O]]]:
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
                prompt=libero_prompt(
                    "put both the alphabet soup and the cream cheese box in the basket"
                ),
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
                prompt=ai_worker_prompt(
                    "Pick up items from the conveyor belt, scan their barcodes, and put them into the basket."
                ),
                image_columns=AIWorkerColumns,
                output_file=Path("./test_conveyor.json"),
            ),
        ),
        "test": (
            "test",
            LeRobotConfig(
                repo_id="physical-intelligence/libero",
                episode_index=0,
                model=test_model,
                prompt=libero_prompt("test"),
                image_columns=LiberoColumns,
                output_file=Path("./test.json"),
            ),
        ),
    }
    return _CONFIGS
