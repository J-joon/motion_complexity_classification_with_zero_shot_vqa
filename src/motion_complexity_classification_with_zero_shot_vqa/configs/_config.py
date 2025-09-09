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
from more_itertools import windowed, take
from itertools import chain
from dataclasses import dataclass, field
import tyro
from functools import cache, partial
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from vqa_pipeline.vlm import VLM, ImageLabelProvider, InternVL3
from static_error_handler import Ok, Err, Result
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from motion_complexity_classification_with_zero_shot_vqa.configs import (
    _movement_vqa_protocol,
)
from ._movement_vqa_protocol import MovementVQAConfig
from ._object_detection_protocol import ObjectDetectionConfig


# ========
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


# ========


class Config[
    T_MovementVQA_State,
    T_MovementVQA_Input,
    T_MovementVQA_Output,
    T_ObjectDetection_State,
    T_ObjectDetection_Input,
    T_ObjectDetection_Output,
](Protocol):
    @property
    def movement_vqa_config(
        self,
    ) -> MovementVQAConfig[
        T_MovementVQA_State,
        T_MovementVQA_Input,
        T_MovementVQA_Output,
    ]: ...

    @property
    def object_detection_config(
        self,
    ) -> ObjectDetectionConfig[
        T_ObjectDetection_State,
        T_ObjectDetection_Input,
        T_ObjectDetection_Output,
    ]: ...


@dataclass(frozen=True)
class MovementVQAConfigImpl(
    MovementVQAConfig[
        Result[InferenceState, str],
        tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        None,
    ]
):
    _parent: ConfigImpl

    @property
    def initial_state(
        self,
    ) -> Result[InferenceState, str]:
        model = self._parent.vlm_model
        return (
            InternVL3.create(model)
            .inspect(lambda _: print(f"{model} created successfully"))
            .inspect_err(print)
            .map(lambda vlm: InferenceState(vlm, -1, ()))
        )

    @property
    @cache
    def data_stream(
        self,
    ) -> Iterable[tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]]]:
        repo_id = self._parent.repo_id
        episode_index = self._parent.episode_index
        prompt = self._parent.movement_vqa_prompt
        image_columns = self._parent.image_columns
        window_size = self._parent.movement_vqa_window_size
        step = self._parent.movement_vqa_step
        return (
            (list(chain.from_iterable(x for x in win if x is not None)), prompt)
            for win in windowed(
                [
                    ImageLabelProviderImpl.from_frame(
                        frame,
                        image_columns,
                    )
                    for frame in LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
                ],
                window_size,
                step=step,
            )
        )

    def _inference(
        self,
        state: Result[InferenceState, str],
        input_data: tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
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
            return (
                st.vlm.question(*input_data)
                .inspect(print)
                .map(lambda result: update_state(result, st))
            )

        return state.and_then(run)

    @property
    def inference(
        self,
    ) -> Callable[
        [
            Result[InferenceState, str],
            tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        ],
        Result[InferenceState, str],
    ]:
        return self._inference

    @property
    def consume(
        self,
    ) -> Callable[[Result[InferenceState, str]], None]:
        return self._consume

    def _consume(self, state: Result[InferenceState, str]) -> None:
        output_file = self._parent.output_file

        def save(state: InferenceState) -> None:
            with open(output_file, "w") as file:
                json.dump(state.data, file)

        state.inspect(save)


@dataclass(frozen=True)
class ObjectDetectionConfigImpl(
    ObjectDetectionConfig[
        Result[InferenceState, str],
        tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        None,
    ]
):
    _parent: ConfigImpl

    @property
    def initial_state(
        self,
    ) -> Result[InferenceState, str]:
        model = self._parent.object_detection_vlm_model
        return (
            InternVL3.create(model)
            .inspect(lambda _: print(f"{model} created successfully"))
            .inspect_err(print)
            .map(lambda vlm: InferenceState(vlm, -1, ()))
        )

    @property
    @cache
    def data_stream(
        self,
    ) -> Iterable[tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]]]:
        repo_id = self._parent.repo_id
        episode_index = self._parent.episode_index
        prompt = self._parent.object_detection_prompt
        image_columns = self._parent.image_columns
        window_size = 1
        step = 1
        return take(
            1,
            (
                (list(chain.from_iterable(x for x in win if x is not None)), prompt)
                for win in windowed(
                    [
                        ImageLabelProviderImpl.from_frame(
                            frame,
                            image_columns,
                        )
                        for frame in LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
                    ],
                    window_size,
                    step=step,
                )
            ),
        )

    def _inference(
        self,
        state: Result[InferenceState, str],
        input_data: tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
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
            return (
                st.vlm.question(*input_data)
                .inspect(print)
                .map(lambda result: update_state(result, st))
            )

        return state.and_then(run)

    @property
    def inference(
        self,
    ) -> Callable[
        [
            Result[InferenceState, str],
            tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        ],
        Result[InferenceState, str],
    ]:
        return self._inference

    @property
    def consume(
        self,
    ) -> Callable[[Result[InferenceState, str]], None]:
        return self._consume

    def _consume(self, state: Result[InferenceState, str]) -> None:
        output_file = self._parent.object_detection_output_file

        def save(state: InferenceState) -> None:
            with open(output_file, "w") as file:
                json.dump(state.data, file)

        state.inspect(save)


@dataclass(frozen=True)
class ConfigImpl(
    Config[
        Result[InferenceState, str],
        tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        None,
        Result[InferenceState, str],
        tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        None,
    ]
):

    repo_id: str
    episode_index: int
    image_columns: tuple[tuple[str, str], ...]
    object_detection_vlm_model: str
    object_detection_output_file: Path
    object_detection_prompt: tuple[tuple[str, str], ...]
    movement_vqa_vlm_model: str
    movement_vqa_prompt: tuple[tuple[str, str], ...]
    movement_vqa_window_size: int
    movement_vqa_step: int
    movement_vqa_output_file: Path

    @property
    @cache
    def movement_vqa_config(
        self,
    ) -> MovementVQAConfigImpl:
        return MovementVQAConfigImpl(
            _parent=self,
        )

    @property
    @cache
    def object_detection_config(
        self,
    ) -> ObjectDetectionConfigImpl:
        return ObjectDetectionConfigImpl(
            _parent=self,
        )
