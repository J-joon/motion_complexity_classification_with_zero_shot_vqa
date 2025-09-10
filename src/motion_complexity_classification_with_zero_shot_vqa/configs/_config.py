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
from vqa_pipeline.vlm import VLM, ImageLabelProvider, InternVL3, BBoxProvider, GroundingDino, ImageProvider
from vqa_pipeline import Box 
from static_error_handler import Ok, Err, Result
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from motion_complexity_classification_with_zero_shot_vqa.configs import (
    _movement_vqa_protocol,
)
from ._movement_vqa_protocol import MovementVQAConfig
from ._bbox_protocol import BBoxConfig
from ._object_detection_protocol import ObjectDetectionConfig
import re

# for object detection result -> grounding dino query
def extract_json_from_string(text: str) -> Result[dict[str, str], str]:
    """
    Extracts the first JSON object found within a string and returns it as a dictionary.
    
    Args:
        text (str): The input string containing JSON.

    Returns:
        Result[dict[str, str], str]: Parsed JSON dictionary, or an empty dict if no valid JSON is found.
    """
    # Regex to capture JSON block (between curly braces)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Err("Empty")

    json_str = match.group(0)

    try:
        return Ok(json.loads(json_str))
    except Exception as e:
        return Err(str(e))


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
class ImageProviderImpl(ImageProvider):
    image: Image.Image

    @staticmethod
    def from_frame(
            frame: Any, column_name: str
    ) -> ImageProviderImpl:
        return ImageProviderImpl(image = to_pil_image(frame[column_name]),)


@dataclass(frozen=True)
class InferenceState:
    vlm: VLM
    frame_index: int
    data: tuple[tuple[int, tuple[tuple[str, str], ...]], ...]

@dataclass(frozen=True)
class BBoxState:
    bbox_provider: BBoxProvider
    frame_index: int
    data: tuple[tuple[int, tuple[tuple[str, tuple[float, float, float, float]],...]], ...]

# ========


class Config[
    T_MovementVQA_State,
    T_MovementVQA_Input,
    T_MovementVQA_Output,
    T_ObjectDetection_State,
    T_ObjectDetection_Input,
    T_ObjectDetection_Output,
    T_BBox_State,
    T_BBox_Input,
    T_BBox_Output,
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

    @property
    def bbox_config(
        self,
    ) -> BBoxConfig[
        T_BBox_State,
        T_BBox_Input,
        T_BBox_Output,
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
        model = self._parent.movement_vqa_vlm_model
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
        output_file = self._parent.movement_vqa_output_file

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
class BBoxConfigImpl(
    BBoxConfig[
        Result[BBoxState, str],
        tuple[ImageProviderImpl, str],
        None,
    ]
):
    _parent: ConfigImpl

    @property
    def initial_state(
        self,
    ) -> Result[BBoxState, str]:
        try:
            grounding_dino = GroundingDino()
            bbox_state = BBoxState(
                    bbox_provider = GroundingDino(text_threshold=0.3),
                    frame_index = -1,
                    data = (),
                    )
            return Ok(bbox_state)
        except Exception as e:
            return Err(str(e))

    @property
    @cache
    def query(
            self,
            ) -> Result[str, str]:
        with open(
        self._parent.object_detection_output_file,
        "r",
        ) as file:
            data = json.load(file)
        content = data[0][1][0][1]
        return extract_json_from_string(content).map(
                lambda item_dict: ".".join(( item_dict[key] for key in item_dict))
                )

    @property
    @cache
    def data_stream(
        self,
    ) -> Iterable[tuple[ImageProviderImpl, str]]:
        repo_id = self._parent.repo_id
        episode_index = self._parent.episode_index
        query = self.query.unwrap()
        column_name = self._parent.bbox_column_name
        return take(
            1,
            ( (ImageProviderImpl.from_frame( frame, column_name,), query) for frame in LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav") ),
        )

    def _inference(
        self,
        state: Result[BBoxState, str],
        input_data: tuple[ImageProviderImpl, str],
    ) -> Result[BBoxState, str]:
        image, query = input_data
        def update_state(
            result: tuple[Box, ...], state: BBoxState
        ) -> BBoxState:
            boxes: tuple[tuple[str, tuple[float,float,float,float,],],...] = tuple( ( box.label if box.label is not None else "none", ( box.minimum.x, box.minimum.y, box.maximum.x, box.maximum.y, ), ) for box in result )
            return BBoxState(
                state.bbox_provider,
                state.frame_index + 1,
                state.data + ((state.frame_index + 1, boxes),),
            )

        def run(st: BBoxState) -> Result[BBoxState, str]:
            return (
                st.bbox_provider.query(image, query)
                .inspect_err(lambda e: print(f"inference of gdino: {e}"))
                .map(lambda result: update_state(result, st))
            )

        return state.and_then(run)

    @property
    def inference(
        self,
    ) -> Callable[
        [
            Result[BBoxState, str],
            tuple[ImageProviderImpl, str],
        ],
        Result[BBoxState, str],
    ]:
        return self._inference

    @property
    def consume(
        self,
    ) -> Callable[[Result[BBoxState, str]], None]:
        return self._consume

    def _consume(self, state: Result[BBoxState, str]) -> None:
        output_file = self._parent.bbox_output_file

        def save(state: BBoxState) -> None:
            with open(output_file, "w") as file:
                json.dump(state.data, file)

        state.inspect(save).inspect_err(print)

@dataclass(frozen=True)
class ConfigImpl(
    Config[
        Result[InferenceState, str],
        tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        None,
        Result[InferenceState, str],
        tuple[list[ImageLabelProviderImpl], tuple[tuple[str, str], ...]],
        None,
        Result[BBoxState, str],
        tuple[ImageProviderImpl, str],
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
    bbox_column_name: str
    bbox_output_file: Path

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

    @property
    @cache
    def bbox_config(
            self,
            ) -> BBoxConfigImpl:
        return BBoxConfigImpl(
                _parent = self,
                )
