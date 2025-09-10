from motion_complexity_classification_with_zero_shot_vqa.configs._config import (
    Config,
    ConfigImpl,
)
from motion_complexity_classification_with_zero_shot_vqa.configs._movement_vqa_protocol import (
    MovementVQAConfig,
)
from motion_complexity_classification_with_zero_shot_vqa.configs._object_detection_protocol import (
        ObjectDetectionConfig,
        )
from motion_complexity_classification_with_zero_shot_vqa.configs._bbox_protocol import (
        BBoxConfig,
        )
from motion_complexity_classification_with_zero_shot_vqa.configs._set_seed import (
    set_seed,
)
from functools import cache
from typing import Generic
from dataclasses import dataclass
from collections.abc import Mapping
from ._vlm_protocol import VLMProtocol
from ._config import (
    ConfigImpl,
    MovementVQAConfigImpl,
    ObjectDetectionConfigImpl,
    InferenceState,
    ImageLabelProviderImpl,
    BBoxConfigImpl,
)
from pathlib import Path
from static_error_handler import Ok, Err, Result

__all__ = [
    "get_configs",
    "set_seed",
    "InferenceConfig",
    "MovementVQAConfig",
    "BBoxConfig",
    "ObjectDetectionConfig",
    "VLMProtocol",
]


@cache
def get_configs() -> Mapping[str, tuple[str, ConfigImpl]]:
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
    configs: Mapping[str, tuple[str, ConfigImpl]] = {
        "aloha_sim_insertion_human": (
            "aloha sim insertion human",
            ConfigImpl(
                repo_id="lerobot/aloha_sim_insertion_human",
                episode_index=0,
                image_columns=AlohaColumns,
                object_detection_vlm_model="OpenGVLab/InternVL3-78B",
                object_detection_prompt=(
                    (
                        "test",
                        """
            Describe things in this scene and their spatial relations briefly.
            And then list up them in the following json format strictly:
            { "tag_id": detail description, "tag2_id": detail description, ... }
            note tat tag_id should be unique, it's for internal use to identify objects.
            descriptions should include details on appearance for object detection.
            e.g. { "can 1": blue can, ... }
                    """,
                    ),
                ),
                movement_vqa_vlm_model="OpenGVLab/InternVL3_5-1B",
                movement_vqa_prompt=aloha_prompt(
                    "pick red block by right arm and transfer it to left arm"
                ),
                movement_vqa_window_size=4,
                movement_vqa_step=1,
                bbox_column_name = "observation.images.top",
                movement_vqa_output_file=Path("results/aloha_sim_insertion_human/movement_vqa.json"),
                object_detection_output_file=Path( "results/aloha_sim_insertion_human/object_detection.json"),
                bbox_output_file = Path("results/aloha_sim_insertion_human/bbox.json"),
            ),
        ),
    }
    return configs
