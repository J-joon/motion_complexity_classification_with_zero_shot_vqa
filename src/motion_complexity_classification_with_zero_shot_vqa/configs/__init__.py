from motion_complexity_classification_with_zero_shot_vqa.configs._config import Config, ConfigImpl
from motion_complexity_classification_with_zero_shot_vqa.configs._movement_vqa_protocol import MovementVQAConfig
from motion_complexity_classification_with_zero_shot_vqa.configs._set_seed import set_seed
from functools import cache
from typing import Generic
from dataclasses import dataclass
from collections.abc import Mapping
from ._vlm_protocol import VLMProtocol
from ._config import ConfigImpl, MovementVQAConfigImpl, InferenceState, ImageLabelProviderImpl
from pathlib import Path
from static_error_handler import Ok, Err, Result

__all__ = [
    "get_configs",
    "set_seed",
    "InferenceConfig",
    "MovementVQAConfig",
    "VLMProtocol",
]

def get_aloha_config(
    repo_id: str,
    episode_index: int,
    vlm_model: str,
    movement_vqa_prompt: tuple[tuple[str, str], ...],
    movement_vqa_window_size: int,
    movement_vqa_step: int,
    image_columns: tuple[tuple[str, str], ...],
    output_file: Path,
    ) -> ConfigImpl:
    return ConfigImpl(
            repo_id = repo_id,
            episode_index = episode_index,
            vlm_model = vlm_model,
            movement_vqa_prompt = movement_vqa_prompt,
            movement_vqa_window_size = movement_vqa_window_size,
            movement_vqa_step = movement_vqa_step,
            image_columns = image_columns,
            output_file = output_file,
            )
    

@cache
def get_configs()->Mapping[str, tuple[str, ConfigImpl]]:
    configs: Mapping[str, tuple[str, ConfigImpl]] =  {
            "test": ("test", get_aloha_config(
                "J-joon/sim_inserted_scripted",
                0,
                'OpenGVLab/InternVL3_5-1B',
                (("test", "describe the scene"),),
                4,
                1,
                (("observation.images.top", "observation.images.top"),),
                Path("test_1.json"),
                )),
            }
    return configs
