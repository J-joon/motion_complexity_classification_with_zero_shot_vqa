from motion_complexity_classification_with_zero_shot_vqa.configs._config import Config, ConfigImpl
from motion_complexity_classification_with_zero_shot_vqa.configs._movement_vqa_protocol import get_movement_vqa_configs, MovementVQAConfig
from motion_complexity_classification_with_zero_shot_vqa.configs._set_seed import set_seed
from functools import cache
from typing import Generic
from dataclasses import dataclass
from collections.abc import Mapping

__all__ = [
    "get_configs",
    "set_seed",
    "InferenceConfig",
    "MovementVQAConfig",
]


@cache
def get_configs[S, I, O]() -> Mapping[str, tuple[str, Config[S, I, O]]]:
    _configs: Mapping[str, tuple[str, Config[S, I, O]]] = {
        "test": (
            "test",
            ConfigImpl(
                movement_vqa_config = get_movement_vqa_configs()["test"][1],
                ),
            ),
    }
    return _configs
