from motion_complexity_classification_with_zero_shot_vqa.configs._vlm_protocol import VLMProtocol

from typing import Protocol, runtime_checkable

@runtime_checkable
class BBoxConfig[T_State, T_Input, T_Output](VLMProtocol[T_State, T_Input, T_Output], Protocol):
    pass
