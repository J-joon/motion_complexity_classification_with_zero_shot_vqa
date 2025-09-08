from typing import Protocol, runtime_checkable
from ._vlm_protocol import VLMProtocol

@runtime_checkable
class ObjectDetectionConfig[T_State, T_Input, T_Output](VLMProtocol[T_State, T_Input, T_Output], Protocol):
    pass
