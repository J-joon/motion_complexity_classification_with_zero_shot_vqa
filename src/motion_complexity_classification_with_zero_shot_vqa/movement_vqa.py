from .config import set_seed, get_configs, InferenceConfig
from functools import reduce
from typing import TypeVar
from more_itertools import take
import tyro

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")
T_State = TypeVar("T_State")


def entrypoint() -> int:
    set_seed()
    config = tyro.extras.overridable_config_cli(get_configs())
    main(config)
    return 0


_TEST = True


def main(
    config: InferenceConfig[
        T_State,
        T_Input,
        T_Output,
    ],
) -> None:
    initial_state = config.initial_state
    inference = config.inference
    consume = config.consume
    data_stream = config.data_stream if not _TEST else take(4, config.data_stream)
    result = reduce(inference, data_stream, initial_state)
    consume(result)
    print("done")


if __name__ == "__main__":
    entrypoint()
