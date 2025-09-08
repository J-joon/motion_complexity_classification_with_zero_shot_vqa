from .configs import set_seed, get_configs, Config
from functools import reduce
from more_itertools import take
import tyro

# Input Definition
_TEST = True


def main(config: Config) -> None:
    initial_state = config.initial_state
    inference = config.inference
    consume = config.consume
    data_stream = config.data_stream if not _TEST else take(4, config.data_stream)
    result = reduce(inference, data_stream, initial_state)
    consume(result)
    print("done")


def entrypoint() -> int:
    set_seed()
    config = tyro.extras.overridable_config_cli(get_configs())
    print(config)
    main(config.movement_vqa_config)
    return 0

if __name__ == "__main__":
    entrypoint()
