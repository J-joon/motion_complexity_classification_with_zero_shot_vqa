from functools import reduce
from more_itertools import take
from motion_complexity_classification_with_zero_shot_vqa.configs import MovementVQAConfig, set_seed, get_configs
import tyro

# Input Definition
_TEST = False


def main[S, I, O](config: MovementVQAConfig[S, I, O]) -> None:
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
