from .config import set_seed, get_configs
from functools import reduce
import tyro

def entrypoint():
    set_seed()
    config = tyro.extras.overridable_config_cli(get_configs())
    main(config)

_TEST = False
def main(config: InferenceConfig):
    initial_state = config.initial_state
    inference = config.inference
    consume = config.consume
    data_stream = config.data_stream if not _TEST else take(4, config.data_stream)
    result = reduce(inference, data_stream, initial_state)
    consume(result)
    print("done")


if __name__ == "__main__":
    entrypoint()
