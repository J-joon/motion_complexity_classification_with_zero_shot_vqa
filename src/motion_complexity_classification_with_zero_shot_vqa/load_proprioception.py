from dataclasses import dataclass
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import json
import tyro

@dataclass(frozen=True)
class Config:
    repo_id: str = tyro.MISSING
    episode_index: int = tyro.MISSING
    output_path: Path = tyro.MISSING
    proprioception_column: str = "state"
    frame_index_column: str = "frame_index"

def get_proprioception(frame, frame_index_column, proprioception_column):
    return (frame[frame_index_column].item(), frame[proprioception_column].tolist())

def save_proprioceptions(proprioceptions, output_path):
    with open(output_path, "w") as file:
        json.dump(proprioceptions, file)

def get_dataset(repo_id, episode_index):
    return LeRobotDataset(repo_id, episodes=[episode_index])

def main(config: Config):
    dataset = get_dataset(config.repo_id, config.episode_index)
    proprioceptions = dict(map(lambda frame: get_proprioception(frame, config.frame_index_column, config.proprioception_column), dataset))
    save_proprioceptions(proprioceptions, config.output_path)

def entrypoint():
    _CONFIGS = {
            "libero": ("libero", Config(repo_id = "physical-intelligence/libero")),
            "aloha_sim_insertion_scripted": ("aloha sim insertion scripted", Config(repo_id="J-joon/sim_insertion_scripted")),
            "aloha_sim_transfer_cube_scripted": ("aloha sim transfer cube scripted", Config(repo_id="J-joon/sim_transfer_cube_scripted")),
            }
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    main(config)

if __name__ == "__main__":
    entrypoint()
