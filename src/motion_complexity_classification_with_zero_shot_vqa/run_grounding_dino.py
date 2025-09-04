import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from torchvision.transforms.functional import to_pil_image
from typing import Literal, Protocol, Iterator
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torchvision.transforms.functional as F
import h5py
import json
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from static_error_handler import *
from vqa_pipeline.visualisation import *
import random
import numpy as np
import os

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def split_text_by_tokens(text, tokenizer, max_tokens=128):
    """Split text into chunks of max_tokens using model tokenizer"""
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

@dataclass(frozen=True)
class LiberoData:
    image: torch.Tensor
    wrist_image: torch.Tensor
    state: torch.Tensor
    actions: torch.Tensor
    timestamp: torch.Tensor
    frame_index: torch.Tensor
    episode_index: torch.Tensor
    index: torch.Tensor
    task_index: torch.Tensor
    task: str

@dataclass
class Data:
    image: Image.Image
    query: str

class DataLoader(Protocol):
    def __next__(self) -> Data: ...
    def __iter__(self) -> Iterator[Data]: ...

class ConfigProvider(Protocol):
    def get_model(self) -> Result[str, str]: ...
    def get_device(self) -> Result[str, str]: ...
    def get_output_dir(self) -> Result[Path, str]: ...
    def get_dataloader(self) -> DataLoader: ...

def is_rank_valid(rank: int) -> Result[bool, str]:
    try:
        num_gpus = torch.cuda.device_count()
        if rank < num_gpus:
            return Ok(True)
        else:
            return Ok(False)
    except Exception as e:
        return Err(e)

@dataclass(frozen=True)
class Config(ConfigProvider):
    episode_range: tuple[int, int]
    output_dir: Path
    query_list_path: Path
    repo_id: str
    device: Literal["cuda", "cpu"] = "cuda"
    rank: int = 0
    model_id: Literal["base", "tiny"] = "tiny"
    image_column:str = "image"

    def get_model(self) -> Result[str, str]:
        return Ok(f"IDEA-Research/grounding-dino-{self.model_id}")

    def get_device(self) -> Result[str, str]:
        match self.device:
            case "cuda":
                num_gpus = torch.cuda.device_count()
                rank = self.rank
                return is_rank_valid(rank).and_then(lambda is_valid: Ok(f"cuda:{rank}") if is_valid else Err(f"rank: {rank} out of range"))
            case "cpu":
                return Ok("cpu")
    def get_output_dir(self) -> Result[str, str]:
        return Ok(self.output_dir)

    def get_dataloader(self) -> DataLoader:
        image_column = self.image_column
        class LiberoDataLoader(DataLoader):
            dataset: LeRobotDataset
            query_list: list[str]
            dataset_iterator: Iterator[LiberoData]
            query_list_iterator: Iterator[str]
            idx: int
            def __init__(self, dataset: LeRobotDataset, query_list: list[str], batch_size: int = 1):
                self.dataset = dataset
                self.query_list = query_list
            def __next__(self) -> Data:
                frame = next(iter(self.dataset))
                query = next(iter(query_list))
                image = to_pil_image(frame[image_column])
                self.idx += 1
                return Data(image=image, query=query)
            def __iter__(self) -> Iterator[Data]:
                self.idx = 0
                self.dataset_iterator = iter(self.dataset)
                self.query_iterator = iter(self.query_list)
                return self
        dataset = LeRobotDataset(self.repo_id, episodes=list(range(self.episode_range[0], self.episode_range[1])))
        with open(self.query_list_path, "r") as query_list_file:
            query_list = json.load(query_list_file)

        my_list = [ "" for _ in range(302) ]
        my_list[0] = ".".join([f"{value}" for key, value in query_list["dict"].items()])
        query_list = my_list

        return LiberoDataLoader(dataset = dataset, query_list = query_list, batch_size = 1)


libero_config = Config(
        episode_range = (9, 10),
        output_dir = Path("./ws/2.grounding_dino"),
        query_list_path = Path("./ws/1.object_detection/result.json"),
        repo_id = "physical-intelligence/libero",
        device = "cuda",
        rank = 0,
        model_id = "base"
        )

aloha_config = Config(
        episode_range = (0, 1),
        output_dir = Path("./aloha/2.grounding_dino"),
        query_list_path = Path("./aloha/1.object_detection/result.json"),
        repo_id = "J-joon/sim_insertion_scripted",
        device = "cuda",
        rank = 0,
        model_id = "base",
        image_column = "observation.images.top"
        )


def main(config: ConfigProvider):
    set_seed(42)
    to_visualise = True
    # === CONFIGURATION ===
    model_id = config.get_model().unwrap()
    device = config.get_device().unwrap()
    output_dir = config.get_output_dir().unwrap()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_tokens = 128  # Adjust to 256 if you want full model capacity
    dataloader = config.get_dataloader()

    # === LOAD MODELS ===
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    data = next(iter(dataloader))
    image, query = data.image, data.query
    print(query)
    inputs = processor(
        images=image, text=query, return_tensors="pt" 
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.4,
        target_sizes=[image.size[::-1]],
    )
    result = results[0]
    if to_visualise:
        boxes = [ Box(minimum=Point(x=box[0], y=box[1]), maximum=Point(x=box[2], y=box[3]), label=label) for box, label in zip(result["boxes"], result["labels"]) ]
        recipe = Recipe(boxes=boxes)
        image_with_boxes = draw_recipe(image, recipe)
        image_with_boxes.save(output_dir / "result.png")
    output = { label: box.tolist() for box, label in zip(result["boxes"], result["labels"]) }
    with open(output_dir / "result.json", "w") as file:
        json.dump(output, file)
def entrypoint():
    _CONFIGS = {
            "libero": ("libero", libero_config),
            "aloha": ("aloha", aloha_config),
            }
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    main(config)
if __name__=="__main__":
    entrypoint()
