from sam2.build_sam import build_sam2_video_predictor
import os
import torch
#from scripts.recipe_providers import BoundingboxRecipeProvider
import numpy as np
from torchvision.transforms.functional import to_pil_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from vqa_pipeline.visualisation import *
from pathlib import Path
from dataclasses import dataclass
import tyro
import h5py
import pickle
from typing import Literal


def save_hdf5(filename, data):
    with h5py.File(filename, "w") as f:
        for timestep, objects in data.items():
            ts_group = f.create_group(str(timestep))  # timestep도 str 변환
            for obj_label, state in objects.items():
                obj_group = ts_group.create_group(str(obj_label))  # 반드시 str 변환!

                # bounding_box (None → NaN)
                bbox = state.get("bounding_box")
                if bbox is None:
                    bbox = [np.nan, np.nan, np.nan, np.nan]
                obj_group.create_dataset("bounding_box", data=np.array(bbox))

                # centre_point (None → NaN)
                centre = state.get("centre_point")
                if centre is None:
                    centre = [np.nan, np.nan]
                obj_group.create_dataset("centre_point", data=np.array(centre))


def load_hdf5(filename):
    result = {}
    with h5py.File(filename, "r") as f:
        for timestep in f.keys():
            result[int(timestep)] = {}
            ts_group = f[timestep]
            for obj_label in ts_group.keys():
                obj_group = ts_group[obj_label]

                bbox = obj_group["bounding_box"][:]
                if np.isnan(bbox).all():
                    bbox = None
                else:
                    bbox = bbox.tolist()

                centre = obj_group["centre_point"][:]
                if np.isnan(centre).all():
                    centre = None
                else:
                    centre = centre.tolist()

                # obj_label을 원래 int로 쓰고 싶으면 int 변환
                try:
                    key = int(obj_label)
                except ValueError:
                    key = obj_label

                result[int(timestep)][key] = {
                    "bounding_box": bbox,
                    "centre_point": centre,
                }
    return result


def compute_centroid(mask: np.ndarray):
    """
    Compute the centroid (x, y) of a binary mask.

    Args:
        mask (np.ndarray): 2D array (H, W), values are boolean or 0/1.

    Returns:
        tuple: (x, y) centroid coordinates or None if mask is empty.
    """
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None  # No pixels in the mask
    mean_y, mean_x = coords.mean(axis=0)  # rows = y, cols = x
    return (float(mean_x), float(mean_y))


def mask_to_bbox(mask: np.ndarray):
    """
    Convert a binary SAM mask into a bounding box.

    Args:
        mask (np.ndarray): Boolean or 0/1 array of shape (H, W)

    Returns:
        (x_min, y_min, x_max, y_max): bounding box coordinates
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Find indices where mask is True
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        # No foreground pixels found
        return None

    x_min = x_indices.min().item()
    x_max = x_indices.max().item()
    y_min = y_indices.min().item()
    y_max = y_indices.max().item()

    return (x_min, y_min, x_max, y_max)


@dataclass(frozen=True)
class SamPoint:
    point: Point
    is_inclusive: bool


@dataclass(frozen=True)
class SamPoints:
    label: str
    sam_points: list[SamPoint]


class Config(Protocol):
    repo_id: str
    image_column: str
    episode_index: int
    output_dir: Path
    frames_dir: Path
    sam2_checkpoint: str
    model_cfg: str
    boxes: list[Box]
    points: list[SamPoints]


@dataclass(frozen=True)
class LiberoConfig(Config):
    repo_id: str = "physical-intelligence/libero"
    image_column: str = "image"
    episode_index: int = tyro.MISSING
    output_dir: Path = tyro.MISSING
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    _bounding_box_dir: Path = Path("./libero/2.grounding_dino")
    _points_dir: Path = Path("./libero/sam2_points")

    @property
    def frames_dir(self) -> Path:
        root_dir = Path("./libero/frames")
        return root_dir / f"{self.episode_index}"

    @property
    def boxes(self) -> list[Box]:
        with open(self._bounding_box_dir / f"{self.episode_index:04d}.json", "r") as file:
            boxes_json = json.load(file)
        boxes = [ Box.from_list(box, label) for label, box in boxes_json.items() ]
        return boxes

    @property
    def points(self) -> list[SamPoints]:
        with open(self._points_dir / f"{self.episode_index:04d}.json", "r") as file:
            sam_points_json = json.load(file)
        ls_sam_points = [
            SamPoints(
                label=label,
                sam_points=[
                    SamPoint(
                        point=Point(x=point["x"], y=point["y"]),
                        is_inclusive=point["is_inclusive"],
                    )
                    for point in points
                ],
            )
            for label, points in sam_points_json.items()
        ]
        return ls_sam_points

@dataclass(frozen=True)
class AlohaConfig(Config):
    _task: Literal["sim_insertion_scripted", "sim_transfer_cube_scripted"]
    episode_index: int = tyro.MISSING
    output_dir: Path = tyro.MISSING
    image_column: str = "observation.images.top"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    @property
    def _root_dir(self) -> Path:
        return Path(f"./aloha_{self._task}")

    @property
    def repo_id(self) -> str:
        return f"J-joon/{self._task}"

    @property
    def frames_dir(self) -> Path:
        return self._root_dir / "frames" / f"{self.episode_index}"

    @property
    def boxes(self) -> list[Box]:
        with open(self._root_dir / "2.grounding_dino"/ f"ep_{self.episode_index}.json", "r") as file:
            boxes_json = json.load(file)
        boxes = [ Box.from_list(box, label) for label, box in boxes_json.items() ]
        return boxes

    @property
    def points(self) -> list[SamPoints]:
        points_dir = self._root_dir / "sam2_points"
        with open(points_dir / f"{self.episode_index}.json", "r") as file:
            sam_points_json = json.load(file)
        ls_sam_points = [
            SamPoints(
                label=label,
                sam_points=[
                    SamPoint(
                        point=Point(x=point["x"], y=point["y"]),
                        is_inclusive=point["is_inclusive"],
                    )
                    for point in points
                ],
            )
            for label, points in sam_points_json.items()
        ]
        return ls_sam_points


def set_device() -> torch.cuda.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def set_torch_by_device(device: torch.cuda.device):
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


def get_frame_names(video_dir: Path) -> list[str]:
    frame_names = [
        p for p in video_dir.iterdir() if p.suffix in {".jpg", ".jpeg", ".JPG", ".JPEG"}
    ]
    frame_names.sort(key=lambda p: int(p.stem))
    return frame_names


def main(config: Config):
    repo_id = config.repo_id
    image_column = config.image_column
    episode_index = config.episode_index
    output_dir = config.output_dir
    video_dir = config.frames_dir
    device = set_device()
    print(f"using device: {device}")
    set_torch_by_device(device)
    sam2_checkpoint = config.sam2_checkpoint
    model_cfg = config.model_cfg

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    frame_names = get_frame_names(video_dir)
    points = config.points
    boxes = config.boxes

    inference_state = predictor.init_state(video_path=str(video_dir))
    predictor.reset_state(inference_state)
    id2label = dict()
    ann_frame_idx = 0
    for ann_obj_id, box in enumerate(boxes):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box.to_numpy(),
        )
        id2label[ann_obj_id] = box.label
    for ann_obj_id, sam_points in enumerate(points, start = ann_obj_id + 1):
        points, labels = list(), list()
        for sam_point in sam_points.sam_points:
            points.append(sam_point.point.to_list())
            labels.append(1 if sam_point.is_inclusive else 0)

        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        id2label[ann_obj_id] = sam_points.label
    
    video_information = dict()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        frame_information = dict()
        for out_obj_idx, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[out_obj_idx] > 0.0).cpu().numpy().squeeze()
            point = compute_centroid(mask)
            box = mask_to_bbox(mask)
            object_information = {"mask": mask, "point": point, "box": box}
            frame_information[out_obj_id] = object_information
        video_information[out_frame_idx] = frame_information

    def frame_information2recipe(out_frame_idx) -> Recipe:
        frame_information = video_information[out_frame_idx]
        masks, points, boxes = list(), list(), list()
        for out_obj_id, object_information in frame_information.items():
            masks.append(Mask(mask=object_information["mask"].tolist()))
            point = object_information["point"]
            box = object_information["box"]
            if point is not None:
                points.append(Point(x=point[0], y=point[1]))
            if box is not None:
                boxes.append(
                    Box(
                        minimum=Point(x=box[0], y=box[1]),
                        maximum=Point(x=box[2], y=box[3]),
                    )
                )
        return Recipe(points=points, boxes=boxes, masks=masks)

    recipes = list(map(frame_information2recipe, video_information))
    images = [
        to_pil_image(frame[image_column])
        for frame in LeRobotDataset(repo_id, episodes=[episode_index])
    ]

    annotated_images = [
        draw_recipe(image, recipe) for image, recipe in zip(images, recipes)
    ]
    pil_images_to_video(annotated_images, output_dir / f"video_ep{episode_index}.mp4")

    with open(output_dir / f"object_states_ep{episode_index}.pkl", "wb") as f:
        pickle.dump(video_information, f)

    print("done")


def entrypoint():
    _Configs = {
            "libero": ("libero", LiberoConfig()),
            "aloha_sim_insertion_scripted": ("aloha sim insertion scripted", AlohaConfig(_task="sim_insertion_scripted")),
            "aloha_sim_transfer_cube_scripted": ("aloha sim transfer cube scripted", AlohaConfig(_task="sim_transfer_cube_scripted")),
            }
    config = tyro.extras.overridable_config_cli(_Configs)
    main(config)

if __name__ == "__main__":
    entrypoint()
