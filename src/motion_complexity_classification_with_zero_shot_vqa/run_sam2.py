from sam2.build_sam import build_sam2_video_predictor
import os
import torch
from scripts.recipe_providers import BoundingboxRecipeProvider
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
                    "centre_point": centre
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
class Config:
    episode_index: int = 9
    output_dir: Path = Path("./ws/3.sam2")

def main(config: Config):
    episode_index = config.episode_index
    output_dir = config.output_dir
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

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


    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    video_dir = f"/home/work/.jinupahk/UROP/jaejoon/libero_pipeline/frames/{episode_index:04d}"

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    boundingbox_recipe_provider = BoundingboxRecipeProvider()

    with open("./ws/2.grounding_dino/result.json", "r") as file:
        boxes = json.load(file)

    id2label = { object_id: label for object_id, label in enumerate(boxes) }
    print(f"labels: {id2label}")
    boxes = { object_id: np.array(boxes[label], dtype=np.float32) for object_id, label in id2label.items() }
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    ann_frame_idx = 0
    for ann_obj_id, box in boxes.items():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )
    ann_obj_id += 1
    id2label[ann_obj_id] = "gripper"

    points = np.array([[128, 86], [123, 22], [121, 51]], dtype=np.float32)
    labels = np.array([0, 0, 1], np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points = points,
        labels = labels,
    )

    video_information = dict()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame_information = dict()
        for out_obj_idx, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[out_obj_idx] > 0.0).cpu().numpy().squeeze()
            point = compute_centroid(mask)
            box = mask_to_bbox(mask)
            object_information = { "mask": mask, "point": point, "box": box }
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
                boxes.append(Box(minimum=Point(x=box[0], y=box[1]), maximum=Point(x=box[2], y=box[3])))
        return Recipe(points=points, boxes=boxes, masks=masks)

    recipes = list(map(frame_information2recipe, video_information))
    images = [ to_pil_image(frame["image"]) for frame in  LeRobotDataset("physical-intelligence/libero", episodes = [ episode_index ]) ]

    annotated_images = [ draw_recipe(image, recipe) for image, recipe in zip(images, recipes) ]
    pil_images_to_video(annotated_images, output_dir / f"result.mp4")

    #save_hdf5("object_states.h5", video_information)
    with open("./object_states.pkl", "wb") as f:
        pickle.dump(video_information, f)

    print("done")

def entrypoint():
    config = tyro.cli(Config)
    main(config)

if __name__=="__main__":
    entrypoint()
