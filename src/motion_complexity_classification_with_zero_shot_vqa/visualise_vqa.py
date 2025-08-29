from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tyro
from dataclasses import dataclass, field
from typing import Protocol
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
from vqa_pipeline.visualisation import *
import json
import cv2

class GetLabel(Protocol):
    def __call__(self, episode_index: int, frame_index: int) -> str: ...

class Exp1Label(GetLabel):
    cache: dict[int, dict[int, str]] = dict()
    def __call__(self, episode_index: int, frame_index: int) -> str:
        if episode_index not in self.cache:
            label_path = Path(f"./assets/vqa/exp1_{episode_index:04d}.json")
            with open(label_path, 'r') as label_path_file:
                label = json.load(label_path_file)
            self.cache[episode_index] = label
        frame_index = str(frame_index)
        if frame_index not in self.cache[episode_index]:
            return "slow"
        return self.cache[episode_index][str(frame_index)]

class Exp2Label(GetLabel):
    cache: dict[int, dict[int, str]] = dict()
    def __call__(self, episode_index: int, frame_index: int) -> str:
        if episode_index not in self.cache:
            label_path = Path(f"./result.json")
            with open(label_path, 'r') as label_path_file:
                label = json.load(label_path_file)
            self.cache[episode_index] = label
        frame_index = str(frame_index)
        if frame_index not in self.cache[episode_index]:
            return "slow"
        return self.cache[episode_index][str(frame_index)]

@dataclass(frozen=True)
class Config:
    get_label: GetLabel
    repo_id: str
    episode_index: int = tyro.MISSING
    output_dir: Path = tyro.MISSING

_CONFIGS = {
        "exp1": ("basic experiment", Config(get_label=Exp1Label(), repo_id = "physical-intelligence/libero")),
        "exp2": ("basic experiment", Config(get_label=Exp2Label(), repo_id = "physical-intelligence/libero")),
        }



from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v3 as iio

def draw_label_on_pil(
    img: Image.Image,
    text: Optional[str],
    xy=(10, 10),
    font_path: Optional[str] = None,
    font_size: int = 28,
    padding: int = 6,
    box_color=(0, 0, 0, 160),  # 반투명 배경
    text_fill=(255, 255, 255, 255),
    stroke_width: int = 0,
    stroke_fill: tuple = (0, 0, 0, 255),
) -> Image.Image:
    """PIL 이미지 왼쪽 위에 라벨 텍스트 박스를 그려 반환."""
    if not text:
        return img

    # RGBA 보장
    if img.mode != "RGBA":
        base = img.convert("RGBA")
    else:
        base = img.copy()

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 폰트 (한글 라벨 가능하면 DejaVuSans/나눔고딕 등 지정)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # 텍스트 크기 계산
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]

    x, y = xy
    # 배경 박스
    box = (x - padding, y - padding, x + tw + padding, y + th + padding)
    draw.rectangle(box, fill=box_color)

    # 텍스트
    draw.text(
        (x, y),
        text,
        font=font,
        fill=text_fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    # 합성
    return Image.alpha_composite(base, overlay).convert("RGB")

def make_video_with_labels_imageio(
    frames: List[Image.Image],
    out_path: str,
    fps: int = 24,
    font_path: Optional[str] = None,
):
    """frames[i]에 labels.get(i) 라벨을 그려 out_path 동영상(mp4)로 저장."""
    # imageio는 RGB numpy 배열(HxWx3) 필요
    writer = iio.imopen(out_path, "w", plugin="FFMPEG", fps=fps, codec="libx264", pix_fmt="yuv420p")
    with writer:
        for frame in frames:
            arr = np.asarray(frame)  # RGB
            writer.write(arr)


def make_video_with_labels_cv2(
    frames: List[Image.Image],
    out_path: str,
    fps: int = 24,
    font_path: Optional[str] = None,  # 사용 안 함(인터페이스 유지)
):
    assert len(frames) > 0, "frames is empty"

    # 기준 해상도 (첫 프레임)
    w, h = frames[0].width, frames[0].height

    # mp4 인코더 (환경에 따라 'avc1'이 더 호환될 수 있음)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    try:
        for im in frames:
            if im.size != (w, h):
                im = im.resize((w, h), Image.BICUBIC)

            # PIL(RGB) -> NumPy -> BGR
            arr = np.asarray(im.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
    finally:
        vw.release()


def main(config: Config):
    dataset = LeRobotDataset(config.repo_id, episodes=[config.episode_index])
    video_frames = list()
    for frame in dataset:
        image = to_pil_image(frame["image"])
        episode_index = frame["episode_index"].item()
        frame_index = frame["frame_index"].item()
        label = config.get_label(episode_index=episode_index, frame_index=frame_index)
        video_frames.append(draw_label_on_pil(to_pil_image(frame["image"]), label))
    make_video_with_labels_cv2(
            frames=video_frames,
            out_path=config.output_dir / f"{config.episode_index:04d}.mp4",
            fps = 24,
            font_path = None,
            )


if __name__=="__main__":
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    main(config)
