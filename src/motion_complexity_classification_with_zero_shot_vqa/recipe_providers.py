from vqa_pipeline.visualisation import *
from typing import TypeAlias
from dataclasses import dataclass

t_label: TypeAlias = str

@dataclass(frozen=True)
class BoundingBox:
    boxes: tuple[float, float, float, float]
    score: float

class BoundingboxRecipeProvider(RecipeProvider):
    boundingbox_by_episode: dict[t_episode_index, dict[t_frame_index, Recipe]] = dict()
    def __call__(self, episode_index: t_episode_index, frame_index: t_frame_index) -> Recipe:
        if episode_index not in self.boundingbox_by_episode:
            bounding_boxes_path = Path("bounding_boxes") / f"gdino_ep{episode_index:04d}.json"
            with open(bounding_boxes_path, "r") as file:
                bounding_boxes_dict = json.load(file)
            def frame2recipe(frame_index: t_frame_index) -> Recipe:
                frame = bounding_boxes_dict[str(frame_index)]
                bounding_boxes = frame["bounding_boxes"]
                def boundingbox2Box(boundingbox: dict[t_label, BoundingBox]) -> Box:
                    label = next(iter(boundingbox.keys()))
                    boxes = boundingbox[label]["boxes"]
                    return Box(minimum=Point(x=boxes[0], y=boxes[1]), maximum=Point(x=boxes[2], y=boxes[3]), label = label)
                return Recipe(boxes = list(map(boundingbox2Box, bounding_boxes)))
            frame_indices = list(map(int, bounding_boxes_dict))
            boundingbox_by_frame = { frame_index: frame2recipe(frame_index) for frame_index in frame_indices }
            self.boundingbox_by_episode[episode_index] = boundingbox_by_frame
        return self.boundingbox_by_episode[episode_index][frame_index]

if __name__=="__main__":
    boundingbox_recipe_provider = BoundingboxRecipeProvider()
    frame_0009_000 = boundingbox_recipe_provider(9, 0)
    frame_0009_001 = boundingbox_recipe_provider(9, 1)
    print(frame_0009_000)
    print(frame_0009_001)
