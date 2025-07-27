import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = "[Path of your vision tower model]"
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
