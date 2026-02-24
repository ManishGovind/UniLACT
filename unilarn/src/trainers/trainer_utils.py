import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os,torch
import torchvision.transforms as T
import numpy as np



def  to_pil(t: torch.Tensor) -> Image.Image:
    return T.ToPILImage()(t)

def visualize_latent_motion_reconstruction(
    initial_frame=None,
    next_frame=None,
    recons_next_frame=None,
    initial_depth=None,
    next_depth=None,
    recons_next_depth_frame=None,
    latent_action_ids=None,
    path=""
):
    
    
    
    panels: List[Tuple[str, Image.Image]] = []

    if initial_frame is not None:
        panels.append(("First RGB",   to_pil(initial_frame)))
    if next_frame is not None:
        panels.append(("next RGB",   to_pil(next_frame)))
    if recons_next_frame is not None:
        panels.append(("recon RGB",  to_pil(recons_next_frame)))

    if initial_depth is not None:
        panels.append(("First Depth",    to_pil(initial_depth)))
    if next_depth is not None:
        panels.append(("next Depth",     to_pil(next_depth)))
    if recons_next_depth_frame is not None:
        panels.append(("recon Depth",    to_pil(recons_next_depth_frame)))
    
    if not panels:
        raise ValueError("No images provided!")
    
    w, h = panels[0][1].size
    h = h + 30
    
    # print(panels)
    
    latent_action_ids = latent_action_ids.numpy().tolist()
    n = len(panels)
    compare_img = Image.new('RGB', size=(n*w, h))
    draw_compare_img = ImageDraw.Draw(compare_img)
    for col, (label, im) in enumerate(panels):
        compare_img.paste(im, box=(col*w, 0))

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    draw_compare_img.text((w, h-20), f"{latent_action_ids}", font=font, fill=(0, 255, 0))
    compare_img.save(path)
