# imagery_io.py
from PIL import Image
import folium
from typing import List, Tuple

def load_image(path: str) -> Image.Image:
    return Image.open(path)

def add_corners_to_map(m: folium.Map,
                       corners: List[Tuple[float,float]],
                       color: str="blue") -> None:
    folium.Polygon(locations=corners,
                   color=color,
                   fill=True,
                   fill_opacity=0.5).add_to(m)
