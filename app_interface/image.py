#!/usr/bin/env python3
import os
from PIL import Image

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
photos = [
    {'lat': 45.7525, 'lon': 21.20, 'alt': 180, 'path': 'img1.jpeg'},
    {'lat': 45.753, 'lon': 21.20, 'alt': 185, 'path': 'img2.jpeg'},
    {'lat': 45.7535, 'lon': 21.20, 'alt': 175, 'path': 'img3.jpeg'},
    # …etc
]

THUMB_SIZE = (120, 120)
SCALE = 80000           # degrees → pixels; bump up if points cluster very tightly
MARGIN = THUMB_SIZE[0]  # give equal margin on all sides
BG_COLOR = (240, 240, 240)
OUTPUT_FILE = 'photo_map.png'


def make_photo_map(photo_list, scale, thumb_size, margin, bg_color, out_path):
    # 1) compute bounding box
    lats = [p['lat'] for p in photo_list]
    lons = [p['lon'] for p in photo_list]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # 2) canvas dimensions (add margin)
    width  = int((max_lon - min_lon) * scale) + margin * 2
    height = int((max_lat - min_lat) * scale) + margin * 2
    canvas = Image.new('RGB', (width, height), color=bg_color)

    # 3) place thumbnails
    for info in photo_list:
        path = info['path']
        if not os.path.isfile(path):
            print(f"⚠️ Warning: file not found: {path}")
            continue

        img = Image.open(path)
        img.thumbnail(thumb_size)

        # compute pixel coords (add margin so nothing gets clipped)
        x = int((info['lon'] - min_lon) * scale) + margin
        y = int((max_lat - info['lat']) * scale) + margin

        # center the thumbnail on (x,y)
        paste_x = x - img.width  // 2
        paste_y = y - img.height // 2

        canvas.paste(img, (paste_x, paste_y))

    # 4) save AND preview
    canvas.save(out_path)
    print(f"✅ Map saved to {out_path}")
    canvas.show()


if __name__ == '__main__':
    make_photo_map(photos, SCALE, THUMB_SIZE, MARGIN, BG_COLOR, OUTPUT_FILE)
