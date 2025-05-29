import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from .trace import TraceCreator
from PIL import ImageDraw, Image
from shapely.geometry import Polygon, MultiPolygon, box
import detectree as dtr
from matplotlib.widgets import Button

def extract_overlap(img_path1, img_path2, out1='overlap_1.png', out2='overlap_2.png'):
    """
    Detects the overlapping region between two images (img1 → img2 via homography),
    then crops and pads both images so that the overlap area is saved into two
    same-sized PNG files: out1 (from img1) and out2 (from img2).
    """
    # 1) Load images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # 2) Detect ORB keypoints & descriptors
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 3) Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:100]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # 4) Compute homography (img1 → img2)
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # 5) Project img1 corners into img2 frame
    h1, w1 = img1.shape[:2]
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners1, H).reshape(-1,2)

    # 6) Build & clean Shapely polygon for img1 footprint
    raw_poly1 = Polygon(warped)
    if not raw_poly1.is_valid:
        cleaned = raw_poly1.buffer(0)
        if isinstance(cleaned, MultiPolygon):
            # keep the largest piece
            raw_poly1 = max(cleaned, key=lambda p: p.area)
        else:
            raw_poly1 = cleaned

    # 7) Define img2 full-frame polygon & intersect
    poly2 = box(0, 0, img2.shape[1], img2.shape[0])
    overlap = raw_poly1.intersection(poly2)
    if overlap.is_empty:
        print("No overlapping region found.")
        return

    # 8) Crop overlap from img2
    pil2 = Image.open(img_path2).convert("RGBA")
    coords2 = [(int(x), int(y)) for x, y in overlap.exterior.coords]
    mask2 = Image.new('L', pil2.size, 0)
    ImageDraw.Draw(mask2).polygon(coords2, outline=1, fill=255)
    canvas2 = Image.new('RGBA', pil2.size)
    canvas2.paste(pil2, mask=mask2)
    crop2 = canvas2.crop(mask2.getbbox())

    # 9) Crop overlap from img1 via inverse homography
    inv_H = np.linalg.inv(H)
    pts_overlap = np.float32(overlap.exterior.coords).reshape(-1,1,2)
    warped_back = cv2.perspectiveTransform(pts_overlap, inv_H).reshape(-1,2)
    coords1 = [(int(x), int(y)) for x, y in warped_back]
    pil1 = Image.open(img_path1).convert("RGBA")
    mask1 = Image.new('L', pil1.size, 0)
    ImageDraw.Draw(mask1).polygon(coords1, outline=1, fill=255)
    canvas1 = Image.new('RGBA', pil1.size)
    canvas1.paste(pil1, mask=mask1)
    crop1 = canvas1.crop(mask1.getbbox())

    # 10) Compute common dimensions & pad crops
    w_common = max(crop1.width, crop2.width)
    h_common = max(crop1.height, crop2.height)

    final1 = Image.new('RGBA', (w_common, h_common), (0,0,0,0))
    final1.paste(crop1, (0,0))
    final2 = Image.new('RGBA', (w_common, h_common), (0,0,0,0))
    final2.paste(crop2, (0,0))

    # 11) Save results
    final1.save(out1)
    final2.save(out2)
    print(f"Overlap saved as '{out1}' and '{out2}' ({w_common}x{h_common})")
    return final1, final2
def compare_overlapping_zones(
    t1: TraceCreator,
    t2: TraceCreator,
    figwidth: float = 6,
    figheight: float = 6,
    min_area: int = 100,
    ssim_thresh: float = 0.8,
    iou_thresh: float = 0.9
):
    # 1) ensure footprints exist
    if not t1.overlaps(t2):
        print("No overlap between the two traces.")
        return []

    figs = []

    # 2) scan for a tile‐to‐tile overlap
    for i1, p1 in enumerate(t1.polygons):
        for i2, p2 in enumerate(t2.polygons):
            if p1.intersects(p2):
                # 3) crop overlap
                pil1, pil2 = extract_overlap(
                    t1.infos[i1].path(),
                    t2.infos[i2].path(),
                    out1="overlap_1.png",
                    out2="overlap_2.png"
                )

                # 4) run Detectree
                m1 = dtr.Classifier().predict_img("overlap_1.png")
                m2 = dtr.Classifier().predict_img("overlap_2.png")

                # 5) compute filtered diff
                diff = m2.astype(np.int8) - m1.astype(np.int8)
                raw = (diff != 0).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
                clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
                n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(clean)
                filt = np.zeros_like(clean)
                for lbl in range(1, n_lbl):
                    if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                        filt[lbls == lbl] = 1
                cnts, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    x,y,w,h = cv2.boundingRect(c)
                    ar = w/float(h) if h else 0
                    if not (0.5 < ar < 2.0):
                        cv2.drawContours(filt, [c], -1, 0, cv2.FILLED)
                _, s_map = ssim(m1.astype(float), m2.astype(float), full=True, data_range=1.0)
                struct_mask = (s_map < ssim_thresh).astype(np.uint8)
                final_mask = filt & struct_mask

                # 6) IoU skip
                union = np.logical_or(m1, m2).sum()
                iou = final_mask.sum()/union if union else 0
                if iou >= iou_thresh:
                    print(f"IoU={iou:.2f} ≥ {iou_thresh}: skipping small change.")
                    continue

                lat = t1.infos[i1].latitude()
                lon = t1.infos[i1].longitude()

                figs.append({
                    "crop1": pil1,
                    "crop2": pil2,
                    "mask1": m1,
                    "mask2": m2,
                    "diff": final_mask,
                    "lat": lat,
                    "lon": lon
                })

    return figs