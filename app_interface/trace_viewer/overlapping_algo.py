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
    Detects the overlapping region between two images (img1 - img2 via homography),
    then crops and pads both images so that the overlap area is saved into two
    same-sized PNG files: out1 (from img1) and out2 (from img2).
    """
    # Load images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Detect ORB keypoints & descriptors
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:100]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

<<<<<<< small_bug_fixes
    # Compute homography (img1 - img2)
=======
    # Compute homography (img1 → img2)
>>>>>>> main
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Project img1 corners into img2 frame
    h1, w1 = img1.shape[:2]
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners1, H).reshape(-1,2)

    # Build & clean Shapely polygon for img1 footprint
    raw_poly1 = Polygon(warped)
    if not raw_poly1.is_valid:
        cleaned = raw_poly1.buffer(0)
        if isinstance(cleaned, MultiPolygon):
            # keep the largest piece
            raw_poly1 = max(cleaned, key=lambda p: p.area)
        else:
            raw_poly1 = cleaned

    # Define img2 full-frame polygon & intersect
    poly2 = box(0, 0, img2.shape[1], img2.shape[0])
    overlap = raw_poly1.intersection(poly2)
    if overlap.is_empty:
        print("No overlapping region found.")
        return

    # Crop overlap from img2
    pil2 = Image.open(img_path2).convert("RGBA")
    coords2 = [(int(x), int(y)) for x, y in overlap.exterior.coords]
    mask2 = Image.new('L', pil2.size, 0)
    ImageDraw.Draw(mask2).polygon(coords2, outline=1, fill=255)
    canvas2 = Image.new('RGBA', pil2.size)
    canvas2.paste(pil2, mask=mask2)
    crop2 = canvas2.crop(mask2.getbbox())

    # Crop overlap from img1 via inverse homography
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

    # Compute common dimensions & pad crops
    w_common = max(crop1.width, crop2.width)
    h_common = max(crop1.height, crop2.height)

    final1 = Image.new('RGBA', (w_common, h_common), (0,0,0,0))
    final1.paste(crop1, (0,0))
    final2 = Image.new('RGBA', (w_common, h_common), (0,0,0,0))
    final2.paste(crop2, (0,0))

    # Save results
    final1.save(out1)
    final2.save(out2)
    print(f"Overlap saved as '{out1}' and '{out2}' ({w_common}x{h_common})")
    return final1, final2

def filter_changes_by_percentage(raw_diff, pct_min_area=0.002, ar_bounds=(0.75, 1.33), solidity_thresh=0.6, border_margin=5, morph_kernel_size=(5, 5)):

    H, W = raw_diff.shape
    img_area = H * W
    min_area_px = int(pct_min_area * img_area)

    # Morphological opening/closing to remove tiny noise 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    clean = cv2.morphologyEx(raw_diff, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(clean)
    filt = np.zeros_like(clean)

    # Threshold by dynamic min_area_px
    for lbl in range(1, n_lbl):
        area_lbl = stats[lbl, cv2.CC_STAT_AREA]
        if area_lbl < min_area_px:
            continue
        filt[lbls == lbl] = 1

    # Contour‐level shape filters
    cnts, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(filt)
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh) if bh > 0 else 0
        # Aspect ratio filter
        if not (ar_bounds[0] < ar < ar_bounds[1]):
            continue
        # Border contact filter
        H_mask, W_mask = filt.shape
        if (x <= border_margin or y <= border_margin or
            x + bw >= W_mask - border_margin or
            y + bh >= H_mask - border_margin):
            continue  
        # Solidity filter

        pts = c.squeeze()
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        hull_area = poly.convex_hull.area if poly.convex_hull is not None else 0
        if hull_area <= 0:
            continue
        solidity = poly.area / hull_area
        if solidity < solidity_thresh:
            continue  

        cv2.drawContours(final_mask, [c], -1, 1, cv2.FILLED)

    return final_mask


def compare_overlapping_zones(
    t1: TraceCreator,
    t2: TraceCreator,
    figwidth: float = 6,
    figheight: float = 6,
    pct_min_area: float = 0.001,
    ssim_thresh: float = 0.90,
    iou_thresh: float = 0.03,
    log_image_info: bool = False
):
    if not t1.overlaps(t2):
        print("No overlap between the two traces.")
        return []

    figs = []
    for i1, p1 in enumerate(t1.polygons):
        for i2, p2 in enumerate(t2.polygons):
            if not p1.intersects(p2):
                continue

            # Crop the overlapping region from both images
            pil1, pil2 = extract_overlap(
                t1.infos[i1].path(),
                t2.infos[i2].path(),
                out1="overlap_1.png",
                out2="overlap_2.png"
            )

            # Run Detectree (binary masks)
            classifier = dtr.Classifier()
            m1 = classifier.predict_img("overlap_1.png")
            m2 = classifier.predict_img("overlap_2.png")

            # Raw difference mask
            diff = m2.astype(np.int8) - m1.astype(np.int8)
            raw_diff = (diff != 0).astype(np.uint8)

            # SSIM Mask
            m1f = m1.astype(float)
            m2f = m2.astype(float)
            _, s_map = ssim(m1f, m2f, full=True, data_range=1.0)
            struct_mask = (s_map < ssim_thresh).astype(np.uint8)
            combined_raw = raw_diff & struct_mask

            # Filter changes by percentage‐based min_area
            final_mask = filter_changes_by_percentage(
                combined_raw,
                pct_min_area=pct_min_area,
                ar_bounds=(0.5, 2.0),
                solidity_thresh=0.4,
                border_margin=1,
                morph_kernel_size=(3, 3)
            )

            # IoU Skip
            union = np.logical_or(m1, m2).sum()
            iou = final_mask.sum() / union if union > 0 else 0
            if log_image_info:
                print("######################################")
                print(f"Comparing {t1.infos[i1].path()} with {t2.infos[i2].path()}")
                computed_ssim = ssim(m1f, m2f, data_range=m1f.max() - m1f.min())
                print(f"SSIM={ssim_thresh:.2f} (computed={computed_ssim:.2f})")
                print(f"Final mask size: {final_mask.sum()} pixels")
                print(f"Union size: {union} pixels")
                print(f"Raw diff size: {raw_diff.sum()} pixels")
                print(f"Filtered diff size: {final_mask.sum()} pixels")
                print(f"SSIM mask size: {struct_mask.sum()} pixels")
                print(f"Crop 1: {pil1.size}, Crop 2: {pil2.size}")
                print(f"Crop 1: {t1.infos[i1].path()}, Crop 2: {t2.infos[i2].path()}")
                print(f"Crop 1: {t1.infos[i1].latitude()}, {t1.infos[i1].longitude()}")
                print(f"Crop 2: {t2.infos[i2].latitude()}, {t2.infos[i2].longitude()}")
                print(f"Final mask IoU: {iou:.2f} (threshold={iou_thresh})")
                print(f"Final mask size: {final_mask.sum()} pixels")
                print(f"IoU={iou:.2f} (threshold={iou_thresh})")
                print("######################################")

            if iou < iou_thresh:
                if log_image_info:
                    print(f"IoU={iou:.2f} < {iou_thresh}: skipping small change.")
                continue

            # Record result
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
    print(f"Found {len(figs)} overlapping zones with significant changes.")
    return figs