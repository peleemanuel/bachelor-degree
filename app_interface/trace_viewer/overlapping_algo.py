import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def compare_overlapping_zones(
    t1: TraceCreator,
    t2: TraceCreator,
    figwidth: float = 4,
    figheight: float = 4,
    min_area: int = 100,
    ssim_thresh: float = 0.8,
    iou_thresh: float = 0.9
):
    """
    Find the first pair of overlapping image‐tiles between two traces,
    crop their overlap (via extract_overlap), run Detectree, and filter
    false positives by area / shape / SSIM / IoU, then plot.
    """
    # ensure footprints exist
    if not t1.overlaps(t2):
        print("No overlap between the two traces.")
        return

    # scan for a tile‐to‐tile overlap
    for i1, p1 in enumerate(t1.polygons):
        for i2, p2 in enumerate(t2.polygons):
            if p1.intersects(p2):
                # 1) crop overlap into overlap1.png & overlap2.png
                extract_overlap(
                    t1.infos[i1].path,
                    t2.infos[i2].path,
                    out1="overlap1.png",
                    out2="overlap2.png"
                )

                # 2) load masks
                y1 = dtr.Classifier().predict_img("overlap1.png")
                y2 = dtr.Classifier().predict_img("overlap2.png")

                # 3) raw diff mask
                diff = (y2.astype(np.int8) - y1.astype(np.int8))
                raw = (diff != 0).astype(np.uint8)

                # 4) morph clean + area filter
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
                clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

                _, lbls, stats, _ = cv2.connectedComponentsWithStats(clean)
                filt = np.zeros_like(clean)
                for lbl in range(1, stats.shape[0]):
                    if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                        filt[lbls == lbl] = 1

                # 5) drop elongated
                cnts, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    x,y,w,h = cv2.boundingRect(c)
                    ar = w/float(h) if h>0 else 0
                    if not (0.5 < ar < 2.0):
                        cv2.drawContours(filt, [c], -1, 0, cv2.FILLED)

                # 6) SSIM filter
                _, smap = ssim(y1.astype(float), y2.astype(float), full=True, data_range=1.0)
                struct = (smap < ssim_thresh).astype(np.uint8)

                final = filt & struct

                # 7) IoU skip
                union = np.logical_or(y1, y2).sum()
                iou = final.sum()/union if union>0 else 0
                if iou >= iou_thresh:
                    print(f"IoU={iou:.2f} ≥ {iou_thresh}: skipping small change.")
                    return

                # 8) report & plot
                print(f"Significant change: IoU={iou:.2f}")
                fig, ax = plt.subplots(figsize=(figwidth,figheight))
                ax.imshow(final, cmap='gray')
                ax.set_title(f"Filtered Change (IoU={iou:.2f})")
                ax.axis('off')
                plt.show()
                return
    print("No individual tile overlaps the combined footprint intersection.")
