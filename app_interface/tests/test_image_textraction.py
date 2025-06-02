import cv2
import numpy as np
from shapely.geometry import Polygon, box
from PIL import Image, ImageDraw


def detect_and_match_keypoints(img1: np.ndarray, img2: np.ndarray, 
                                max_features: int = 5000) -> tuple:
    """
    Detect ORB keypoints and descriptors, then match them between two images.
    Returns matched points in both images.
    """
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    raw_matches = matcher.match(des1, des2)
    matches = sorted(raw_matches, key=lambda m: m.distance)[:100]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    return pts1, pts2


def compute_homography(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Compute homography matrix using RANSAC.
    """
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H


def polygon_from_image_bounds(img: np.ndarray, H: np.ndarray) -> Polygon:
    """
    Project the corners of img through H into the second image frame
    and return as a Shapely polygon.
    """
    h, w = img.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1,2)
    return Polygon(warped)


def crop_to_polygon(img_path: str, overlap_poly: Polygon) -> Image.Image:
    """
    Crop the region defined by overlap_poly from img_path and return the cropped PIL image.
    """
    pil = Image.open(img_path).convert("RGBA")
    mask = Image.new('L', pil.size, 0)
    coords = [(int(x), int(y)) for x,y in overlap_poly.exterior.coords]
    ImageDraw.Draw(mask).polygon(coords, outline=1, fill=255)
    result = Image.new('RGBA', pil.size)
    result.paste(pil, mask=mask)
    crop = result.crop(mask.getbbox())
    return crop


def pad_to_common_size(im1: Image.Image, im2: Image.Image) -> tuple:
    """
    Pad two PIL images to the same dimensions, returning the padded images.
    """
    w = max(im1.width, im2.width)
    h = max(im1.height, im2.height)
    canvas1 = Image.new('RGBA', (w,h), (0,0,0,0))
    canvas2 = Image.new('RGBA', (w,h), (0,0,0,0))
    canvas1.paste(im1, (0,0))
    canvas2.paste(im2, (0,0))
    return canvas1, canvas2


def extract_overlap(img_path1: str, img_path2: str, out1: str = 'overlap1.png', out2: str = 'overlap2.png') -> None:
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    pts1, pts2 = detect_and_match_keypoints(img1, img2)
    H = compute_homography(pts1, pts2)

    poly1 = polygon_from_image_bounds(img1, H)
    poly2 = box(0, 0, img2.shape[1], img2.shape[0])
    overlap = poly1.intersection(poly2)
    if overlap.is_empty:
        raise ValueError('No overlapping region.')

    crop1 = crop_to_polygon(img_path1, overlap)
    crop2 = crop_to_polygon(img_path2, overlap)
    final1, final2 = pad_to_common_size(crop1, crop2)

    final1.save(out1)
    final2.save(out2)


# Example usage
if __name__ == '__main__':
    # Replace with actual paths
    img_path1 = 'raw_images/test_home/DJI_0088.JPG'
    img_path2 = 'raw_images/overlap_test_home/DJI_0093.JPG'
    extract_overlap(img_path1, img_path2)
