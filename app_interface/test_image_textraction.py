import cv2
import numpy as np
from shapely.geometry import Polygon, box
from PIL import Image, ImageDraw

def extract_overlap(img_path1, img_path2, out1='overlap1.png', out2='overlap2.png'):
    # 1) Încarcă imaginile
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # 2) Detectează puncte-cheie + descrieri cu ORB
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 3) Potrivește descriptorii
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:100]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # 4) Calculează homografia
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # 5) Proiectează colțurile lui img1 în coordonate img2
    h1, w1 = img1.shape[:2]
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners1, H).reshape(-1,2)

    # 6) Construiește poligoane Shapely și calculează intersecția
    poly1 = Polygon(warped)
    poly2 = box(0, 0, img2.shape[1], img2.shape[0])
    overlap = poly1.intersection(poly2)

    if overlap.is_empty:
        print("Nu s-a găsit nicio zonă de suprapunere.")
        return

    # 7) Crează mască & decupează pentru img2
    pil2 = Image.open(img_path2).convert("RGBA")
    coords2 = [(int(x), int(y)) for x, y in overlap.exterior.coords]
    mask2 = Image.new('L', pil2.size, 0)
    ImageDraw.Draw(mask2).polygon(coords2, outline=1, fill=255)
    canvas2 = Image.new('RGBA', pil2.size)
    canvas2.paste(pil2, mask=mask2)
    crop2 = canvas2.crop(mask2.getbbox())

    # 8) Crează mască & decupează pentru img1 (inversează homografia)
    inv_H = np.linalg.inv(H)
    corners2 = np.float32(overlap.exterior.coords).reshape(-1,1,2)
    warped_back = cv2.perspectiveTransform(corners2, inv_H).reshape(-1,2)
    coords1 = [(int(x), int(y)) for x, y in warped_back]

    pil1 = Image.open(img_path1).convert("RGBA")
    mask1 = Image.new('L', pil1.size, 0)
    ImageDraw.Draw(mask1).polygon(coords1, outline=1, fill=255)
    canvas1 = Image.new('RGBA', pil1.size)
    canvas1.paste(pil1, mask=mask1)
    crop1 = canvas1.crop(mask1.getbbox())

    # 9) Calculează dimensiunile comune
    w_common = max(crop1.width, crop2.width)
    h_common = max(crop1.height, crop2.height)

    # 10) Pad-uiește cele două crop-uri pe canvase de aceeași dimensiune
    final1 = Image.new('RGBA', (w_common, h_common), (0,0,0,0))
    final1.paste(crop1, (0,0))
    final2 = Image.new('RGBA', (w_common, h_common), (0,0,0,0))
    final2.paste(crop2, (0,0))

    # 11) Salvează rezultatele
    final1.save(out1)
    final2.save(out2)
    print(f"Overlap salvat în '{out1}' și '{out2}' cu dimensiuni {w_common}×{h_common}")

if __name__ == "__main__":
    extract_overlap('image_1.JPG', 'image_2.JPG')