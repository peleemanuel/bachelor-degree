import os
import sys
import numpy as np
import cv2

from gps_info import GPSInfo, DroneType, SelfMadeDrone, DJI, DJIMini4K, DroneInteface
from image import get_drone, GSD_calculator, get_image_corner_coordinates, latlon_to_pixel
import folium
import math
from shapely.geometry import Polygon, MultiPolygon, box
import pickle
from PIL import ImageDraw, Image

import detectree as dtr
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import plot
from skimage.metrics import structural_similarity as ssim


class TraceCreator:
    def __init__(self, folder_path: str, drone: DroneInteface):
        self.folder_path = folder_path
        self.informations = []
        self.drone = drone
        self.trace = []
        self.coords_for_each_image = []
        self.polygons = []
        self.unified_polygon = None

    def generate_informations(self):
        # Given folder path should contain 2 images at least otherwise it will not work
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"'{self.folder_path}' is not a valid directory.")
        # Get all the files in the folder
        files = os.listdir(self.folder_path)
        # Filter the files to get only the images
        images = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.JPG', '.JPEG', '.PNG', '.TIFF', '.BMP'))]
        # Check if there are at least 2 images
        if len(images) < 2:
            raise ValueError(f"'{self.folder_path}' should contain at least 2 images.")
        
        # Loop through the images and create the GPSInfo objects
        for image in images:
            # Create the GPSInfo object
            # TODO: Add the secondary file path for self-made drone 
            gps_info = GPSInfo(os.path.join(self.folder_path, image), self.drone.drone_type)
            # Extract the GPS info from the image
            gps_info.extract_gps_info()
            # Append the GPSInfo object to the list
            self.informations.append(gps_info)

    def generate_trace(self):
        # Check if the informations list is not empty
        if not self.informations:
            raise ValueError("No GPS information found.")
        
        # Create the trace
        trace = []
        for gps_info in self.informations:
            # Get the coordinates
            lat = gps_info.lat
            lon = gps_info.lon
            # Append the coordinates to the trace
            trace.append((lat, lon))
        
        self.trace = trace

    def get_trace_map(self, m: folium.Map) -> folium.Map:
        """
        Given an existing folium.Map `m`, add this trace's path, point
        markers, and unified footprint polygon(s) onto it, then return it.
        """
        if not self.trace:
            raise ValueError("No trace coordinates available. Run generate_trace() first.")
        
        # 1) Add the GPS trace as a polyline
        folium.PolyLine(
            locations=self.trace,
            color='blue',
            weight=5,
            opacity=0.8,
            tooltip='GPS Trace'
        ).add_to(m)

        # 2) Add a marker for each point in the GPS trace
        for idx, coord in enumerate(self.trace, start=1):
            folium.Marker(
                location=coord,
                popup=f'Point {idx}',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        # 3) Draw unified footprint — handle both Polygon and MultiPolygon
        if self.unified_polygon is not None:
            # Build a list of Polygon objects
            if isinstance(self.unified_polygon, Polygon):
                polys = [self.unified_polygon]
            elif isinstance(self.unified_polygon, MultiPolygon):
                polys = list(self.unified_polygon.geoms)
            else:
                raise ValueError(f"Unsupported geometry type: {type(self.unified_polygon)}")

            # Add each polygon ring to the map
            for poly in polys:
                # shapely coords are (lon, lat) but Folium expects (lat, lon)
                unified_coords = [(lat, lon) for lon, lat in poly.exterior.coords]
                folium.Polygon(
                    locations=unified_coords,
                    color='green',
                    fill=True,
                    fill_opacity=0.3,
                    popup='Unified Footprint'
                ).add_to(m)

        return m

    def calculate_initial_bearing(self, lat_1, lon_1, lat_2, lon_2):
        # Calculate the initial bearing between two points
        lat1 = math.radians(lat_1)
        lat2 = math.radians(lat_2)
        d_lon = math.radians(lon_2 - lon_1)

        x = math.sin(d_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)

        # Normalize the bearing to 0-360 degrees
        return (initial_bearing + 360) % 360

    def calculate_points_direction(self):
        # Take each two points and calculate the direction between them

        for i in range(len(self.trace) - 1):
            lat1, lon1 = self.trace[i]
            lat2, lon2 = self.trace[i + 1]
            # Calculate the initial bearing
            initial_bearing = self.calculate_initial_bearing(lat1, lon1, lat2, lon2)
            # Append the initial bearing to the gps_info
            self.informations[i].image_direction = initial_bearing

            if i == len(self.trace) - 2:
                self.informations[i + 1].image_direction = initial_bearing
    
    def get_coords_for_each_image(self):
        # Get the coordinates for each image
        for gps_info in self.informations:
            gsd = GSD_calculator(self.drone, gps_info)
            coords = get_image_corner_coordinates(gsd, gps_info)

            self.coords_for_each_image.append(coords)
            # build a Shapely Polygon (lon,lat) for spatial ops :contentReference[oaicite:2]{index=2}
            poly = Polygon([(lon, lat) for lat, lon in coords])
            area = poly.area

            # Check for overlapping polygons and remove it
            #new_poly = self.check_overlapping_and_remove_it(poly)
            #new_poly_area = new_poly.area
            #if new_poly_area < 0.05 * area:
            #    continue
            self.polygons.append({"polygon": poly, "initial_area": area})

    def check_overlapping_and_remove_it(self, poly):
        # Fix invalid polygons before intersection checks
        if not poly.is_valid:
            poly = poly.buffer(0)
        # Check if the polygon overlaps with any of the existing polygons
        for existing in self.polygons:
            existing_poly = existing["polygon"]
            if not existing_poly.is_valid:
                existing_poly = existing_poly.buffer(0)
            if poly.intersects(existing_poly):
                # Truncate the overlapping area
                intersection = poly.intersection(existing_poly)
                # Remove the intersection area from the poly
                poly = poly.difference(intersection)

        return poly                
    
    def get_one_polygon(self):
        # Unify all polygons into one
        if self.unified_polygon is None:
            self.unified_polygon = self.polygons[0]["polygon"]
            for existing in self.polygons[1:]:
                existing_poly = existing["polygon"]
                if not existing_poly.is_valid:
                    existing_poly = existing_poly.buffer(0)
                self.unified_polygon = self.unified_polygon.union(existing_poly)
    
    def extract_shape(self, info, shape_geo, out_path):
        # open image + get GSD, gps_info
        img = Image.open(info.path).convert('RGBA')
        w, h = img.size
        gsd = GSD_calculator(self.drone, info)

        # map shape to pixels
        pixel_poly = [
            latlon_to_pixel(lat, lon, info, gsd, w, h)
            for lat, lon in shape_geo
        ]

        # create mask & extract
        mask = Image.new('L', (w,h), 0)
        ImageDraw.Draw(mask).polygon(pixel_poly, outline=1, fill=255)
        result = Image.new('RGBA', (w,h))
        result.paste(img, mask=mask)
        crop = result.crop(mask.getbbox())
        crop.save(out_path)


    def check_overlapping_unique(self, external_unique_poly):
        if self.unified_polygon is None or external_unique_poly is None:
            return False
        # Check if the external unique polygon overlaps with the unified polygon
        if self.unified_polygon.intersects(external_unique_poly):
            return True
        return False

    def update_all_polygons(self, new_unique_poly):
        """Remove all intersecting zones from the existing polygons and the unified polygon."""
        if self.unified_polygon is None or new_unique_poly is None:
            return
        
        # Remove intersecting zones from the unified polygon
        if self.unified_polygon.intersects(new_unique_poly):
            self.unified_polygon = self.unified_polygon.difference(new_unique_poly)

        # Remove intersecting zones from all existing polygons
        for existing in self.polygons:
            existing_poly = existing["polygon"]
            if not existing_poly.is_valid:
                existing_poly = existing_poly.buffer(0)
            if existing_poly.intersects(new_unique_poly):
                existing["polygon"] = existing_poly.difference(new_unique_poly)


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


def compare_overlapping_zones(trace1, trace2, figwidth=4, figheight=4,
                              min_area=100, ssim_thresh=0.8, iou_thresh=0.9):
    """
    Compare two TraceCreator instances to find and visualize
    *significant* differences in Detectree masks over their first overlapping tile.
    Filters out false positives via morphology, area, shape, SSIM, and IoU.
    """
    # 1) Ensure unified footprints exist and intersect
    if trace1.unified_polygon is None or trace2.unified_polygon is None:
        print("Error: One of the traces is missing its unified polygon.")
        return
    if not trace1.unified_polygon.intersects(trace2.unified_polygon):
        print("No overlap between the two traces.")
        return

    # 2) Find first overlapping tile
    for idx1, e1 in enumerate(trace1.polygons):
        for idx2, e2 in enumerate(trace2.polygons):
            if e1["polygon"].intersects(e2["polygon"]):
                # Crop overlap into overlap_1.png & overlap_2.png:
                extract_overlap(trace1.informations[idx1].path,
                                trace2.informations[idx2].path)

                # Load the two binary masks (0/1) from Detectree
                m1 = dtr.Classifier().predict_img("overlap_1.png")
                m2 = dtr.Classifier().predict_img("overlap_2.png")

                # 3) Raw diff mask: +1 added, -1 removed
                diff = (m2.astype(np.int8) - m1.astype(np.int8))
                raw_change = (diff != 0).astype('uint8')

                # 4) Morphological cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                clean = cv2.morphologyEx(raw_change, cv2.MORPH_OPEN, kernel)
                clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

                # 5) Connected components + area filter
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
                filtered = np.zeros_like(clean)
                for lbl in range(1, n_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if area >= min_area:
                        filtered[labels == lbl] = 1

                # 6) Shape heuristic: drop elongated regions
                cnts, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    x,y,w,h = cv2.boundingRect(cnt)
                    ar = w / float(h) if h>0 else 0
                    if not (0.5 < ar < 2.0):
                        cv2.drawContours(filtered, [cnt], -1, 0, thickness=cv2.FILLED)

                # 7) SSIM structural mask
                # Normalize m1, m2 to [0,1]
                m1f = m1.astype('float32')
                m2f = m2.astype('float32')
                _ , ssim_map = ssim(m1f, m2f, full=True, data_range=1.0)
                struct_mask = (ssim_map < ssim_thresh).astype('uint8')

                # 8) Combine morphological & structural
                final_mask = filtered & struct_mask

                # 9) IoU check
                union = np.logical_or(m1, m2).sum()
                iou   = final_mask.sum() / union if union>0 else 0
                if iou >= iou_thresh:
                    print(f"IoU={iou:.3f} ≥ {iou_thresh}: changes too small, skipping.")
                    return

                # 10) Report & visualize
                print(f"Detected significant change: IoU={iou:.3f}")
                fig, ax = plt.subplots(1,1,figsize=(figwidth,figheight))
                ax.imshow(final_mask, cmap='gray')
                ax.set_title(f"Filtered Change Mask (IoU={iou:.3f})")
                ax.axis('off')

                # Optionally show side-by-side full comparisons:
                fig2, axes = plt.subplots(1,3, figsize=(3*figwidth,figheight))
                axes[0].imshow(m1, cmap='gray'); axes[0].set_title("Mask 1")
                axes[1].imshow(m2, cmap='gray'); axes[1].set_title("Mask 2")
                im = axes[2].imshow(final_mask, cmap='viridis'); axes[2].set_title("Filtered Diff")
                fig2.colorbar(im, ax=axes[2])
                for ax in axes: ax.axis('off')
                plt.tight_layout()
                plt.show()

                return  # stop after first tile
    print("No individual tile overlaps the combined footprint intersection.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_image_folder>")
        sys.exit(1)
    folder_path = sys.argv[1]
    drone_type = DroneType.DJI_MINI_4K
    # Get the drone object
    drone = get_drone(drone_type)

    trace_creator = TraceCreator(folder_path, drone)
    trace_creator.generate_informations()
    trace_creator.generate_trace()
    trace_creator.calculate_points_direction()
    trace_creator.get_coords_for_each_image()
    trace_creator.get_one_polygon()
    
    # trace_creator.get_trace_map()


    #with open('trace_creator_state.pkl', 'wb') as f:
    #    pickle.dump(trace_creator, f)
    #print("Saved trace_creator to disk.")