import os
import sys
from gps_info import GPSInfo, DroneType, SelfMadeDrone, DJI, DJIMini4K, DroneInteface
from image import get_drone, GSD_calculator, get_image_corner_coordinates, latlon_to_pixel
import folium
import math
from shapely.geometry import Polygon
import pickle
from PIL import ImageDraw, Image

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

    #def create

    def get_trace_map(self):
        # Initialize the map centered on the first coordinate
        m = folium.Map(location=self.trace[0], zoom_start=15)

        # Add the GPS trace as a polyline
        folium.PolyLine(
            locations=self.trace,
            color='blue',
            weight=5,
            opacity=0.8,
            tooltip='GPS Trace'
        ).add_to(m)

        # Add a marker for each point in the GPS trace
        for idx, coord in enumerate(self.trace, start=1):
            folium.Marker(
                location=coord,
                popup=f'Point {idx}',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        if self.unified_polygon is not None:
            # Extract (lat, lon) pairs from Shapely Polygon (lon,lat) coords
            unified_coords = [(lat, lon) for lon, lat in self.unified_polygon.exterior.coords]
            folium.Polygon(
            locations=unified_coords,
            color='green',
            fill=True,
            fill_opacity=0.3,
            popup='Unified Footprint'
            ).add_to(m)

        # Save the map to an HTML file
        m.save('gps_trace_map.html')

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

    def compare_outer_form(self, new_poly, new_gps_info):

        for idx, existing in enumerate(self.polygons):
            if new_poly.intersects(existing):
                intersection = new_poly.intersection(existing)
                # extract for both images, use unique filenames:
                coords_int = [(lat, lon) for lon, lat in intersection.exterior. coords]
                self.extract_shape(new_gps_info, coords_int, "new_overlap.png")
                info_old = self.informations[idx]
                self.extract_shape(info_old,     coords_int, "old_overlap.png")

    def check_overlapping_unique(self, external_unique_poly):
        if self.unified_polygon is None:
            return False
        # Check if the external unique polygon overlaps with the unified polygon
        if self.unified_polygon.intersects(external_unique_poly):
            return True
        return False

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
    
    trace_creator.get_trace_map()


    #with open('trace_creator_state.pkl', 'wb') as f:
    #    pickle.dump(trace_creator, f)
    #print("Saved trace_creator to disk.")