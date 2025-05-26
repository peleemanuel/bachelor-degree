import sys
import folium

from gps_info import DroneType
from image import get_drone
from trace_creating import TraceCreator, compare_overlapping_zones 

def process_trace(folder_path: str, drone):
    tc = TraceCreator(folder_path, drone)
    tc.generate_informations()
    tc.generate_trace()
    tc.calculate_points_direction()
    tc.get_coords_for_each_image()
    tc.get_one_polygon()
    return tc

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path_to_image_folder1> <path_to_image_folder2>")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    # Set the drone type
    drone_type = DroneType.DJI_MINI_4K
    drone = get_drone(drone_type)

    # Process the traces from the provided folders
    trace1 = process_trace(folder1, drone)
    trace2 = process_trace(folder2, drone)

    cent1 = trace1.unified_polygon.centroid
    cent2 = trace2.unified_polygon.centroid
    center_lat = (cent1.y + cent2.y) / 2
    center_lon = (cent1.x + cent2.x) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    trace1.get_trace_map(m)
    trace2.get_trace_map(m)
    out_map = "trace_comparison_map.html"
    m.save(out_map)
    print(f"Combined Folium map saved to {out_map}")
    if trace1.check_overlapping_unique(trace2.unified_polygon):
        # Compare the overlapping zones
        compare_overlapping_zones(trace1, trace2, figwidth=6, figheight=4)
