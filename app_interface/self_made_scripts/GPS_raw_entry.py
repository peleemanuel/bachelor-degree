import os
import ast

class GPSRawInt:
    def __init__(self, data: dict):
        # Store raw values
        self.time_usec = data.get("time_usec", 0)
        self.fix_type = data.get("fix_type", 0)
        self.lat = data.get("lat", 0)
        self.lon = data.get("lon", 0)
        self.alt = data.get("alt", 0)
        self.eph = data.get("eph", 0)
        self.epv = data.get("epv", 0)
        self.vel = data.get("vel", 0)
        self.cog = data.get("cog", 0)
        self.satellites_visible = data.get("satellites_visible", 0)
        self.alt_ellipsoid = data.get("alt_ellipsoid", 0)
        self.h_acc = data.get("h_acc", 0)
        self.v_acc = data.get("v_acc", 0)
        self.vel_acc = data.get("vel_acc", 0)
        self.hdg_acc = data.get("hdg_acc", 0)
        self.yaw = data.get("yaw", 0)

    @property
    def latitude_deg(self) -> float:
        return self.lat / 1e7

    @property
    def longitude_deg(self) -> float:
        return self.lon / 1e7

    @property
    def altitude_m(self) -> float:
        return self.alt / 1000.0

    @property
    def horizontal_accuracy_m(self) -> float:
        return self.h_acc / 1000.0

    @property
    def vertical_accuracy_m(self) -> float:
        return self.v_acc / 1000.0

    @property
    def speed_m_s(self) -> float:
        return self.vel / 100.0

    @property
    def course_deg(self) -> float:
        return self.cog / 100.0

    @property
    def time_s(self) -> float:
        return self.time_usec / 1e6

    def __repr__(self):
        return (
            f"GPSRawInt(time={self.time_s:.2f}s, fix={self.fix_type}, "
            f"lat={self.latitude_deg:.7f}°, lon={self.longitude_deg:.7f}°, "
            f"alt={self.altitude_m:.2f}m, sats={self.satellites_visible})"
        )
    
def parse_capture_file(folder_path: str):
    gps_objects = []

    # Construct full path to capture_gps_info.txt
    file_path = os.path.join(folder_path, "capture_gps_info.txt")

    if not os.path.isfile(file_path):
        print(f"Error: '{file_path}' not found or is not a file.")
        return gps_objects

    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Each line is like: [0] [GPS_RAW_INT] { ... }
            # Find the first '{' and last '}' to extract the dict literal
            start = line.find('{')
            end = line.rfind('}')
            if start == -1 or end == -1:
                print(f"[Warning] Line {line_number}: No JSON-like dict found. Skipping.")
                continue

            dict_str = line[start:end+1]

            try:
                # Safely parse the dictionary literal
                data = ast.literal_eval(dict_str)  # 
                # Instantiate a GPSRawInt object
                gps_obj = GPSRawInt(data)
                gps_objects.append(gps_obj)
            except (ValueError, SyntaxError) as e:
                print(f"[Error] Line {line_number}: Failed to parse dict: {e}. Skipping.")
                continue

    return gps_objects