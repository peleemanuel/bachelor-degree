import os
import ast

class GPSRawInt:
    def __init__(self, data: dict):
        """
        Initialize from a dictionary produced by msg.to_dict() on a GPS_RAW_INT message.
        
        Args:
            data (dict): Dictionary containing all GPS_RAW_INT fields.
        """
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
        """
        Convert raw latitude (1E-7 degrees) to decimal degrees.
        
        Returns:
            float: Latitude in degrees (e.g., 46.2865373).
        """
        return self.lat / 1e7

    @property
    def longitude_deg(self) -> float:
        """
        Convert raw longitude (1E-7 degrees) to decimal degrees.
        
        Returns:
            float: Longitude in degrees (e.g., 21.3863903).
        """
        return self.lon / 1e7

    @property
    def altitude_m(self) -> float:
        """
        Convert raw altitude (millimeters) to meters.
        
        Returns:
            float: Altitude in meters (e.g., 128.7).
        """
        return self.alt / 1000.0

    @property
    def horizontal_accuracy_m(self) -> float:
        """
        Convert horizontal accuracy (millimeters) to meters.
        
        Returns:
            float: Horizontal accuracy in meters.
        """
        return self.h_acc / 1000.0

    @property
    def vertical_accuracy_m(self) -> float:
        """
        Convert vertical accuracy (millimeters) to meters.
        
        Returns:
            float: Vertical accuracy in meters.
        """
        return self.v_acc / 1000.0

    @property
    def speed_m_s(self) -> float:
        """
        Convert ground speed (cm/s) to m/s.
        
        Returns:
            float: Speed in meters per second.
        """
        return self.vel / 100.0

    @property
    def course_deg(self) -> float:
        """
        Convert course over ground (cdeg) to degrees.
        
        Returns:
            float: Course over ground in degrees.
        """
        return self.cog / 100.0

    @property
    def time_s(self) -> float:
        """
        Convert time_usec (microseconds) to seconds.
        
        Returns:
            float: Time since boot (or GPS epoch) in seconds.
        """
        return self.time_usec / 1e6

    def __repr__(self):
        """
        Return a concise string representation with key fields.
        """
        return (
            f"GPSRawInt(time={self.time_s:.2f}s, fix={self.fix_type}, "
            f"lat={self.latitude_deg:.7f}°, lon={self.longitude_deg:.7f}°, "
            f"alt={self.altitude_m:.2f}m, sats={self.satellites_visible})"
        )
    
def parse_capture_file(folder_path: str):
    """
    Given a folder path, locate 'capture_gps_info.txt', parse each line,
    and return a list of GPSRawInt objects.

    Args:
        folder_path (str): Path to the folder containing 'capture_gps_info.txt'.

    Returns:
        List[GPSRawInt]: List of instantiated GPSRawInt objects.
    """
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