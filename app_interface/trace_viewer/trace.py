from pathlib import Path
from typing import List, Optional

import folium
from shapely.geometry import Polygon, MultiPolygon

from .gps import ExifGPS
from .drones import DroneSpec, DroneType
from .imagery_math import calculate_gsd, image_corners, latlon_to_pixel

class TraceCreator:
    """
    Builds a GPS trace and image footprint for a folder of drone images.
    """
    def __init__(self, folder: str, drone_spec: DroneSpec):
        self.folder = Path(folder)
        self.drone = drone_spec
        self.infos: List[ExifGPS] = []
        self.trace: List[tuple[float,float]] = []
        self.polygons: List[Polygon] = []
        self.unified: Optional[Polygon] = None
        self.total_area_percent: float = 0.0

    def build(self) -> None:
        self._load_gps_info()
        self._build_trace()
        self._build_polygons()
        self._unify_polygons()

    def _load_gps_info(self) -> None:
        if not self.folder.is_dir():
            raise ValueError(f"Not a directory: {self.folder}")
        paths = sorted(self.folder.glob("*.JPG")) + sorted(self.folder.glob("*.jpg"))
        if len(paths) < 2:
            raise ValueError("Need at least two images to build trace")
        for p in paths:
            gps = ExifGPS(str(p))
            self.infos.append(gps)

    def _build_trace(self) -> None:
        for info in self.infos:
            lat, lon = info.latitude(), info.longitude()
            if lat is None or lon is None:
                raise RuntimeError(f"Missing GPS in {info.path}")
            self.trace.append((lat, lon))

    def _build_polygons(self) -> None:
        for info in self.infos:
            lat, lon = info.latitude(), info.longitude()
            direction = info.img_direction() or 0.0
            gsd = calculate_gsd(
                alt_m=info.altitude() or 0.0,
                sensor_mm=self.drone.sensor_width_mm,
                focal_mm=self.drone.focal_length_mm,
                img_px=info.img_width(),  # assume ExifGPS exposes width
            )
            corners = image_corners(
                center_lat=lat,
                center_lon=lon,
                direction_deg=direction,
                gsd_cm=gsd,
                img_width=info.img_width(),
                img_height=info.img_height(),
                adjust_for_mini4k=(self.drone.type == DroneType.MINI_4K)
            )
            poly = Polygon([(lon, lat) for (lat, lon) in corners])
            self.polygons.append(poly)

    def _unify_polygons(self) -> None:
        if not self.polygons:
            return
        u = self.polygons[0]
        for p in self.polygons[1:]:
            u = u.union(p)
        self.total_area_percent = 100.0
        self.unified = u

    def overlaps(self, other: "TraceCreator") -> bool:
        """Return True if this traces unified footprint overlaps the others."""
        if self.unified is None or other.unified is None:
            return False
        return self.unified.intersects(other.unified)

    def remove_overlap(self, other: "TraceCreator") -> float:
        """
        Subtract the overlapping area of `other` from this unified polygon,
        returning the percentage of area removed.
        """
        if not self.overlaps(other):
            return 0.0
        base_area = self.unified.area
        inter = self.unified.intersection(other.unified)
        perc = inter.area / base_area * 100.0

        self.total_area_percent -= perc / 100.0 * self.total_area_percent
        # remove it
        self.unified = self.unified.difference(other.unified)
        
        # remove also from polygons
        for poly in self.polygons:
            if poly.intersects(other.unified):
                poly = poly.difference(other.unified)

        return perc
    
    def center(self) -> tuple[float, float]:
        """
        Return the latitude/longitude to center the map on.
        """
        if self.unified is not None and not self.unified.is_empty:
            c = self.unified.centroid
            return (c.y, c.x)

        if self.trace:
            return self.trace[0]

        return (45.9432, 24.9668)

    def add_to_map(self, m: folium.Map) -> None:
        # Draw trace
        folium.PolyLine(locations=self.trace, color='blue', weight=3).add_to(m)
        for idx, coord in enumerate(self.trace, 1):
            folium.Marker(location=coord, popup=f"{idx}").add_to(m)
        # Draw footprint
        if self.unified is None:
            return
        geoms = (
            [self.unified] if isinstance(self.unified, Polygon)
            else list(self.unified.geoms)
        )
        for poly in geoms:
            coords = [(lat, lon) for lon, lat in poly.exterior.coords]
            folium.Polygon(locations=coords, color='green', fill=True, fill_opacity=0.2).add_to(m)
